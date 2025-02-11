# main_server.py
import PIL.Image
import cv2
import numpy as np
import argparse
import socket
import struct
import threading
import time
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
parser.add_argument("--host", type=str, default="0.0.0.0", help="server for VisionPro feed")
parser.add_argument("--port", type=int, default=12345, help="server port for VisionPro feed")
parser.add_argument("--child_host", type=str, default="127.0.0.1", help="child GPU server IP")
parser.add_argument("--child_port", type=int, default=55555, help="child GPU server port")
args = parser.parse_args()

# We'll store the mask from child
mask = None
point = None

# We'll store the latest frame from VisionPro
latestFrame = None
latestFrameLock = threading.Lock()
RUNNING = True

# We'll spawn a separate childInferServer
childProc = None
scriptDir = os.path.dirname(os.path.abspath(__file__))
childInferServerPath = os.path.join(scriptDir, "child_infer_server.py")

def spawn_child_infer_server():
    global childProc
    if childProc is not None:
        return
    env = os.environ.copy()
    env["ENCODER_PATH"] = args.image_encoder
    env["DECODER_PATH"] = args.mask_decoder
    childProc = subprocess.Popen(["python", childInferServerPath], env=env)
    print("[Main] Spawned child_infer_server process (model loaded).")

def kill_child_infer_server():
    global childProc
    if childProc is not None:
        print("[Main] Killing child_infer_server.")
        childProc.terminate()
        childProc = None

def ensure_child_is_running():
    global childProc
    if childProc is None or childProc.poll() is not None:
        print("[Main] Child not running. Spawning again.")
        spawn_child_infer_server()
        time.sleep(0.5)  # allow it to start listening

def send_command_to_child(cmd_str):
    """
    e.g. "INIT 200 150 tmp_input.jpg"
         "UPDATE tmp_input.jpg"
         "RESET"
    Returns mask as grayscale np.uint8, or None
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((args.child_host, args.child_port))
        cmd_bytes = cmd_str.encode('utf-8')
        header = struct.pack('<i', len(cmd_bytes))
        sock.sendall(header)
        sock.sendall(cmd_bytes)

        length_data = sock.recv(4)
        if not length_data or len(length_data)<4:
            print("[Main->Child] incomplete mask length.")
            return None
        (mask_len,) = struct.unpack('<i', length_data)
        if mask_len<=0:
            print("[Main->Child] mask_len=0 => no mask.")
            return None

        mask_bytes = b''
        to_read = mask_len
        while to_read>0:
            chunk = sock.recv(to_read)
            if not chunk:
                break
            mask_bytes += chunk
            to_read -= len(chunk)
        if len(mask_bytes)<mask_len:
            print("[Main->Child] partial mask => ignoring.")
            return None

        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_img = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        return mask_img
    except Exception as e:
        print("[Main->Child] send_command_to_child error:", e)
        return None
    finally:
        sock.close()

# These track whether we have an active track in the child
activeTrack = False

def on_mouse(event, x, y, flags, param):
    global mask, point, activeTrack
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("[DoubleClick] => INIT track in child.")
        ensure_child_is_running()

        frameCopy = None
        with latestFrameLock:
            if latestFrame is not None:
                frameCopy = latestFrame.copy()
        if frameCopy is None:
            print("[Main] No frame to run child on.")
            return

        frame_path = "tmp_input.jpg"
        cv2.imwrite(frame_path, frameCopy)

        cmd_str = f"INIT {x} {y} {frame_path}"
        new_mask = send_command_to_child(cmd_str)
        if new_mask is not None:
            mask = new_mask
            point = (x, y)
            activeTrack = True
            print("[Main] INIT done. Mask shape:", new_mask.shape)

    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[RightClick] => kill child + reset local mask.")
        kill_child_infer_server()
        mask = None
        point = None
        activeTrack = False

cv2.namedWindow("image")
cv2.namedWindow("mask")
cv2.setMouseCallback("image", on_mouse)

def server_thread():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"[Main] Listening for VisionPro feed on {args.host}:{args.port}")

    s.settimeout(0.5)
    conn = None
    while RUNNING:
        if not conn:
            try:
                conn, addr = s.accept()
                conn.settimeout(0.1)
                print("Got connection from", addr)
            except socket.timeout:
                continue

        try:
            length_data = conn.recv(4)
            if not length_data or len(length_data)<4:
                print("Client disconnected or no length data.")
                conn.close()
                conn = None
                continue

            (length,) = struct.unpack('<i', length_data)
            if length <= 0:
                print("Invalid length:", length)
                continue

            jpeg_bytes = b''
            to_read = length
            while to_read>0:
                chunk = conn.recv(to_read)
                if not chunk:
                    break
                jpeg_bytes += chunk
                to_read -= len(chunk)
            if len(jpeg_bytes)<length:
                print("Client disconnected mid-frame.")
                conn.close()
                conn = None
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode feed frame.")
                continue

            with latestFrameLock:
                global latestFrame
                latestFrame = frame

        except socket.timeout:
            continue
        except Exception as e:
            print("VisionPro feed server error:", e)
            conn.close()
            conn = None
            continue

    if conn:
        conn.close()
    s.close()
    print("[Main] Feed server stopped.")

def main():
    global RUNNING, mask, point, activeTrack
    spawn_child_infer_server()  # start child once
    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    while True:
        key = cv2.waitKey(30)
        if key == ord('q'):
            RUNNING = False
            break
        elif key == ord('r'):
            ensure_child_is_running()
            send_command_to_child("RESET")
            mask = None
            point = None
            activeTrack = False
            print("[Main] Reset done => no active track now.")

        # if we have an active track, let's do "UPDATE" for each new frame => dynamic
        frameCopy = None
        with latestFrameLock:
            if latestFrame is not None:
                frameCopy = latestFrame.copy()
                latestFrame = None

        if frameCopy is not None:
            # if activeTrack => do an "UPDATE" call
            if activeTrack:
                ensure_child_is_running()
                # Save frame for update
                frame_path = "tmp_input.jpg"
                cv2.imwrite(frame_path, frameCopy)
                new_mask = send_command_to_child(f"UPDATE {frame_path}")
                if new_mask is not None:
                    mask = new_mask

            # overlay local mask
            disp = frameCopy.copy()
            if mask is not None:
                bin_mask = (mask < 128)
                green = np.zeros_like(disp)
                green[:] = (0,185,118)
                green[bin_mask] = 0
                disp = cv2.addWeighted(disp, 0.4, green, 0.6, 0)

                obj_mask = np.logical_not(bin_mask)
                mask_disp = np.zeros_like(disp)
                mask_disp[obj_mask] = (255,255,255)
                cv2.imshow("mask", mask_disp)
            cv2.imshow("image", disp)

    kill_child_infer_server()
    cv2.destroyAllWindows()
    print("[Main] Exiting main...")

if __name__=="__main__":
    main()
