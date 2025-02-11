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

# We remove local predictor/tracker usage. It's all in child_infer
parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen IP")
parser.add_argument("--port", type=int, default=12345, help="Listen Port")
args = parser.parse_args()

def cv2_to_pil(image_bgr):
    return PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

# We'll keep your global overlay variables
mask = None
point = None

latestFrame = None
latestFrameLock = threading.Lock()
RUNNING = True

childProc = None  # reference to the spawned child process

# Build an absolute path to child_infer.py
scriptDir = os.path.dirname(os.path.abspath(__file__))  # folder of main_server.py
childInferPath = os.path.join(scriptDir, "child_infer.py")

def on_mouse(event, x, y, flags, param):
    global childProc, mask, point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("[DoubleClick] => spawn child process")
        frameCopy = None
        with latestFrameLock:
            if latestFrame is not None:
                frameCopy = latestFrame.copy()
        if frameCopy is None:
            print("No frame to run child on.")
            return

        input_path = "tmp_input.jpg"
        output_path = "tmp_mask.png"
        cv2.imwrite(input_path, frameCopy)

        # kill old child if any
        if childProc is not None:
            childProc.terminate()
            childProc = None

        # We pass encoder, decoder, input, output, plus click_x, click_y
        cmd = [
            "python", childInferPath,
            args.image_encoder,
            args.mask_decoder,
            input_path,
            output_path,
            str(x), str(y)
        ]
        childProc = subprocess.Popen(cmd)
        print(f"Spawned child_infer: click=({x},{y}).")

    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[RightClick] => kill child process")
        if childProc is not None:
            childProc.terminate()
            childProc = None
            print("Child process killed.")
        # also reset local mask/point
        mask = None
        point = None

cv2.namedWindow("image")
cv2.namedWindow("mask")
cv2.setMouseCallback("image", on_mouse)

def server_thread(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"NanoSAM server listening at {host}:{port}")

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
            if not length_data or len(length_data) < 4:
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
            while to_read > 0:
                chunk = conn.recv(to_read)
                if not chunk:
                    break
                jpeg_bytes += chunk
                to_read -= len(chunk)
            if len(jpeg_bytes) < length:
                print("Client disconnected mid-frame.")
                conn.close()
                conn = None
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode.")
                continue

            with latestFrameLock:
                global latestFrame
                latestFrame = frame

        except socket.timeout:
            continue
        except Exception as e:
            print("Server error:", e)
            conn.close()
            conn = None
            continue

    if conn:
        conn.close()
    s.close()
    print("Server closed...")

def read_child_mask_if_done():
    global childProc, mask
    if childProc is None:
        return
    retcode = childProc.poll()
    if retcode is None:
        return  # still running
    # done or crashed
    childProc = None
    output_path = "tmp_mask.png"
    if not os.path.exists(output_path):
        print("[ChildInfer] done but no output mask found.")
        mask = None
        return
    new_mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    if new_mask is None:
        print("[ChildInfer] mask file invalid.")
        mask = None
        return
    mask = new_mask
    print("[ChildInfer] loaded new mask from disk.")

def handle_frame(frame_bgr):
    global mask, point

    # If childProc finished, read the resulting mask
    read_child_mask_if_done()

    disp = frame_bgr.copy()
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

def main():
    global RUNNING
    global latestFrame

    t = threading.Thread(target=server_thread, args=(args.host, args.port), daemon=True)
    t.start()

    while True:
        ret = cv2.waitKey(30)
        if ret == ord('q'):
            RUNNING = False
            break

        frameCopy = None
        with latestFrameLock:
            if latestFrame is not None:
                frameCopy = latestFrame.copy()
                latestFrame = None

        if frameCopy is not None:
            handle_frame(frameCopy)

    cv2.destroyAllWindows()
    print("Exiting main...")

if __name__=="__main__":
    main()
