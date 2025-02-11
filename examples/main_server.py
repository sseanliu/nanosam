# main_server.py (or your existing script)
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

# We no longer do: from nanosam.utils.predictor import Predictor
# or from nanosam.utils.tracker import Tracker
# Because child_infer.py has them instead.

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen IP")
parser.add_argument("--port", type=int, default=12345, help="Listen Port")
args = parser.parse_args()

def cv2_to_pil(image_bgr):
    # We keep your function, but we won't do local predictor usage
    return PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

# Keep these global for overlay
mask = None
point = None

latestFrame = None
latestFrameLock = threading.Lock()
RUNNING = True

# We no longer keep updateInProgress or concurrency logic, since child does the net pass
childProc = None       # reference to the spawned child process
maskTimestamp = 0.0    # to track if we read the new mask file

def init_track(event, x, y, flags, param):
    global mask, point
    global childProc

    if event == cv2.EVENT_LBUTTONDBLCLK:
        # double-click => spawn child_infer
        print("[DoubleClick] => spawn child process")
        frameCopy = None
        with latestFrameLock:
            if latestFrame is not None:
                frameCopy = latestFrame.copy()
        if frameCopy is None:
            print("No frame to run child on.")
            return

        # 1) Write the frame to disk
        input_path = "tmp_input.jpg"
        output_path = "tmp_mask.png"
        cv2.imwrite(input_path, frameCopy)

        # 2) If there's an existing childProc, kill it
        if childProc is not None:
            print("Killing old childProc before starting new.")
            childProc.terminate()
            childProc = None

        # 3) Build the command, pass encoder/decoder from your original args
        cmd = [
            "python", "child_infer.py",
            args.image_encoder,
            args.mask_decoder,
            input_path,
            output_path,
            str(x), str(y)    # pass the click coords
        ]

        # 4) Launch child process
        childProc = subprocess.Popen(cmd)
        print(f"Spawned child_infer with click=({x},{y}).")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # right-click => forcibly kill child if any
        print("[RightClick] => kill child process")
        if childProc is not None:
            childProc.terminate()
            childProc = None
            print("Child process killed.")
        # Also clear existing mask
        mask = None
        point = None

cv2.namedWindow('image')
cv2.namedWindow('mask')
cv2.setMouseCallback('image', init_track)

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

def handle_frame(frame_bgr):
    """
    We no longer do local tracking. We only overlay the 'mask' if we have it from the child.
    If the child finishes, we read tmp_mask.png and store in 'mask'.
    """
    global mask, point
    # 1) Check if childProc finished => read mask from disk if new
    refresh_mask_if_child_done()

    # 2) Overlay mask if present
    disp = frame_bgr.copy()
    if mask is not None:
        bin_mask = (mask < 128)  # treat <128 as background
        green = np.zeros_like(disp)
        green[:] = (0,185,118)
        green[bin_mask] = 0
        disp = cv2.addWeighted(disp, 0.4, green, 0.6, 0)

        obj_mask = np.logical_not(bin_mask)
        mask_disp = np.zeros_like(disp)
        mask_disp[obj_mask] = (255,255,255)
        cv2.imshow("mask", mask_disp)

    # 3) Show in main window
    cv2.imshow("image", disp)

def refresh_mask_if_child_done():
    global childProc, mask, maskTimestamp
    if childProc is None:
        return
    # poll() returns None if still running, or exit code if finished
    retcode = childProc.poll()
    if retcode is None:
        # child still running
        return

    # child finished, so read tmp_mask.png
    childProc = None
    output_path = "tmp_mask.png"
    if not os.path.exists(output_path):
        print("[ChildInfer] finished but no mask file found.")
        mask = None
        return

    # read it
    new_mask_img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    if new_mask_img is None:
        print("[ChildInfer] Mask file invalid or empty.")
        mask = None
        return

    # store in 'mask'
    mask = new_mask_img
    print("[ChildInfer] loaded new mask from disk.")

def main():
    global RUNNING
    global latestFrame   # <-- add this line

    t = threading.Thread(target=server_thread, args=(args.host, args.port), daemon=True)
    t.start()

    while True:
        ret = cv2.waitKey(30)
        if ret == ord('q'):
            RUNNING = False
            break

        frame_to_process = None
        with latestFrameLock:
            if latestFrame is not None:
                frame_to_process = latestFrame.copy()
                latestFrame = None

        if frame_to_process is not None:
            handle_frame(frame_to_process)

    cv2.destroyAllWindows()
    print("Exiting main...")


if __name__=="__main__":
    main()
