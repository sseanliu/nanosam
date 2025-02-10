import PIL.Image
import cv2
import numpy as np
import argparse
import socket
import struct
import threading
import time

from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen IP")
parser.add_argument("--port", type=int, default=12345, help="Listen Port")
args = parser.parse_args()

def cv2_to_pil(image_bgr):
    return PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

predictor = Predictor(args.image_encoder, args.mask_decoder)
tracker = Tracker(predictor)

mask = None
point = None
image_pil = None

latestFrame = None
latestFrameLock = threading.Lock()
RUNNING = True

# (1) Two new rate-limit variables
updateInProgress = False
lastUpdateTime = 0.0
updateInterval = 0.5  # e.g., update tracker at most every 0.5s

lastImagePil = None  # stores the last PIL image used for the worker

def init_track(event, x, y, flags, param):
    global mask, point, image_pil
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if image_pil is not None:
            print("[DoubleClick] init track at:", (x, y))
            mask = tracker.init(image_pil, point=(x, y))
            point = (x, y)
            print("[DoubleClick] track init done.")

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

            # produce mask
            out_mask_data = b''
            if mask is not None:
                bin_mask = (mask[0,0].detach().cpu().numpy() < 0).astype(np.uint8)*255
                ret, png_data = cv2.imencode('.png', bin_mask)
                if ret:
                    out_mask_data = png_data.tobytes()

            mask_len = len(out_mask_data)
            send_len = struct.pack('<i', mask_len)
            conn.sendall(send_len)
            if mask_len > 0:
                conn.sendall(out_mask_data)

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
    We'll just do the overlay portion in the main thread.
    'tracker.update(...)' is called in a separate thread, but also rate-limited
    so we do not saturate the GPU frequently.
    """
    global mask, point, image_pil
    global updateInProgress, lastUpdateTime, updateInterval
    global lastImagePil

    image_pil = cv2_to_pil(frame_bgr)

    # (2) Rate-limiting check + concurrency check
    now = time.time()
    if tracker.token is not None and not updateInProgress and (now - lastUpdateTime) > updateInterval:
        # store the current PIL for the worker
        lastImagePil = image_pil
        updateInProgress = True
        lastUpdateTime = now  # mark time we started an update
        t = threading.Thread(target=tracker_update_worker, daemon=True)
        t.start()

    # overlay with the existing mask
    disp = frame_bgr.copy()
    if mask is not None:
        bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
        green = np.zeros_like(disp)
        green[:] = (0,185,118)
        green[~bin_mask] = 0

        mask_disp = np.zeros_like(disp)
        mask_disp[bin_mask] = (255,255,255)
        cv2.imshow("mask", mask_disp)

        alpha=0.6
        disp = cv2.addWeighted(disp, 1-alpha, green, alpha, 0)

    if point is not None:
        cv2.circle(disp, point, 5, (0,185,118), -1)

    cv2.imshow("image", disp)

def tracker_update_worker():
    """
    A short-lived thread that calls 'tracker.update(lastImagePil)'
    once done, updates global mask + point, sets 'updateInProgress=false'
    """
    global updateInProgress, mask, point, lastImagePil
    print("[TrackerThread] Starting update...")
    try:
        new_mask, new_point = tracker.update(lastImagePil)
        if new_mask is not None:
            mask = new_mask
            point = new_point
    except Exception as e:
        print("[TrackerThread] error:", e)
    updateInProgress = False
    print("[TrackerThread] Done update.")

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
        elif ret == ord('r'):
            tracker.reset()
            print("Tracker reset")

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
