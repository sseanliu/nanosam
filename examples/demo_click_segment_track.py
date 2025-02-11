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

# Initialize predictor + tracker as in original example
predictor = Predictor(args.image_encoder, args.mask_decoder)
tracker = Tracker(predictor)

# Globals for mask, point, etc.
mask = None
point = None
image_pil = None

latestFrame = None
latestFrameLock = threading.Lock()
RUNNING = True

def init_track(event, x, y, flags, param):
    global mask, point, image_pil

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if image_pil is not None:
            print("[DoubleClick] init track at:", (x, y))
            # 1) Initialize tracking
            mask_init = tracker.init(image_pil, point=(x, y))
            mask = mask_init
            point = (x, y)
            print("[DoubleClick] track init done.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[RightClick] Cancel/Reset tracking.")
        # 1) Reset the tracker
        tracker.reset()
        # 2) Clear out the mask + point + token
        tracker.token = None
        mask = None
        point = None

cv2.namedWindow('image')
cv2.namedWindow('mask')
cv2.setMouseCallback('image', init_track)

def server_thread(host, port):
    """
    Receives frames from Vision Pro (or similar) via TCP,
    storing the latest one in `latestFrame`.
    """
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
                print("Failed to decode incoming frame.")
                continue

            # store latest in global
            with latestFrameLock:
                global latestFrame
                latestFrame = frame

            # Also produce + send back mask if we have one
            out_mask_data = b''
            if mask is not None:
                bin_mask = (mask[0,0].detach().cpu().numpy() < 0).astype(np.uint8) * 255
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
            print("Server thread error:", e)
            conn.close()
            conn = None
            continue

    if conn:
        conn.close()
    s.close()
    print("Server closed...")

def handle_frame(frame_bgr):
    """
    Synchronous approach (like the original example).
    If the tracker.token is set, we directly call tracker.update(...) each frame.
    Then we overlay the result.
    """
    global mask, point, image_pil

    # Convert to PIL
    image_pil = cv2_to_pil(frame_bgr)

    # If we have a tracker.token, do a synchronous update
    if tracker.token is not None:
        mask_update, pt_update = tracker.update(image_pil)
        if mask_update is not None:
            mask = mask_update
            point = pt_update

    # Overlay the mask
    disp = frame_bgr.copy()
    if mask is not None:
        bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
        green = np.zeros_like(disp)
        green[:] = (0,185,118)
        green[bin_mask] = 0

        disp = cv2.addWeighted(disp, 0.4, green, 0.6, 0)

        # optional separate window to show the mask in white
        obj_mask = np.logical_not(bin_mask)
        mask_disp = np.zeros_like(disp)
        mask_disp[obj_mask] = (255,255,255)
        cv2.imshow("mask", mask_disp)
    else:
        # if no mask, black mask window
        black_img = np.zeros_like(disp)
        cv2.imshow("mask", black_img)

    # If we have a point, draw a circle
    if point is not None:
        disp = cv2.circle(disp, point, 5, (0,185,118), -1)

    cv2.imshow("image", disp)

def main():
    global RUNNING
    global latestFrame

    # Start the server thread
    t = threading.Thread(target=server_thread, args=(args.host, args.port), daemon=True)
    t.start()

    while True:
        # poll for user input
        ret = cv2.waitKey(30)
        if ret == ord('q'):
            RUNNING = False
            break
        elif ret == ord('r'):
            tracker.reset()
            tracker.token = None
            mask = None
            point = None
            print("Tracker reset")

        # fetch the latest frame
        frame_to_process = None
        with latestFrameLock:
            if latestFrame is not None:
                frame_to_process = latestFrame.copy()
                latestFrame = None

        # If we have a new frame, do synchronous update + display
        if frame_to_process is not None:
            handle_frame(frame_to_process)

    cv2.destroyAllWindows()
    print("Exiting main...")

if __name__=="__main__":
    main()
