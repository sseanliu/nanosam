# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA ...
# SPDX-License-Identifier: Apache-2.0

import PIL.Image
import cv2
import numpy as np
import argparse
import socket
import struct
import threading

from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen IP")
parser.add_argument("--port", type=int, default=12345, help="Listen Port")
args = parser.parse_args()

def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)

predictor = Predictor(args.image_encoder, args.mask_decoder)
tracker = Tracker(predictor)

mask = None
point = None

# We'll no longer open a local webcam. Instead, we create a server socket
# to receive frames from the Vision Pro.

def init_track(event,x,y,flags,param):
    global mask, point, image_pil
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # The user can double-click to init a new track
        mask = tracker.init(image_pil, point=(x, y))
        point = (x, y)

cv2.namedWindow('image')
cv2.namedWindow('mask')
cv2.setMouseCallback('image', init_track)

def handle_frame(frame):
    """
    This function replicates the logic we originally had in the while loop with
    the local camera. We'll parse the incoming BGR frame, do the tracker update, 
    and show the result in the same windows 'image' and 'mask'.
    """
    global mask, point, image_pil

    # Convert to PIL for the predictor/tracker
    image_pil = cv2_to_pil(frame)

    if tracker.token is not None:
        # If we have an ongoing track, update it
        mask, point = tracker.update(image_pil)

    # Draw mask
    display_frame = frame.copy()
    if mask is not None:
        bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
        green_image = np.zeros_like(display_frame)
        green_image[:] = (0, 185, 118)
        # wherever bin_mask==True, we keep green, else 0
        green_image[~bin_mask] = 0
        
        mask_display = np.zeros_like(display_frame)
        mask_display[bin_mask] = (255, 255, 255)
        cv2.imshow("mask", mask_display)

        # alpha blend
        alpha = 0.6
        display_frame = cv2.addWeighted(display_frame, 1-alpha, green_image, alpha, 0)

    if point is not None:
        cv2.circle(display_frame, point, 5, (0, 185, 118), -1)

    cv2.imshow("image", display_frame)

def server_thread(host, port):
    """
    A simple TCP server that listens on (host, port).
    Each time a frame is received (JPEG), decode it, 
    call handle_frame(...) 
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print(f"NanoSAM server listening at {host}:{port}")

    conn, addr = s.accept()
    print("Got connection from", addr)
    with conn:
        try:
            while True:
                # 1) read 4 bytes -> length
                length_data = conn.recv(4)
                if not length_data or len(length_data)<4:
                    print("Client disconnected or no length data.")
                    break
                (length,) = struct.unpack('<i', length_data)
                length = int(length)
                if length <= 0:
                    print("Invalid length:", length)
                    break
                
                # 2) read the JPEG data
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
                    break

                # 3) decode JPEG => BGR
                nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("Failed to decode JPEG.")
                    continue

                # 4) pass to handle_frame
                handle_frame(frame)

                # 5) produce a mask to send back
                # for demonstration, we'll replicate your approach:
                # if mask is not None, let's encode it as a PNG
                # if mask is None, we send length=0
                out_mask_data = b''
                if mask is not None:
                    bin_mask = (mask[0,0].detach().cpu().numpy() < 0).astype(np.uint8) * 255
                    ret, png_data = cv2.imencode('.png', bin_mask)
                    if ret:
                        out_mask_data = png_data.tobytes()
                mask_len = len(out_mask_data)
                # send 4 bytes + mask data
                send_len = struct.pack('<i', mask_len)
                conn.sendall(send_len)
                if mask_len > 0:
                    conn.sendall(out_mask_data)
        except Exception as e:
            print("Server thread error:", e)
    s.close()
    print("Server closed.")


def main():
    # parse your args if needed
    # but let's just run the server
    t = threading.Thread(target=server_thread, args=(args.host, args.port))
    t.start()

    while True:
        # keep reading the UI windows
        # so that double-click event can happen
        ret = cv2.waitKey(30)
        if ret == ord('q'):
            break
        elif ret == ord('r'):
            tracker.reset()
            print("Tracker reset")
    
    cv2.destroyAllWindows()
    print("Exiting main...")

if __name__ == "__main__":
    main()
