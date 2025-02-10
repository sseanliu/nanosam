#!/usr/bin/env python3
# server_nanosam.py

import socket
import struct
import cv2
import numpy as np
import sys

# If 'nanosam' is installed as a package, you can do:
# from nanosam.utils.predictor import Predictor
# Otherwise, adapt to how you have your local code.

from nanosam.utils.predictor import Predictor

# -------------------------------------------------------------------
# 1. Initialize your nanoSAM predictor
#    Adjust paths according to your environment.
# -------------------------------------------------------------------
predictor = Predictor(
    image_encoder="data/resnet18_image_encoder.engine",
    mask_decoder="data/mobile_sam_mask_decoder.engine"
)
print("Loaded nanoSAM predictor.")

# -------------------------------------------------------------------
# 2. Server config
# -------------------------------------------------------------------
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 12345

# -------------------------------------------------------------------
# 3. Helper function to run your nanoSAM on an incoming BGR image
#    This function is purely a demonstration. Real usage will vary.
# -------------------------------------------------------------------
def run_nanosam(img_bgr):
    """
    Run nanoSAM inference on the input BGR image.
    Return a single-channel uint8 mask (255=object, 0=background) with same size as 'img_bgr'.
    """
    # Typically you'd provide a prompt or some point/box for nanoSAM. 
    # For demonstration, let's define a center point or any logic.
    # We'll just do a trivial step or skip if we don't have an ROI.
    height, width = img_bgr.shape[:2]

    # Suppose we define a single "point" near the center, or none
    # If your application requires user-driven prompts, you'd keep track of them.
    # For example:
    center_pt = (width // 2, height // 2)
    
    # predictor.run() might expect an entire 2D detection pipeline or the code from your nanosam usage
    # Example usage from your code might be:
    # mask = tracker.init(...) or predictor.predict_mask(...)
    # We'll produce a dummy circle for now:
    
    # For a real example, you might do:
    # mask = predictor.predict(img_bgr, point=center_pt)
    # But let's do a quick circle as a placeholder
    mask_bin = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_bin, center_pt, min(width//4, height//4), 255, -1)
    
    return mask_bin

# -------------------------------------------------------------------
# 4. Main loop handling each client
# -------------------------------------------------------------------
def handle_client(conn):
    """
    For each client connection:
      1. Repeatedly read 4 bytes (length).
      2. Read 'length' bytes of JPEG.
      3. Decode -> BGR image.
      4. Run nanoSAM -> single-channel mask.
      5. Encode mask as PNG.
      6. Send back (length + data).
    """
    while True:
        # Step A: read 4 bytes for length
        length_data = recv_exact(conn, 4)
        if not length_data:
            print("Client disconnected.")
            break
        (length,) = struct.unpack('<i', length_data)
        
        # Step B: read 'length' bytes of JPEG
        jpeg_bytes = recv_exact(conn, length)
        if not jpeg_bytes:
            print("No image data received, client might have closed.")
            break
        
        # Step C: decode JPEG
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode JPEG. Skipping.")
            continue
        
        # Step D: run nanoSAM -> mask
        mask_bin = run_nanosam(img)
        
        # Step E: encode mask as PNG
        # mask_bin is [0 or 255], shape=(H,W). We'll do grayscale PNG
        ret, png_data = cv2.imencode('.png', mask_bin)
        if not ret:
            print("PNG encode fail.")
            continue
        
        # Step F: send back (length + data)
        out_len = len(png_data).to_bytes(4, 'little', signed=True)
        conn.sendall(out_len + png_data)

    conn.close()

def recv_exact(conn, length):
    """
    Utility to read exactly 'length' bytes from conn.
    If we can't, return None.
    """
    buf = b''
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

# -------------------------------------------------------------------
# 5. Server main
# -------------------------------------------------------------------
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening on port {PORT} ...")
    
    while True:
        conn, addr = s.accept()
        print("Got connection from:", addr)
        handle_client(conn)

if __name__ == "__main__":
    main()
