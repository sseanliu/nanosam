# child_infer_server.py

import sys
import cv2
import numpy as np
import socket
import struct
import os
import threading
import PIL.Image

from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker

HOST = "127.0.0.1"
PORT = 55555

predictor = None
tracker = None
mask_cache = None
point_cache = None

def load_model_once():
    global predictor, tracker
    encoder = os.environ.get("ENCODER_PATH", "data/resnet18_image_encoder.engine")
    decoder = os.environ.get("DECODER_PATH", "data/mobile_sam_mask_decoder.engine")
    print(f"[ChildServer] Loading model with {encoder} {decoder}")
    predictor = Predictor(encoder, decoder)
    tracker = Tracker(predictor)
    print("[ChildServer] Model loaded successfully.")

def handle_init(parts, conn):
    if len(parts) < 4:
        print("[ChildServer] INIT missing args.")
        send_mask(None, conn)
        return
    x = int(parts[1])
    y = int(parts[2])
    img_path = parts[3]

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print("[ChildServer] INIT could not read frame:", img_path)
        send_mask(None, conn)
        return

    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    global mask_cache, point_cache
    print(f"[ChildServer] tracker.init @ ({x}, {y}) with {img_path}")
    init_mask = tracker.init(image_pil, point=(x, y))
    mask_cache = init_mask
    point_cache = (x, y)
    send_mask(init_mask, conn)

def handle_update(parts, conn):
    if len(parts) < 2:
        print("[ChildServer] UPDATE missing frame path.")
        send_mask(None, conn)
        return
    img_path = parts[1]

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print("[ChildServer] UPDATE could not read frame:", img_path)
        send_mask(None, conn)
        return

    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    global mask_cache, point_cache
    if tracker.token is None:
        print("[ChildServer] UPDATE but no active track.")
        send_mask(None, conn)
        return

    print("[ChildServer] tracker.update() with:", img_path)
    new_mask, new_pt = tracker.update(image_pil)
    if new_mask is not None:
        mask_cache = new_mask
        point_cache = new_pt
    send_mask(new_mask, conn)

def handle_reset(parts, conn):
    print("[ChildServer] tracker.reset()")
    tracker.reset()
    global mask_cache, point_cache
    mask_cache = None
    point_cache = None
    send_mask(None, conn)

def send_mask(sam_mask, conn):
    """Send a PNG mask to the parent. We interpret '>= 0' as object => 255, background => 0."""
    if sam_mask is None:
        hdr = struct.pack('<i', 0)
        conn.sendall(hdr)
        return

    # FIX: Use >= 0 for object => 255. This matches the original highlight logic.
    bin_mask = (sam_mask[0,0].detach().cpu().numpy() >= 0).astype(np.uint8)*255

    ret, png_data = cv2.imencode('.png', bin_mask)
    if not ret:
        hdr = struct.pack('<i', 0)
        conn.sendall(hdr)
        return

    mask_bytes = png_data.tobytes()
    hdr = struct.pack('<i', len(mask_bytes))
    conn.sendall(hdr)
    conn.sendall(mask_bytes)

def client_handler(conn):
    while True:
        head = conn.recv(4)
        if not head or len(head)<4:
            print("[ChildServer] client disconnected.")
            break
        (cmd_len,) = struct.unpack('<i', head)
        if cmd_len<=0:
            print("[ChildServer] invalid cmd_len:", cmd_len)
            break

        cmd_data = b''
        to_read = cmd_len
        while to_read>0:
            chunk = conn.recv(to_read)
            if not chunk:
                break
            cmd_data += chunk
            to_read -= len(chunk)
        if len(cmd_data)<cmd_len:
            print("[ChildServer] partial cmd => break")
            break

        cmd_str = cmd_data.decode('utf-8').strip()
        parts = cmd_str.split()
        if not parts:
            continue

        cmd = parts[0].upper()
        if cmd=="INIT":
            handle_init(parts, conn)
        elif cmd=="UPDATE":
            handle_update(parts, conn)
        elif cmd=="RESET":
            handle_reset(parts, conn)
        else:
            print("[ChildServer] unknown command:", cmd)
            send_mask(None, conn)
    conn.close()

def main():
    load_model_once()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"[ChildServer] listening at {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        print("[ChildServer] Got connection from", addr)
        th = threading.Thread(target=client_handler, args=(conn,), daemon=True)
        th.start()

if __name__=="__main__":
    main()
