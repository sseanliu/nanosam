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
PORT = 55555   # local port for parent->child

# Global model objects
predictor = None
tracker = None
mask_cache = None  # last known mask
point_cache = None # last known point

def load_model_once():
    global predictor, tracker
    # For demonstration, we just hardcode or read from environment
    encoder = os.environ.get("ENCODER_PATH", "data/resnet18_image_encoder.engine")
    decoder = os.environ.get("DECODER_PATH", "data/mobile_sam_mask_decoder.engine")

    print("[ChildServer] Loading model with", encoder, decoder)
    predictor = Predictor(encoder, decoder)
    tracker = Tracker(predictor)
    print("[ChildServer] Model loaded successfully.")

def handle_init(command_parts, conn):
    # command_parts = ["INIT", x, y, "frame.jpg"]
    if len(command_parts) < 4:
        print("[ChildServer] INIT missing arguments.")
        send_mask(None, conn)
        return
    x = int(command_parts[1])
    y = int(command_parts[2])
    frame_path = command_parts[3]

    # read frame
    image_bgr = cv2.imread(frame_path)
    if image_bgr is None:
        print("[ChildServer] INIT could not read frame:", frame_path)
        send_mask(None, conn)
        return

    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # call tracker.init
    print(f"[ChildServer] tracker.init @ ({x}, {y}) with frame={frame_path}")
    global mask_cache, point_cache
    mask_init = tracker.init(image_pil, point=(x, y))
    mask_cache = mask_init
    point_cache = (x, y)
    send_mask(mask_init, conn)

def handle_update(command_parts, conn):
    # command_parts = ["UPDATE", "frame.jpg"]
    if len(command_parts) < 2:
        print("[ChildServer] UPDATE missing arguments.")
        send_mask(None, conn)
        return
    frame_path = command_parts[1]
    image_bgr = cv2.imread(frame_path)
    if image_bgr is None:
        print("[ChildServer] UPDATE could not read frame:", frame_path)
        send_mask(None, conn)
        return

    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    print("[ChildServer] tracker.update with frame =", frame_path)
    global mask_cache, point_cache
    new_mask, new_point = tracker.update(image_pil)
    if new_mask is not None:
        mask_cache = new_mask
        point_cache = new_point
    send_mask(new_mask, conn)

def handle_reset(command_parts, conn):
    print("[ChildServer] tracker.reset()")
    tracker.reset()
    global mask_cache, point_cache
    mask_cache = None
    point_cache = None
    send_mask(None, conn)  # no mask

def send_mask(sam_mask, conn):
    """
    Convert mask to bin_mask. If None, length=0
    """
    if sam_mask is None:
        # length=0
        header = struct.pack('<i', 0)
        conn.sendall(header)
        return
    # else
    bin_mask = (sam_mask[0,0].detach().cpu().numpy() < 0).astype(np.uint8)*255
    # encode to png in memory
    ret, png_data = cv2.imencode('.png', bin_mask)
    if not ret:
        header = struct.pack('<i', 0)
        conn.sendall(header)
        return
    mask_bytes = png_data.tobytes()
    header = struct.pack('<i', len(mask_bytes))
    conn.sendall(header)
    conn.sendall(mask_bytes)

def client_handler(conn):
    while True:
        # each command is: length(4 bytes) + payload
        header = conn.recv(4)
        if not header or len(header) < 4:
            print("[ChildServer] Client disconnected.")
            break
        (cmd_len,) = struct.unpack('<i', header)
        if cmd_len <= 0:
            print("[ChildServer] Invalid cmd_len:", cmd_len)
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
            print("[ChildServer] partial command read => disconnecting.")
            break
        command_str = cmd_data.decode('utf-8').strip()
        command_parts = command_str.split()
        if not command_parts:
            continue

        cmd = command_parts[0].upper()
        if cmd == "INIT":
            handle_init(command_parts, conn)
        elif cmd == "UPDATE":
            handle_update(command_parts, conn)
        elif cmd == "RESET":
            handle_reset(command_parts, conn)
        else:
            print("[ChildServer] Unknown command:", cmd)
            send_mask(None, conn)  # or do nothing
    conn.close()

def main():
    load_model_once()
    # Start a simple server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"[ChildServer] listening at {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        print("[ChildServer] Got connection from", addr)
        threading.Thread(target=client_handler, args=(conn,), daemon=True).start()

if __name__ == "__main__":
    main()
