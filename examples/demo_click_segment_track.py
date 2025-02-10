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

# ========== Your original NanoSAM objects ==========
predictor = Predictor(args.image_encoder, args.mask_decoder)
tracker = Tracker(predictor)

mask = None
point = None
image_pil = None

# ========== Global to hold the latest BGR frame from the server ==========
latestFrame = None
latestFrameLock = threading.Lock()

# ========== We'll keep the double-click callback for local usage ==========
def init_track(event, x, y, flags, param):
    global mask, point, image_pil
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # The user double-clicked in the main window -> init track
        if image_pil is not None:
            mask = tracker.init(image_pil, point=(x, y))
            point = (x, y)

cv2.namedWindow('image')
cv2.namedWindow('mask')
cv2.setMouseCallback('image', init_track)

def server_thread(host, port):
    """
    Minimal TCP server that receives frames from Vision Pro,
    decodes JPEG -> BGR, stores them in `latestFrame` for main thread to handle.
    Also returns mask data if we have a valid mask.
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
                if not length_data or len(length_data) < 4:
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

                # decode JPEG => BGR
                nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("Failed to decode JPEG.")
                    continue

                # Store in global variable
                with latestFrameLock:
                    # Just keep the newest frame
                    global latestFrame
                    latestFrame = frame

                # 3) produce and send back the current mask
                #    We'll do it here, but the actual mask is computed in main
                #    after handle_frame. We can let main update a global "mask" variable.
                #    To keep minimal changes, let's assume the same old logic:
                #    We'll just send the existing "mask" if available.

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

        except Exception as e:
            print("Server thread error:", e)
    s.close()
    print("Server closed.")


def handle_frame(frame_bgr):
    """
    Moved from your original 'handle_frame' logic,
    but used in the main thread so we can do cv2.imshow safely.
    """
    global mask, point, image_pil

    image_pil = cv2_to_pil(frame_bgr)

    if tracker.token is not None:
        mask, point = tracker.update(image_pil)

    # draw results
    display_frame = frame_bgr.copy()
    if mask is not None:
        bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
        green_image = np.zeros_like(display_frame)
        green_image[:] = (0, 185, 118)
        green_image[~bin_mask] = 0

        mask_display = np.zeros_like(display_frame)
        mask_display[bin_mask] = (255, 255, 255)
        cv2.imshow("mask", mask_display)

        alpha = 0.6
        display_frame = cv2.addWeighted(display_frame, 1-alpha, green_image, alpha, 0)

    if point is not None:
        cv2.circle(display_frame, point, 5, (0,185,118), -1)

    cv2.imshow("image", display_frame)


def main():
    t = threading.Thread(target=server_thread, args=(args.host, args.port), daemon=True)
    t.start()

    while True:
        # WaitKey in main so the windows can respond
        ret = cv2.waitKey(30)
        if ret == ord('q'):
            break
        elif ret == ord('r'):
            tracker.reset()
            print("Tracker reset")

        # check if there's a new frame
        frame_to_process = None
        with latestFrameLock:
            if latestFrame is not None:
                # copy it for local usage
                frame_to_process = latestFrame.copy()

        if frame_to_process is not None:
            # call handle_frame in main thread
            handle_frame(frame_to_process)

    cv2.destroyAllWindows()
    print("Exiting main...")

if __name__ == "__main__":
    main()
