# child_infer.py
import sys
import cv2
import numpy as np
import PIL.Image

from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker

def main():
    if len(sys.argv) < 6:
        # We expect: child_infer.py encoder_path decoder_path input.jpg output.png click_x click_y
        print("Usage: python child_infer.py <encoder_path> <decoder_path> <input_image> <output_mask> <click_x> <click_y>")
        sys.exit(1)

    encoder_path = sys.argv[1]
    decoder_path = sys.argv[2]
    input_image_path = sys.argv[3]
    output_mask_path = sys.argv[4]
    click_x = int(sys.argv[5])
    click_y = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    # 1) Load model
    predictor = Predictor(encoder_path, decoder_path)
    tracker = Tracker(predictor)

    # 2) Read input
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        print("Error reading input image.")
        sys.exit(1)

    # Convert to PIL
    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # 3) We do a typical "init" + "update" based on the click
    print(f"[child_infer] Doing init at ({click_x}, {click_y})")
    init_mask = tracker.init(image_pil, point=(click_x, click_y))
    new_mask, new_point = tracker.update(image_pil)

    # whichever final mask is not None
    final_mask = new_mask if new_mask is not None else init_mask
    if final_mask is None:
        print("[child_infer] No mask produced.")
        sys.exit(0)

    # 4) Convert to bin_mask
    bin_mask = (final_mask[0,0].detach().cpu().numpy() < 0).astype(np.uint8)*255

    # 5) Write to disk
    cv2.imwrite(output_mask_path, bin_mask)
    print("[child_infer] Wrote mask to", output_mask_path)

if __name__ == "__main__":
    main()
