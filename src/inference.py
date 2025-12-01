"""
inference.py
--------------
YOLOv8 helmet + license plate detection + EasyOCR.
Saves annotated output + license plate crops.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import argparse
import os


# ---------------------------------------------------
# Simple LP preprocessing
# ---------------------------------------------------
def preprocess_lp(crop):
    """Upscale + contrast enhance license plate."""
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 5 or w < 15:
        return None

    crop = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------
# ---------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------
def run_inference(weights, image_path, save=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Fixed output folder
    output_dir = r"C:\Users\nick2\Desktop\github\Bike helmet\src\output"
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO(weights)
    class_names = model.names

    # Load EasyOCR
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        print("[INFO] EasyOCR running on GPU.")
    except:
        reader = easyocr.Reader(['en'], gpu=False)
        print("[INFO] EasyOCR running on CPU.")

    frame = cv2.imread(image_path)
    h_img, w_img = frame.shape[:2]

    # Run YOLO
    results = model(frame, conf=0.25, device=0)[0]

    lp_texts = []
    annotated = frame.copy()
    lp_crop_count = 0

    # ---------------------------------------------
    # Loop through detections
    # ---------------------------------------------

    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0])
        label = class_names[cls]

        pad = 5
        x1 = max(0, xyxy[0] - pad)
        y1 = max(0, xyxy[1] - pad)
        x2 = min(w_img, xyxy[2] + pad)
        y2 = min(h_img, xyxy[3] + pad)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ------- Ensure label is always visible -------
        label_text = label
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw above box if enough space, else below
        if y1 - th - 6 < 0:
            bg_y1 = y2
            bg_y2 = min(h_img, y2 + th + 6)
            text_y = bg_y1 + th - 2
        else:
            bg_y1 = y1 - th - 6
            bg_y2 = y1
            text_y = y1 - 5

        bg_x2 = min(w_img, x1 + tw + 6)

        # Draw label background
        cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)

        # Draw label text
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # -----------------------------------------
        # OCR only for license plate
        # -----------------------------------------
        if label.lower() == "license plate":
            lp_crop = frame[y1:y2, x1:x2]
            lp_crop_count += 1

            # Save the raw crop
            crop_path = os.path.join(output_dir, f"lp_crop_{lp_crop_count}.jpg")
            cv2.imwrite(crop_path, lp_crop)
            print(f"[INFO] Saved LP crop → {crop_path}")

            pre = preprocess_lp(lp_crop)
            if pre is not None:
                result = reader.readtext(pre)
                text = "".join([r[1] for r in result]).strip()
                if text:
                    lp_texts.append(text)

    # Save annotated output
    if save:
        out_path = os.path.join(output_dir, "annotated_output.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Saved annotated image → {out_path}")

    return lp_texts, annotated



# ---------------------------------------------------
# CLI execution
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="C:\\Users\\nick2\\Desktop\\github\\Bike helmet\\models\\best.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--save", action="store_true", help="Save output image")

    args = parser.parse_args()

    texts, _ = run_inference(args.weights, args.image, args.save)

    if texts:
        print("\n=== LICENSE PLATE TEXT DETECTED ===")
        for t in texts:
            print("•", t)
    else:
        print("\nNo license plate text found.")
