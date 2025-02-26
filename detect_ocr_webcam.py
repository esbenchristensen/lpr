import cv2
import torch
import easyocr
import numpy as np
import re
import time

# -----------------------------
# 1) Load YOLOv5 Model
# -----------------------------
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='/Users/esbenchristensen/Github/lpr/yolov5/runs/train/exp/weights/best.pt',
    force_reload=True
)

# -----------------------------
# 2) Initialize EasyOCR Reader (global instance)
# -----------------------------
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available

# A set to store unique detected plates.
registered_plates = set()
output_file = "registered_plates.txt"

# -----------------------------
# 3) Preprocessing Function for EasyOCR
# -----------------------------
def preprocess_for_easyocr(plate_img):
    """
    Preprocess the cropped license plate image:
      1. Convert to grayscale.
      2. Apply bilateral filtering.
      3. Apply Otsu thresholding.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# -----------------------------
# 4) Deskew Function
# -----------------------------
def deskew_plate(img, x1, y1, x2, y2):
    """
    Crop and deskew the plate region from the full image.
    """
    plate = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return plate
    c = max(cnts, key=cv2.contourArea)
    rot_rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rot_rect
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(plate, M, (plate.shape[1], plate.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# -----------------------------
# 5) EasyOCR-based OCR Function
# -----------------------------
def ocr_license_plate(cropped_image):
    """
    Run OCR on the cropped image after preprocessing.
    Accepts only the format: 2 letters followed by 5 digits.
    """
    processed = preprocess_for_easyocr(cropped_image)
    # Use allowlist to restrict recognition to uppercase letters and digits.
    results = reader.readtext(processed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    predicted_text = "".join(results).replace(" ", "")
    match = re.fullmatch(r'[A-Z]{2}[0-9]{5}', predicted_text)
    if match:
        return match.group(0)
    return ""

# -----------------------------
# 6) Function to Register a New Plate
# -----------------------------
def register_plate(plate_text):
    global registered_plates
    if plate_text and plate_text not in registered_plates:
        registered_plates.add(plate_text)
        with open(output_file, "a") as f:
            f.write(plate_text + "\n")
        print("New plate registered:", plate_text)

# -----------------------------
# 7) Detection & OCR on Webcam Frames (Optimized)
# -----------------------------
def detect_and_read_webcam(process_every_n_frames=3, resize_width=640):
    """
    Process webcam frames for detection and OCR.
    To improve efficiency:
      - Only process every n-th frame.
      - Optionally resize frame for faster processing.
    """
    cap = cv2.VideoCapture(0)  # Default webcam
    if not cap.isOpened():
        print("Error opening webcam.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Resize frame for faster processing while keeping a copy for display.
        height, width = frame.shape[:2]
        scaling_factor = resize_width / float(width)
        resized_frame = cv2.resize(frame, (resize_width, int(height * scaling_factor)))
        display_frame = frame.copy()

        # Process every n-th frame to lighten processing load.
        if frame_count % process_every_n_frames == 0:
            start_time = time.time()
            results = model(resized_frame)
            detections = results.xyxy[0].cpu().numpy()

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Scale coordinates back to original frame size
                scale = width / float(resize_width)
                x1, y1, x2, y2 = map(lambda v: int(v * scale), [x1, y1, x2, y2])
                
                # Crop the detected plate region from the original frame.
                cropped = frame[y1:y2, x1:x2]
                
                # Optional: Deskew the plate region.
                rotated_crop = deskew_plate(frame, x1, y1, x2, y2)
                
                # Run OCR on the deskewed image.
                plate_text = ocr_license_plate(rotated_crop)
                
                # Optionally, adjust the crop to remove unwanted margins.
                h_crop, w_crop = cropped.shape[:2]
                left_margin = int(w_crop * 0.10)
                right_margin = int(w_crop * 0.87)
                if left_margin < w_crop:
                    cropped = cropped[:, left_margin:]
                if right_margin < w_crop:
                    cropped = cropped[:, :right_margin]
                
                # Run OCR on the final cropped image.
                plate_text = ocr_license_plate(cropped)
                print("Detected license plate:", plate_text)
                
                # Register the plate (if unique).
                register_plate(plate_text)
                
                # Draw bounding box and overlay the text on the original frame.
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            end_time = time.time()
            # Uncomment below to print processing time per processed frame
            # print(f"Processed frame in {end_time - start_time:.2f} seconds")
        
        frame_count += 1

        # Show the original frame with overlay.
        cv2.imshow("Webcam - Detection & OCR", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting webcam processing.")
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# 8) Main Entry Point
# -----------------------------
if __name__ == "__main__":
    detect_and_read_webcam(process_every_n_frames=3, resize_width=640)