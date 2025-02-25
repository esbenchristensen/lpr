import cv2
import torch
import easyocr
import numpy as np
import re

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
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you have a compatible GPU

# -----------------------------
# 3) Preprocessing Function for EasyOCR
# -----------------------------
def preprocess_for_easyocr(plate_img):
    """
    Preprocess the cropped license plate image to improve OCR accuracy:
      1. Convert to grayscale.
      2. Apply bilateral filtering to denoise while preserving edges.
      3. Apply Otsu thresholding to produce a binary image.
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
    Crop the plate region from the full image and attempt to deskew it.
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
    rotated = cv2.warpAffine(
        plate, M, (plate.shape[1], plate.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

# -----------------------------
# 5) EasyOCR-based OCR Function
# -----------------------------
def ocr_license_plate(cropped_image):
    """
    Preprocess the image, run EasyOCR with a restricted allowlist,
    and then check if the recognized text exactly matches
    the pattern for Danish license plates (2 letters followed by 5 digits).
    """
    processed = preprocess_for_easyocr(cropped_image)
    # Restrict recognition to uppercase letters and digits.
    results = reader.readtext(processed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    # Combine results into a single string (remove spaces)
    predicted_text = "".join(results).replace(" ", "")
    print("OCR predicted text:", predicted_text)
    # Use fullmatch to enforce that the entire string matches the pattern.
    match = re.fullmatch(r'[A-Z]{2}[0-9]{5}', predicted_text)
    if match:
        return match.group(0)
    # Return an empty string (or handle it as you see fit) if the result is invalid.
    return ""

# -----------------------------
# 6) Main Detection & OCR Integration
# -----------------------------
def detect_and_read(image_path):
    # Load the full image using OpenCV.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not open image: {image_path}")
        return

    # Run YOLOv5 detection.
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Crop the detected plate region.
        cropped = img[y1:y2, x1:x2]
        
        # Optional: Deskew the plate region.
        rotated_crop = deskew_plate(img, x1, y1, x2, y2)
        
        # Run OCR using EasyOCR on the deskewed image.
        plate_text = ocr_license_plate(rotated_crop)
        
        # Optionally, crop out the left part (e.g., the blue DK icon) if needed.
        h, w, _ = cropped.shape
        left_margin = int(w * 0.10)  # Change to 15% if necessary.
        right_margin = int(w * 0.87) # Change to 85% if necessary.
        if left_margin < w:
            cropped = cropped[:, left_margin:]

        if right_margin < w:
            cropped = cropped[:, :right_margin]
        
        # Save a debug image.
        debug_path = f"cropped_debug_{x1}_{y1}.png"
        cv2.imwrite(debug_path, cropped)
        print("Saved debug image to:", debug_path)
        
        # Run OCR on the final cropped image.
        plate_text = ocr_license_plate(cropped)
        print("Detected license plate:", plate_text)
        
        # Draw the bounding box and overlay the text.
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Detection & OCR", img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

# -----------------------------
# 7) Main Entry Point
# -----------------------------
if __name__ == "__main__":
    # Update this path to the image you wish to test.
    image_path = "/Users/esbenchristensen/Github/lpr/7E4DB52E-B8AA-4FC5-867C-35F814563176_1_105_c.jpeg"
    detect_and_read(image_path)