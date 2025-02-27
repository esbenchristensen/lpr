import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import re
import json
import base64
import io
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import cv2
import torch
import easyocr
import numpy as np
from PIL import Image

app = FastAPI()

# -----------------------------
# 1) Load YOLOv5 Model
# -----------------------------
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='/Users/esbenchristensen/Github/lpr/yolov5/runs/train/exp/weights/best.pt',  # update as needed
    force_reload=True
)

# -----------------------------
# 2) Initialize EasyOCR Reader
# -----------------------------
reader = easyocr.Reader(['en'], gpu=True)

# Directory and file for storing registrations.
REGISTERED_FRAMES_DIR = "registered_frames"
REGISTERED_PLATES_FILE = "registered_plates.txt"
os.makedirs(REGISTERED_FRAMES_DIR, exist_ok=True)

# (Optional) For HTTP-based requests, but we use WebSocket below.
class ImagePayload(BaseModel):
    image: str  # Expecting a base64-encoded JPEG image (with data URI header)

def process_frame(image: np.ndarray) -> list:
    """
    Run YOLO detection on the image, then use EasyOCR (with detail=1)
    to extract text from each detected region.
    Return a list of dictionaries with keys "plate" and "box".
    """
    detected_plates = []
    results = model(image)
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for result in ocr_results:
            bbox, text, conf_ocr = result
            text = text.replace(" ", "")
            if re.fullmatch(r'[A-Z]{2}[0-9]{5}', text) and conf_ocr >= 0.8:
                print(f"OCR detected {text} with confidence {conf_ocr*100:.1f}%")
                # Ensure the same plate is not added twice.
                if not any(d['plate'] == text for d in detected_plates):
                    detected_plates.append({"plate": text, "box": [x1, y1, x2, y2]})
    return detected_plates

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive frame data from client.
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                # Remove any data URI header and get the base64 string.
                base64_image_str = payload.get("image", "").split(",")[-1]
            except Exception as e:
                await websocket.send_json({"error": "Invalid JSON payload."})
                continue

            try:
                image_data = base64.b64decode(base64_image_str)
            except Exception as e:
                await websocket.send_json({"error": "Invalid base64 image data."})
                continue

            try:
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                continue  # Skip if image cannot be opened.

            # Convert image to NumPy array and then to BGR (for OpenCV).
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Offload blocking processing to a separate thread.
            detections = await asyncio.to_thread(process_frame, image)
            
            # Draw bounding boxes around detected license plates.
            for detection in detections:
                box = detection["box"]
                plate = detection["plate"]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, plate, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # --- Show the modified frame in an OpenCV window ---
            cv2.imshow("Incoming Frame", image)
            cv2.waitKey(1)
            
            if detections:
                new_plates = []
                # Check each detected plate against the file.
                if os.path.exists(REGISTERED_PLATES_FILE):
                    with open(REGISTERED_PLATES_FILE, "r") as f:
                        file_contents = f.read()
                else:
                    file_contents = ""
                for detection in detections:
                    plate = detection["plate"]
                    if plate not in file_contents:
                        new_plates.append(plate)
                        # Record the new plate.
                        timestamp = int(time.time())
                        filename = f"{plate}_{timestamp}.jpg"
                        filepath = os.path.join(REGISTERED_FRAMES_DIR, filename)
                        cv2.imwrite(filepath, image)
                        with open(REGISTERED_PLATES_FILE, "a") as f:
                            f.write(f"Plate: {plate}, Frame: {filepath} Time: {timestamp}\n")
                if new_plates:
                    await websocket.send_json({"plates": new_plates})
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        # Clean up: destroy OpenCV window when client disconnects.
        cv2.destroyAllWindows()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)