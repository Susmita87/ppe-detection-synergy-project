from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import os
import uuid
import base64
import time
from app.inference import predict
from app.email_utils import send_email_alert

app = FastAPI(title="PPE Detection API")

# 🔹 Mount static directory for processed results (preserved for images)
STATIC_DIR = "static_results"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=STATIC_DIR), name="results")

# 🔹 CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 🔹 Helper: Draw bounding boxes on frame
def draw_detections(img, detections):
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        is_violation = det["violation"]

        color = (0, 0, 255) if is_violation else (0, 255, 0)
        label = f"{class_name} ({confidence})"

        # 🔹 Draw box
        p1 = (int(bbox["x1"]), int(bbox["y1"]))
        p2 = (int(bbox["x2"]), int(bbox["y2"]))
        cv2.rectangle(img, p1, p2, color, 2)

        # 🔹 Draw background for text
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(img, p1, (p1[0] + text_w, p1[1] - text_h - 10), color, -1)

        # 🔹 Draw text
        cv2.putText(img, label, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


# 🔹 Helper: Check file type
def is_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def is_video(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov"))


# 🔹 Image Processing
async def process_image(file: UploadFile):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict(img)
    
    # 🔹 Draw detections on the image
    processed_img = draw_detections(img, result["detections"])

    # 🔹 Encode into base64 for direct return
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        **result,
        "processed_image": f"data:image/jpeg;base64,{img_base64}"
    }


# 🔹 Video Stream Generator
def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    current_detections = []
    current_status = "SAFE"
    status_color = (0, 255, 0)

    violation_start_time = None
    last_email_time = 0
    EMAIL_COOLDOWN = 60 #seconds (Min time between 2 emails)
    VIOLATION_THRESHOLD = 10 # seconds (Time duration of violation)

    if not cap.isOpened():
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        
        # 🔹 Inference every 5th frame as requested
        if frame_count % 5 == 0:
            result = predict(frame)
            current_detections = result["detections"]
            
            # Update status for this window
            if result["violations_detected"]:
                # 🔹 Start timer if first detection
                if violation_start_time is None:
                    violation_start_time = current_time
                    email_sent_for_violation = False

                # 🔹 Check if violation persists long enough
                elif current_time - violation_start_time >= VIOLATION_THRESHOLD:
                    current_status = "VIOLATION DETECTED"
                    status_color = (0, 0, 255)
                
                    # 🔥 Send email once per violation
                    #if current_time - last_email_time > EMAIL_COOLDOWN:
                    if not email_sent_for_violation:
                        #print("🚨 Email function triggered")
                        #send_email_alert(frame)
                        print("🚨 Sending email alert...")
                        send_email_alert(frame)
                        email_sent_for_violation = True
                        #last_email_time = current_time

                else:
                    current_status = "MONITORING..."
                    status_color = (0, 165, 255)  # orange

            else:
                # 🔹 Reset if no violation
                violation_start_time = None
                email_sent_for_violation = False
                current_status = "SAFE"
                status_color = (0, 255, 0)
            
            # Log for the user in console
            print(f"Frame {frame_count}: {current_status} | Detections: {len(current_detections)}")

        # Draw detections
        frame = draw_detections(frame, current_detections)
        
        # 🔹 Draw Status Banner on top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, f"STATUS: {current_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"FRAME: {frame_count}", (frame.shape[1] - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        frame_count += 1

        # 🔹 Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()


# 🔹 Video Processing (Fast Return for Streaming)
async def process_video_init(file: UploadFile):
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_in.mp4"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    # Save original video temporarily
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    return {
        "type": "video",
        "file_id": file_id,
        "stream_url": f"http://localhost:8000/stream/{file_id}"
    }


# 🔹 Streaming Endpoint
@app.get("/stream/{file_id}")
async def stream_video(file_id: str):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_in.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return StreamingResponse(gen_frames(video_path), 
                             media_type="multipart/x-mixed-replace; boundary=frame")


# 🔹 Main API Endpoint
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    filename = file.filename

    if is_image(filename):
        result = await process_image(file)
        return {
            "type": "image",
            **result
        }

    elif is_video(filename):
        # 🔹 Return stream URL immediately instead of processing whole video
        result = await process_video_init(file)
        return result

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload image or video."
        )


# 🔹 Health Check
@app.get("/")
def health():
    return {"status": "API is running 🚀"}