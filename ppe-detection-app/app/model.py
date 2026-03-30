from ultralytics import YOLO

MODEL_PATH = "weights/best-v1.pt"

model = YOLO(MODEL_PATH)