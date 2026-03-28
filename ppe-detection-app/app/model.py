from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"

model = YOLO(MODEL_PATH)