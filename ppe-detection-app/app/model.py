from ultralytics import YOLO

PPE_MODEL_PATH = "weights/best-v1.pt"
BASE_YOLO_PATH = "weights/yolo11s.pt"

model = YOLO(PPE_MODEL_PATH)
base_model = YOLO(BASE_YOLO_PATH)