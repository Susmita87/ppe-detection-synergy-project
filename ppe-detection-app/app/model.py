import os
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor

PPE_MODEL_PATH = "weights/best-stage2.pt"
BASE_YOLO_PATH = "weights/yolo11s.pt"

model = YOLO(PPE_MODEL_PATH)
base_model = YOLO(BASE_YOLO_PATH)

# Download the SigLIP model from HF
model_id = "google/siglip-base-patch16-224"
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "siglip2-base-patch32-256")
            
if not os.path.exists(save_path):
    print(f"Downloading SigLIP to {save_path}..")
    os.makedirs(save_path, exist_ok=True)
    AutoModel.from_pretrained(model_id).save_pretrained(save_path)
    AutoProcessor.from_pretrained(model_id).save_pretrained(save_path)
    print("Done")