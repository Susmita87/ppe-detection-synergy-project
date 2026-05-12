import os
import mlflow
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv

load_dotenv()

def load_yolo_model(model_path):
    """
    Loads a YOLO model from a local path or an MLflow artifact URI.
    """
    if model_path.startswith("mlflow-artifacts:/") or model_path.startswith("runs:/"):
        print(f"Downloading MLflow artifact: {model_path}")
        download_dir = os.path.join("weights", "mlflow_downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path, dst_path=download_dir)
            print(f"Model downloaded to: {local_path}")
            try:
                return YOLO(local_path)
            except Exception as e:
                print(f"WARNING: Downloaded model invalid ({e}). Falling back to local model...")
                return YOLO("weights/best-stage2-latest.pt")
        except Exception as e:
            print(f"WARNING: MLflow download failed ({e}). Falling back to local model...")
            return YOLO("weights/best-stage2-latest.pt")
    else:
        return YOLO(model_path)

# Globals initialized during startup
model = None
base_model = None

def init_models():
    """
    Initializes all models (YOLO and VLM) during app startup.
    """
    from app.config import PIPELINE_MODE, VLM_MODEL_ID
    global model, base_model
    
    # 1. Initialize Base YOLO Model first (it's fast and local)
    BASE_YOLO_PATH = "weights/yolo11s.pt"
    print(f"Initializing Base YOLO Model: {BASE_YOLO_PATH}")
    base_model = YOLO(BASE_YOLO_PATH)

    # 2. Initialize PPE Model (might involve slow download)
    PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "weights/best-stage2.pt")
    if PIPELINE_MODE != "VLM":
        print(f"Initializing PPE Model: {PPE_MODEL_PATH}")
        model = load_yolo_model(PPE_MODEL_PATH)
    else:
        print("Skipping PPE model loading (PIPELINE_MODE=VLM)")

    # 2. Handle VLM/SigLIP Model Download if needed
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "siglip2-base-patch32-256")
    # Check for core files to ensure download completed
    required_files = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"]
    is_complete = os.path.exists(save_path) and any(os.path.exists(os.path.join(save_path, f)) for f in required_files)

    if not is_complete:
        print(f"Downloading VLM Model to {save_path} (this is a ~1GB download)...")
        try:
            from transformers import SiglipModel, SiglipProcessor
            os.makedirs(save_path, exist_ok=True)
            hf_model_id = "google/siglip2-base-patch32-256"
            print(f"Fetching from HuggingFace: {hf_model_id}")
            SiglipModel.from_pretrained(hf_model_id).save_pretrained(save_path)
            SiglipProcessor.from_pretrained(hf_model_id).save_pretrained(save_path)
            print("VLM Model download complete.")
        except Exception as e:
            print(f"Error downloading VLM Model: {e}")
            print("The app will continue, but VLM validation may fail.")
    
    print("All models loaded successfully!")