import os
import mlflow
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def load_yolo_model(model_path):
    """
    Loads a YOLO model from a local path or an MLflow artifact URI.
    """
    if model_path.startswith("mlflow-artifacts:/") or model_path.startswith("runs:/"):
        print(f"Downloading MLflow artifact: {model_path}")
        # Ensure a local directory exists for downloaded models
        download_dir = os.path.join("weights", "mlflow_downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        # Download the artifact
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path, dst_path=download_dir)
            print(f"Model downloaded to: {local_path}")
            
            # Verify the model can be loaded
            try:
                return YOLO(local_path)
            except Exception as e:
                print(f"WARNING: Downloaded model is corrupted or invalid ({e}). Falling back to local model...")
                return YOLO("weights/best-stage2-latest.pt") # Fallback to a known local file
        except Exception as e:
            print(f"WARNING: Failed to download MLflow artifact ({e}). Falling back to local model...")
            return YOLO("weights/best-stage2-latest.pt")
    else:
        return YOLO(model_path)

# Initialize models as None; they will be loaded during startup
model = None
base_model = None

from app.config import PIPELINE_MODE

def init_models():
    """
    Called during app startup to initialize models.
    """
    global model, base_model
    
    PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "weights/best-stage2.pt")
    BASE_YOLO_PATH = "weights/yolo11s.pt"
    
    if PIPELINE_MODE != "VLM":
        print(f"Initializing PPE Detection Model: {PPE_MODEL_PATH}")
        model = load_yolo_model(PPE_MODEL_PATH)
    else:
        print(f"Skipping PPE model loading (PIPELINE_MODE={PIPELINE_MODE})")
    
    print(f"Initializing Base YOLO Model: {BASE_YOLO_PATH}")
    base_model = YOLO(BASE_YOLO_PATH)
    print("All models loaded successfully!")