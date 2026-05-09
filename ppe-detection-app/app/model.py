import os
import mlflow
from ultralytics import YOLO

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
            return YOLO(local_path)
        except Exception as e:
            print(f"Failed to download MLflow artifact: {e}")
            raise
    else:
        return YOLO(model_path)

PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "weights/best-stage2.pt")
BASE_YOLO_PATH = "weights/yolo11s.pt"

model = load_yolo_model(PPE_MODEL_PATH)
base_model = YOLO(BASE_YOLO_PATH)