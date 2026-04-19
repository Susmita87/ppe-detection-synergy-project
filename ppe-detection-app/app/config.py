#  Inference Settings
CONF_THRESHOLD = 0.6
CONF_IOU = 0.5
CONF_IMGZ = 800

# Pipeline Configuration
# "LEGACY": YOLO Stage 1 -> Crop -> YOLO Stage 2
# "VLM": YOLO Stage 1 -> Crop -> VLM Validation
PIPELINE_MODE = "VLM" 

#  Violation Rules Configuration
# Direct violations (e.g. NO-Hardhat, NO-Vest)
VIOLATION_CLASSES = [2, 4]

# Mandatory gear that must be detected if a person is present
REQUIRED_GEAR_CLASSES = [0, 7]  # 0: Hardhat, 7: Safety Vest

# Class ID for a person (matches CLASS_NAMES in inference.py)
PERSON_CLASS_ID = 5

# Inference frequency in video (every Nth frame)
INFERENCE_INTERVAL = 5

# Maximum size of a person box relative to image (0.0 to 1.0)
MAX_PERSON_BOX_SIZE = 0.8


import os

# Base Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = "static_results"
UPLOAD_DIR = "temp_uploads"
BASE_URL = "http://localhost:8000"

# Tracking & Re-ID Settings
TRACKER_CONFIG = os.path.join(BASE_DIR, "trackers", "custom_tracker.yaml")
REID_THRESHOLD = 0.75
VIOLATION_WAIT_TIME = 0.3  # Seconds to wait before alerting for violation in video

# Database Settings
DB_PATH = os.path.join(BASE_DIR, "..", "database", "embeddings.db")

# VLM Settings
VLM_MODEL_ID = "openai/clip-vit-base-patch32"
VLM_PROMPTS = {
    "hardhat": ["a person wearing a hardhat", "a person without a hardhat"],
    "vest": ["a person wearing a safety vest", "a person without a safety vest"]
}
VLM_CONF_THRESHOLD = 0.5

# Email Settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# Class mapping (update based on your dataset.yaml)
CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}