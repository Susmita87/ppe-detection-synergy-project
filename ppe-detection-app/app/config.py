#  Inference Settings
CONF_THRESHOLD = 0.6
CONF_IOU = 0.5
CONF_IMGZ = 800

# Pipeline Configuration
# "LEGACY": YOLO Stage 1 -> Crop -> YOLO Stage 2
# "VLM": YOLO Stage 1 -> Crop -> VLM Validation
PIPELINE_MODE = "LEGACY" 

#  Violation Rules Configuration
# Direct violations (e.g. NO-Hardhat, NO-Vest)
VIOLATION_CLASSES = [2, 4]

# Mandatory gear that must be detected if a person is present
REQUIRED_GEAR_CLASSES = [0, 7]  # 0: Hardhat, 7: Safety Vest

# Class ID for a person (matches CLASS_NAMES in inference.py)
PERSON_CLASS_ID = 5


import os

# Tracking Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_CONFIG = os.path.join(BASE_DIR, "trackers", "custom_tracker.yaml")

# VLM Settings
VLM_MODEL_ID = "openai/clip-vit-base-patch32"
VLM_PROMPTS = {
    "hardhat": ["a person wearing a hardhat", "a person without a hardhat"],
    "vest": ["a person wearing a safety vest", "a person without a safety vest"]
}