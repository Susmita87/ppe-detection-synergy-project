# 🔹 Inference Settings
CONF_THRESHOLD = 0.3

# 🔹 Violation Rules Configuration
# Direct violations (e.g. NO-Hardhat, NO-Vest)
VIOLATION_CLASSES = [2, 4]

# Mandatory gear that must be detected if a person is present
REQUIRED_GEAR_CLASSES = [0, 7]  # 0: Hardhat, 7: Safety Vest

# Class ID for a person
PERSON_CLASS_ID = 5