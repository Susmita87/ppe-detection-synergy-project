from app.model import model, base_model
from app.config import (
    CONF_THRESHOLD,
    CONF_IOU,
    CONF_IMGZ,
    VIOLATION_CLASSES, 
    REQUIRED_GEAR_CLASSES, 
    PERSON_CLASS_ID,
    TRACKER_CONFIG
)
import os
import datetime

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


def is_overlapping(box1, box2):
    """
    Check if box1 (gear) overlaps box2 (person) significantly
    """
    # Simple check: is the center of box1 inside box2?
    cx = (box1[0] + box1[2]) / 2
    cy = (box1[1] + box1[3]) / 2
    return (box2[0] <= cx <= box2[2]) and (box2[1] <= cy <= box2[3])

def filter_person_boxes(boxes):
    filtered = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Remove overly large boxes
        if (x2 - x1) * (y2 - y1) > 0.8:  # >80% of image
            continue
        
        filtered.append(box)
    
    return filtered


def predict(image, persist=False):
    """
    Run inference on input image and return structured results.
    If persist=True, uses tracking with RE-ID.
    """
    if persist:
        if not os.path.exists(TRACKER_CONFIG):
            print(f"ERROR: Tracker config not found at {TRACKER_CONFIG}")
            # Fallback to default or raise
        base_results = base_model.track(image, persist=True, tracker=TRACKER_CONFIG, conf=CONF_THRESHOLD)
    else:
        base_results = base_model(image, conf=CONF_THRESHOLD)

    detections = []
    
    # Temporary storage for multi-pass logic
    people = []
    gear = []
    explicit_violations = []

    for br in base_results:
        for base_box in br.boxes:
            cls = int(base_box.cls[0])

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, base_box.xyxy[0])
                
                # Ensure valid crop dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue


                results = model(crop, conf=CONF_THRESHOLD, iou=CONF_IOU, imgsz=CONF_IMGZ)
    
                for r in results:
                    filter_person_boxes(r.boxes)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        if conf < CONF_THRESHOLD:
                            print(f"Skipping {CLASS_NAMES.get(cls_id, 'class '+str(cls_id))} with confidence {conf:.3f}")
                            continue

                        # bbox = box.xyxy[0].tolist()
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        bbox = [
                            bx1 + x1,
                            by1 + y1,
                            bx2 + x1,
                            by2 + y1
                        ]
                        class_name = CLASS_NAMES.get(cls_id, "Unknown")
                        is_explicit_violation = cls_id in VIOLATION_CLASSES

                        det = {
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": round(conf, 3),
                            "bbox": bbox, # Keep raw for processing
                            "violation": is_explicit_violation,
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                            "track_id": int(base_box.id[0].item()) if (getattr(base_box, "id", None) is not None and len(base_box.id) > 0) else None
                        }



                        if cls_id == PERSON_CLASS_ID:
                            people.append(det)
                        elif cls_id in REQUIRED_GEAR_CLASSES:
                            gear.append(det)
                        elif is_explicit_violation:
                            explicit_violations.append(det)
                        
                        # Add to main list (formatted for return)
                        detections.append(det)
                        
    # Per-Person Violation Logic
    any_person_missing_gear = False
    
    for p in people:
        p_box = p["bbox"]
        p_has_gear = {gid: False for gid in REQUIRED_GEAR_CLASSES}
        
        # Check if any gear belongs to THIS person
        for g in gear:
            if is_overlapping(g["bbox"], p_box):
                p_has_gear[g["class_id"]] = True
        
        # If this person is missing any mandatory gear
        is_p_violating = not all(p_has_gear.values())
        if is_p_violating:
            p["violation"] = True
            any_person_missing_gear = True

    # Final Violation Logic
    violation_detected = any_person_missing_gear or len(explicit_violations) > 0

    # Clean up bboxes in response to match expected output format
    for d in detections:
        d["bbox"] = {
            "x1": round(d["bbox"][0], 2),
            "y1": round(d["bbox"][1], 2),
            "x2": round(d["bbox"][2], 2),
            "y2": round(d["bbox"][3], 2)
        }

    return {
        "total_detections": len(detections),
        "violations_detected": violation_detected,
        "detections": detections
    }


# def predict(image):
#     """
#     Run inference on input image and return structured results
#     """
#     results = model(image, conf=CONF_THRESHOLD, iou=CONF_IOU, imgsz=CONF_IMGZ)

#     detections = []
    
#     # Temporary storage for multi-pass logic
#     people = []
#     gear = []
#     explicit_violations = []
    
#     for r in results:
#         filter_person_boxes(r.boxes)
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])

#             if conf < CONF_THRESHOLD:
#                 print(f"Skipping {CLASS_NAMES.get(cls_id, 'class '+str(cls_id))} with confidence {conf:.3f}")
#                 continue

#             bbox = box.xyxy[0].tolist()
#             class_name = CLASS_NAMES.get(cls_id, "Unknown")
#             is_explicit_violation = cls_id in VIOLATION_CLASSES

#             det = {
#                 "class_id": cls_id,
#                 "class_name": class_name,
#                 "confidence": round(conf, 3),
#                 "bbox": bbox, # Keep raw for processing
#                 "violation": is_explicit_violation,
#                 "timestamp": datetime.datetime.utcnow().isoformat()
#             }

#             if cls_id == PERSON_CLASS_ID:
#                 people.append(det)
#             elif cls_id in REQUIRED_GEAR_CLASSES:
#                 gear.append(det)
#             elif is_explicit_violation:
#                 explicit_violations.append(det)
            
#             # Add to main list (formatted for return)
#             detections.append(det)

#     # Per-Person Violation Logic
#     any_person_missing_gear = False
    
#     for p in people:
#         p_box = p["bbox"]
#         p_has_gear = {gid: False for gid in REQUIRED_GEAR_CLASSES}
        
#         # Check if any gear belongs to THIS person
#         for g in gear:
#             if is_overlapping(g["bbox"], p_box):
#                 p_has_gear[g["class_id"]] = True
        
#         # If this person is missing any mandatory gear
#         is_p_violating = not all(p_has_gear.values())
#         if is_p_violating:
#             p["violation"] = True
#             any_person_missing_gear = True

#     # Final Violation Logic
#     violation_detected = any_person_missing_gear or len(explicit_violations) > 0
#     # violation_detected = any_person_missing_gear

#     # Clean up bboxes in response to match expected output format
#     for d in detections:
#         d["bbox"] = {
#             "x1": round(d["bbox"][0], 2),
#             "y1": round(d["bbox"][1], 2),
#             "x2": round(d["bbox"][2], 2),
#             "y2": round(d["bbox"][3], 2)
#         }

#     return {
#         "total_detections": len(detections),
#         "violations_detected": violation_detected,
#         "detections": detections
#     }