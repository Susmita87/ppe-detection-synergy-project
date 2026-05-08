from app.model import model, base_model
from app.config import (
    CONF_THRESHOLD,
    CONF_IOU,
    CONF_IMGZ,
    VIOLATION_CLASSES, 
    REQUIRED_GEAR_CLASSES, 
    PERSON_CLASS_ID,
    TRACKER_CONFIG,
    PIPELINE_MODE,
    CLASS_NAMES,
    MAX_PERSON_BOX_SIZE
)
import os
import datetime
import numpy as np
from app.extractor import extractor
from app.vlm_validator import vlm_validator


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
        if (x2 - x1) * (y2 - y1) > MAX_PERSON_BOX_SIZE:
            continue
        
        filtered.append(box)
    
    return filtered



def predict(image, persist=False):
    """
    Run inference on input image and return structured results.
    If persist=True, uses tracking with RE-ID.
    """
    frame_num = 0
    frame_num += 1

    if persist:
        if not os.path.exists(TRACKER_CONFIG):
            print(f"ERROR: Tracker config not found at {TRACKER_CONFIG}")
            # Fallback to default or raise
        base_results = base_model.track(image, persist=True, tracker=TRACKER_CONFIG, conf=CONF_THRESHOLD, embed=None)
    else:
        base_results = base_model(image, conf=CONF_THRESHOLD, embed=None)

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


                # Extract high-quality appearance embedding for Re-ID
                person_embedding = None
                try:
                    feat = extractor.extract(crop)
                    person_embedding = feat.tolist()
                except Exception as e:
                    print(f"Error extracting embedding: {e}")

                # Always add the person
                base_person_det = {
                    "class_id": PERSON_CLASS_ID,
                    "class_name": "Person",
                    "confidence": round(float(base_box.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                    "violation": False,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "track_id": int(base_box.id[0].item()) if (getattr(base_box, "id", None) is not None and len(base_box.id) > 0) else None,
                    "embedding": person_embedding
                }

                if PIPELINE_MODE == "VLM":
                    # --- VLM PIPEINE: VLM (CLIP) → validate PPE ---
                    # vlm_results = vlm_validator.validate_ppe(crop)
                    # Using SigLIP + CLAHE
                    vlm_results = vlm_validator.validate_ppe(crop, track_id=base_person_det["track_id"],frame_num=frame_num)
                    
                    # Map VLM results to detections
                    # Hardhat validation
                    if vlm_results.get("hardhat"):
                        det = {
                            "class_id": 0, "class_name": "Hardhat",
                            "confidence": round(vlm_results["hardhat_confidence"], 3),
                            "bbox": [x1, y1, x2, y2], "violation": False,
                            "timestamp": base_person_det["timestamp"], "track_id": base_person_det["track_id"]
                        }
                        gear.append(det)
                    else:
                        det = {
                            "class_id": 2, "class_name": "NO-Hardhat",
                            "confidence": round(1.0 - vlm_results["hardhat_confidence"], 3),
                            "bbox": [x1, y1, x2, y2], "violation": True,
                            "timestamp": base_person_det["timestamp"], "track_id": base_person_det["track_id"]
                        }
                        explicit_violations.append(det)
                    detections.append(det)

                    # Vest validation
                    if vlm_results.get("vest"):
                        det = {
                            "class_id": 7, "class_name": "Safety Vest",
                            "confidence": round(vlm_results["vest_confidence"], 3),
                            "bbox": [x1, y1, x2, y2], "violation": False,
                            "timestamp": base_person_det["timestamp"], "track_id": base_person_det["track_id"]
                        }
                        gear.append(det)
                    else:
                        det = {
                            "class_id": 4, "class_name": "NO-Safety Vest",
                            "confidence": round(1.0 - vlm_results["vest_confidence"], 3),
                            "bbox": [x1, y1, x2, y2], "violation": True,
                            "timestamp": base_person_det["timestamp"], "track_id": base_person_det["track_id"]
                        }
                        explicit_violations.append(det)
                    detections.append(det)
                    
                    people.append(base_person_det)
                    detections.append(base_person_det)

                else:
                    # --- LEGACY PIPELINE: YOLO Stage 2 ---
                    results = model(crop, conf=CONF_THRESHOLD, iou=CONF_IOU, imgsz=CONF_IMGZ)
                    
                    ppe_person_found = False
                    for r in results:
                        filter_person_boxes(r.boxes)
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            if conf < CONF_THRESHOLD: continue

                            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                            bbox = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                            class_name = CLASS_NAMES.get(cls_id, "Unknown")
                            is_explicit_violation = cls_id in VIOLATION_CLASSES

                            det = {
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": round(conf, 3),
                                "bbox": bbox,
                                "violation": is_explicit_violation,
                                "timestamp": datetime.datetime.utcnow().isoformat(),
                                "track_id": base_person_det["track_id"],
                                "embedding": person_embedding if cls_id == PERSON_CLASS_ID else None
                            }

                            if cls_id == PERSON_CLASS_ID:
                                base_person_det.update(det)
                                ppe_person_found = True
                            elif cls_id in REQUIRED_GEAR_CLASSES:
                                gear.append(det)
                            elif is_explicit_violation:
                                explicit_violations.append(det)
                            
                            if cls_id != PERSON_CLASS_ID:
                                detections.append(det)
                    
                    people.append(base_person_det)
                    detections.append(base_person_det)
                        
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
        missing_items = [CLASS_NAMES[gid] for gid, status in p_has_gear.items() if not status]
        is_p_violating = len(missing_items) > 0
        if is_p_violating:
            p["violation"] = True
            p["missing_gear"] = missing_items
            any_person_missing_gear = True
        else:
            p["missing_gear"] = []

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