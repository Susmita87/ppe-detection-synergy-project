import torch
from PIL import Image
import numpy as np
from app.config import VLM_MODEL_ID, VLM_PROMPTS, VLM_CONF_THRESHOLD
import cv2
from collections import deque

class VLMValidator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # Smoothning buffer per track_id per gear_type (dor SigLIP)
        self.confidence_history = {}   # {track_id: {gear_type: deque}}
        self.SMOOTH_WINDOW = 5
        self.SKIP_FRAMES = 6      # only re-run SigLIP every N frames
        self.ppe_cache = {}     # {track_id: results}
        self.frame_counter = {}   # {track_id: int}

    def _get_smoothed_conf(self, track_id, gear_type, new_conf):
        if track_id not in self.confidence_history:
            self.confidence_history[track_id] = {}
        if gear_type not in self.confidence_history[track_id]:
            self.confidence_history[track_id][gear_type] = deque(maxlen=self.SMOOTH_WINDOW)
        
        self.confidence_history[track_id][gear_type].append(new_conf)
        return sum(self.confidence_history[track_id][gear_type]) / len(self.confidence_history[track_id][gear_type])
    
    def _enhance_crop(self, crop_bgr):
        """CLAHE enhancement for low light / CCTV footage"""
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced

    def _load_model(self):
        if self.model is None:
            # from transformers import CLIPProcessor, CLIPModel
            from transformers import SiglipProcessor, SiglipModel
            print(f"Loading VLM model: {VLM_MODEL_ID} on {self.device}...")
            # self.model = CLIPModel.from_pretrained(VLM_MODEL_ID).to(self.device)
            # self.processor = CLIPProcessor.from_pretrained(VLM_MODEL_ID)
            self.model = SiglipModel.from_pretrained(VLM_MODEL_ID).to(self.device)
            self.processor = SiglipProcessor.from_pretrained(VLM_MODEL_ID)

    def validate_ppe(self, crop_image, track_id = None, frame_num=0):
        """
        Validates if the person in the crop is wearing PPE using CLIP.
        Returns a dictionary of results.
        """
        self._load_model()

        # Return cached result if within skip window
        if track_id is not None:
            last_frame = self.frame_counter.get(track_id, -999)
            if track_id in self.ppe_cache and (frame_num - last_frame) < self.SKIP_FRAMES:
                return self.ppe_cache[track_id]
        
        # Convert numpy array (OpenCV format BGR) to PIL Image (RGB)
        if isinstance(crop_image, np.ndarray):
            print(f"Crop shape before CLAHE: {crop_image.shape}")
            crop_image = self._enhance_crop(crop_image)
            print(f"CLAHE applied ✅") 
            crop_image = Image.fromarray(crop_image[:, :, ::-1])

        results = {}
        
        for gear_type, prompts in VLM_PROMPTS.items():
            # inputs = self.processor(text=prompts, images=crop_image, return_tensors="pt", padding=True).to(self.device)
            inputs = self.processor(text=prompts, images=crop_image, return_tensors="pt", padding="max_length").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            # probs = logits_per_image.softmax(dim=1)
            probs = torch.sigmoid(logits_per_image)
            
            # Assuming the first prompt is "wearing" and second is "not wearing"
            # wearing_prob = probs[0][0].item()
            # results[gear_type] = wearing_prob > VLM_CONF_THRESHOLD
            # results[f"{gear_type}_confidence"] = wearing_prob

            wearing_prob = probs[0][0].item()
            not_wearing_prob = probs[0][1].item()
            raw_conf = wearing_prob / (wearing_prob + not_wearing_prob + 1e-6)

            # Smooth over last N frames per track
            if track_id is not None:
                smoothed_conf = self._get_smoothed_conf(track_id, gear_type, raw_conf)
            else:
                smoothed_conf = raw_conf
            
            results[gear_type] = smoothed_conf > VLM_CONF_THRESHOLD
            results[f"{gear_type}_confidence"] = round(smoothed_conf, 4)
        
        # Cache results
        if track_id is not None:
            self.ppe_cache[track_id] = results
            self.frame_counter[track_id] = frame_num

        return results

vlm_validator = VLMValidator()
