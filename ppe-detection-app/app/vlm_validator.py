import torch
from PIL import Image
import numpy as np
from app.config import VLM_MODEL_ID, VLM_PROMPTS, VLM_MARGIN_THRESHOLD, LOW_LIGHT_THRESHOLD, USE_GAMMA
import cv2

class VLMValidator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def _apply_gamma_correction(self, image, gamma=1.5):
        """Applies gamma correction to brighten the image"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _load_model(self):
        if self.model is None:
            from transformers import CLIPProcessor, CLIPModel
            print(f"Loading VLM model: {VLM_MODEL_ID} on {self.device}...")
            self.model = CLIPModel.from_pretrained(VLM_MODEL_ID).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(VLM_MODEL_ID)

    def validate_ppe(self, crop_image):
        """
        Validates if the person in the crop is wearing PPE using an ensemble of CLIP prompts.
        Returns a dictionary of results.
        """
        self._load_model()
        
        # Preprocessing: Convert numpy array (OpenCV format BGR) to PIL Image (RGB)
        if isinstance(crop_image, np.ndarray):
            if USE_GAMMA:
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                
                # Tiered Gamma Logic
                # < 30: Gamma 1.8 - 2.2
                # 30-50: Gamma 1.4 - 1.8
                gamma_to_apply = None
                if avg_brightness < 30:
                    gamma_to_apply = 2.2  # Midpoint of 1.8 - 2.2
                elif avg_brightness < LOW_LIGHT_THRESHOLD: # Threshold is 50
                    gamma_to_apply = 1.8  # Midpoint of 1.4 - 1.8
                
                if gamma_to_apply:
                    print(f"Low light detected ({avg_brightness:.1f} < {LOW_LIGHT_THRESHOLD}). Applying Tiered Gamma Correction (gamma={gamma_to_apply})...")
                    crop_image = self._apply_gamma_correction(crop_image, gamma=gamma_to_apply)
            
            crop_image = Image.fromarray(crop_image[:, :, ::-1])

        results = {}
        
        for gear_type, ensemble in VLM_PROMPTS.items():
            from app.config import VLM_MARGIN_THRESHOLD
            # Combine all positive and negative prompts into one batch
            all_prompts = ensemble["wearing"] + ensemble["not_wearing"]
            num_pos = len(ensemble["wearing"])
            
            inputs = self.processor(text=all_prompts, images=crop_image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # logits_per_image is [1, num_prompts]
            logits = outputs.logits_per_image[0]
            
            # Average the logits for positive and negative prompts
            avg_wearing_score = torch.mean(logits[:num_pos]).item()
            avg_not_wearing_score = torch.mean(logits[num_pos:]).item()
            
            margin = avg_wearing_score - avg_not_wearing_score
            
            results[gear_type] = margin > VLM_MARGIN_THRESHOLD
            results[f"{gear_type}_confidence"] = round(margin, 4)

        return results

vlm_validator = VLMValidator()
