import torch
from PIL import Image
import numpy as np
from app.config import VLM_MODEL_ID, VLM_PROMPTS, VLM_CONF_THRESHOLD

class VLMValidator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def _load_model(self):
        if self.model is None:
            from transformers import CLIPProcessor, CLIPModel
            print(f"Loading VLM model: {VLM_MODEL_ID} on {self.device}...")
            self.model = CLIPModel.from_pretrained(VLM_MODEL_ID).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(VLM_MODEL_ID)

    def validate_ppe(self, crop_image):
        """
        Validates if the person in the crop is wearing PPE using CLIP.
        Returns a dictionary of results.
        """
        self._load_model()
        
        # Convert numpy array (OpenCV format BGR) to PIL Image (RGB)
        if isinstance(crop_image, np.ndarray):
            crop_image = Image.fromarray(crop_image[:, :, ::-1])

        results = {}
        
        for gear_type, prompts in VLM_PROMPTS.items():
            inputs = self.processor(text=prompts, images=crop_image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Assuming the first prompt is "wearing" and second is "not wearing"
            wearing_prob = probs[0][0].item()
            results[gear_type] = wearing_prob > VLM_CONF_THRESHOLD
            results[f"{gear_type}_confidence"] = wearing_prob

        return results

vlm_validator = VLMValidator()
