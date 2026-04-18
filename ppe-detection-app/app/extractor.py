import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        # Using MobileNetV3 Small as it is lightweight and has strong appearance features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity() # Remove the classification head
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, cv2_img):
        """
        Extracts L2-normalized embedding from a cv2 image (BGR).
        """
        # Convert BGR (cv2) to RGB
        img_rgb = cv2_img[..., ::-1]
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Inference
        features = self.model(input_tensor)
        
        # Flatten and L2 Normalize
        feat = features.cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
            
        return feat

# Global instance
extractor = FeatureExtractor()
