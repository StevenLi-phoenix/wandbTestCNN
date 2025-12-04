import io
import os
from typing import IO, Union

import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN
from utils import get_device


class Predictor:
    def __init__(self):
        self.device = get_device()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.model = self.load_model()
        self.model.eval()
        
    def load_model(self, model_path: str = "model.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device)
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(image)
        
    def predict_file(self, file_path: Union[str, IO[bytes], io.BytesIO]) -> torch.Tensor:
        with Image.open(file_path) as image:
            image = image.convert("L")
            image = image.resize((28, 28))
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.predict(tensor)
