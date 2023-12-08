import torch
import kornia.augmentation as K

class Photometric_aug():
    def __init__(self, config: dict) -> None:
        self.config = config
        self.augmentation = K.ImageSequential()
        for k,v in config["Parameters"].items():
              self.augmentation.append(getattr(K,k)(**v))
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
         image = self.augmentation(image)
         return image