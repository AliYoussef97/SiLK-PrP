import torch
import albumentations as A

class Photometric_aug():
    def __init__(self, config: dict) -> None:

        self.augmentation = A.Compose(transforms=[], p=config["p"])
        for k,v in config["Parameters"].items():
            self.augmentation.transforms.append(getattr(A,k)(**v))
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
         img = img.permute(1, 2, 0)
         img = img.cpu().numpy()
         img = self.augmentation(image=img)["image"]
         return torch.from_numpy(img).permute(2, 0, 1)

