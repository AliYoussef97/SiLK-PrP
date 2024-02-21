import torch
import numpy as np
import albumentations as A
import cv2

class Homographic_aug():
    def __init__(self, config: dict) -> None:
        self.config = config
        
    def get_random_angle(self):
        """
        Returns a random angle between -angle and angle and the corresponding rotation matrix (around z-axis)
        Inputs:
            - angle: None
        Outputs:
            - angle: random angle between -angle and angle
            - Rotation_z: rotation matrix around z-axis
        """
        angle = np.random.randint(-self.config['rotation']['angle'], self.config['rotation']['angle'])
        Rotation_z = np.eye(4)
        Rotation_z[0,0] = np.cos(np.deg2rad(angle))
        Rotation_z[0,1] = -np.sin(np.deg2rad(angle))
        Rotation_z[1,0] = np.sin(np.deg2rad(angle))
        Rotation_z[1,1] = np.cos(np.deg2rad(angle))
        return angle, Rotation_z
    
    def rotate_data(self, image: torch.Tensor, depth: torch.Tensor, transformation: np.array) -> torch.Tensor:
        """
        Rotates the image and depth by a random angle
        Inputs:
            - image: image to rotate
            - depth: depth to rotate
            - transformation: extrinsic transformation matrix to rotate
        Outputs:
            - image: rotated image
            - depth: rotated depth
            - transformation: rotated extrinsic transformation matrix
        """
        angle, rot_z = self.get_random_angle()

        rot = A.augmentations.geometric.rotate.Rotate(limit=[angle,angle],
                                                      interpolation=cv2.INTER_LINEAR,
                                                      value=0.0,
                                                      border_mode=cv2.BORDER_CONSTANT,
                                                      p=1.0)
        
        image = rot(image=image)['image']
        depth = rot(image=depth)['image']

        transformation = transformation @ rot_z

        return image, depth, transformation
    
    def scale_data(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Scales the image and depth by a random factor
        Inputs:
            - image: image to scale
            - depth: depth to scale
        Outputs:
            - image: scaled image
            - depth: scaled depth
            - s: scaling matrix
        """
        s = round(np.random.uniform(self.config['scale']['min'], self.config['scale']['max']), 1)
        scale = A.augmentations.geometric.transforms.Affine(scale=s,
                                                            interpolation=cv2.INTER_LINEAR,
                                                            mode=cv2.BORDER_CONSTANT,
                                                            p=1.0)
        image = scale(image=image)['image']
        depth = scale(image=depth)['image']

        s = torch.tensor(s, dtype=torch.float32)
        s = torch.tensor([s, s, 1.0], dtype=torch.float32)
        s = torch.diag(s)
        
        return image, depth, s
    
    def __call__(self, img: torch.Tensor, depth: torch.Tensor, transformation: np.array) -> torch.Tensor:
        """
        Applies a random homographic transformation to the image and depth
        Inputs:
            - img: image to transform
            - depth: depth to transform
            - transformation: transformation matrix
        Outputs:
            - img: transformed image
            - depth: transformed depth
            - transformation: transformation matrix
        """
        img = img.permute(1,2,0).numpy()
        depth = depth.numpy()

        img, depth, scale_matrix = self.scale_data(img, depth)
        img, depth, transformation = self.rotate_data(img, depth, transformation)

        img = torch.from_numpy(img).permute(2,0,1)
        depth = torch.from_numpy(depth)
        depth = torch.where(depth==0, torch.tensor(torch.inf, dtype=torch.float32), depth)

        return img, depth, transformation, scale_matrix