import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Grayscale
from pointnerf.data.data_utils.photometric_augmentation import Photometric_aug
from pointnerf.data.data_utils.homographic_augmentation import Homographic_aug
from pointnerf.settings import DATA_PATH

class NeRF(Dataset):
    def __init__(self, data_config, task = "training", device="cpu") -> None:
        super(NeRF, self).__init__()
        self.config = data_config
        self.device = device
        self.action = "training" if task == "training" else "validation" if task == "validation" else "test"
        self.samples = self._init_dataset()
        self.camera_intrinsic_matrix = self.get_camera_intrinsic(self.config["image_size"],self.config["fov"])        
        self.photometric_aug = Photometric_aug(self.config["augmentation"]["photometric"])
        self.homographic_aug = Homographic_aug(self.config["augmentation"]["homographic"])

    def _init_dataset(self) -> dict:
        """
        Initialise dataset paths.
        Input:
            None
        Output:
            files: dict containing the paths to the images, camera transforms and depth maps.
        """
        data_dir = Path(DATA_PATH, "NeRF", "images", self.action)
        image_paths = list(data_dir.iterdir())
        if self.config["truncate"]:
            image_paths = image_paths[:int(self.config["truncate"]*len(image_paths))]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {"image_paths":image_paths, "names":names}

        camera_transform_dir = Path(DATA_PATH, "NeRF", "camera_transforms", self.action)
        depth_dir = Path(DATA_PATH, "NeRF", "depth", self.action)
        camera_transform_paths = []
        depth_paths = []
        for n in files["names"]:
            ct_p = Path(camera_transform_dir,'{}.npy'.format(n))
            d_p = Path(depth_dir,'{}.npy'.format(n))
            camera_transform_paths.append(str(ct_p))
            depth_paths.append(str(d_p))
        files["camera_transform_paths"] = camera_transform_paths
        files["depth_paths"] = depth_paths

        return files

    def __len__(self) -> int:
        return len(self.samples["image_paths"])
        
    def get_camera_intrinsic(self, shape: tuple, fov: float) -> torch.Tensor:
        '''
        Initialise camera intrinsic matrix.
        Input:
            shape: tuple containing the image size (H,W).
            fov: float containing the field of view.
        Output:
            cam_intrinsic_matrix: (3,3) torch.Tensor
        '''
        H, W = shape

        c_x = W//2
        c_y = H//2

        fov = np.deg2rad(fov)
        F_L = c_y/np.tan(fov/2)

        cam_intrinsic_matrix = np.array([[ F_L,0,c_x], 
                                         [ 0,F_L,c_y], 
                                         [ 0, 0, 1  ]], dtype=np.float32)
        cam_intrinsic_matrix = torch.as_tensor(cam_intrinsic_matrix, dtype=torch.float32, device=self.device)
        
        return cam_intrinsic_matrix

    def axis_transform(self, cam_matrix: np.ndarray) -> np.ndarray:
        '''
        Transorm y and z axis of camera transformation matrix.
        Input:
            cam_matrix: (4,4) np.ndarray
        Output:
            cam_matrix: (4,4) torch.Tensor
        '''
        reverse = np.diag([1, -1, -1, 1])
        cam_matrix =  cam_matrix @ reverse

        return cam_matrix
    
    def get_rotation_translation(self, transformation_matrix: np.ndarray) -> torch.Tensor:
        '''
        Extract rotation and translation from camera transformation matrix.
        Input:
            transformation_matrix: (4,4) np.ndarray
        Output:
            rotation: (3,3) torch.Tensor
            translation: (3,1) torch.Tensor
        '''
        rotation = transformation_matrix[:3, :3]
        rotation = torch.as_tensor(rotation, dtype=torch.float32,device=self.device)

        translation = transformation_matrix[:3, 3].reshape(3, 1)
        translation = torch.as_tensor(translation, dtype=torch.float32,device=self.device)
        return rotation, translation
        
    def generate_random_frame_index(self, random_frame_number: int) -> int:
        '''
        Generate random frame index.
        Input:
            random_frame_number: int
        Output:
            random_frame_index: int
        '''
        lower_bound = 0
        upper_bound = 1000    
        
        if self.config['bound_frame_sampling']:
            ratio = upper_bound - lower_bound

            if random_frame_number == lower_bound:
                frames = np.arange(random_frame_number+0.07*ratio,random_frame_number+0.15*ratio,1)
                return random.choice(frames)
                
            elif random_frame_number == upper_bound-1:
                frames = np.arange(random_frame_number-0.15*ratio,random_frame_number-0.07*ratio,1)
                return random.choice(frames)
                
            elif random_frame_number - 0.15 * ratio < lower_bound:
                frames = np.arange(random_frame_number+0.07*ratio, random_frame_number+0.15*ratio, 1)
                return random.choice(frames)
                
            elif random_frame_number + 0.15 * ratio > upper_bound-1:
                frames = np.arange(random_frame_number-0.15*ratio,random_frame_number-0.07*ratio)
                return random.choice(frames)
                
            else:
                return random.choice(np.concatenate((np.arange(random_frame_number-0.15*ratio,random_frame_number-0.07*ratio,1),
                                                    np.arange(random_frame_number+0.07*ratio,random_frame_number+0.15*ratio,1)),
                                                    axis=0))
        else:
            return random.randint(lower_bound, upper_bound-1)

    def read_image(self, image: str) -> torch.Tensor:
        '''
        Read image from path.
        Input:
            image: str
        Output:
            image: (1,H,W) torch.Tensor
        '''
        image = torchvision.io.read_file(image)
        image = torchvision.io.decode_image(image,torchvision.io.ImageReadMode.RGB)
        return image.to(torch.float32)
    
    def RGB_to_Grayscale(self, image: torch.Tensor) -> torch.Tensor:
        '''
        Convert RGB image to Grayscale.
        Input:
            image: (1, 3, H, W) torch.Tensor
        Output:
            image: (1, 1, H, W) torch.Tensor
        '''
        image = Grayscale(num_output_channels=1)(image)
        return image/255.0
    
    def relative_pose(self, input_transformation: np.ndarray, warped_transformation: np.ndarray) -> dict:
        '''
        Calculate relative pose between two camera transformations.
        Input:
            input_transformation: (4,4) np.ndarray
            warped_transformation: (4,4) np.ndarray
        Output:
            relative_pose: dict, R_1_2: (3,3) torch.Tensor, T_1_2: (3,1) torch.Tensor
        '''
        
        P1_W_C = np.linalg.inv(input_transformation)
        P2_W_C = np.linalg.inv(warped_transformation)

        R1_W_C, T1_W_C = self.get_rotation_translation(P1_W_C)
        R2_W_C, T2_W_C = self.get_rotation_translation(P2_W_C)

        R_1_2 = R2_W_C @ R1_W_C.T
        T_1_2 = T2_W_C - (R2_W_C @ R1_W_C.T @ T1_W_C)

        P_1_2 = torch.cat([R_1_2, T_1_2], dim=-1)

        return P_1_2
    
    def get_curr_name_number(self, s: str) -> tuple:
        name = ''.join([i for i in s if not i.isdigit()])
        number = ''.join([i for i in s if i.isdigit()])
        return name, number

    def __getitem__(self, index: int) -> dict:

        input_image = self.samples["image_paths"][index]  
        input_image = self.read_image(input_image)
        input_name = self.samples["names"][index]
        input_depth = np.load(self.samples["depth_paths"][index])
        input_depth = torch.as_tensor(input_depth, dtype=torch.float32, device=self.device)
        input_transformation = np.load(self.samples["camera_transform_paths"][index])
        input_transformation = self.axis_transform(input_transformation)
        input_rotation, input_translation = self.get_rotation_translation(input_transformation)
        
        curr_name, curr_index = self.get_curr_name_number(input_name)
        random_idx = int(self.generate_random_frame_index(int(curr_index)))
        random_frame_idx = self.samples["names"].index(curr_name+str(random_idx))
        
        warped_image = self.samples["image_paths"][random_frame_idx]
        warped_image = self.read_image(warped_image)    
        warped_name = self.samples["names"][random_frame_idx]
        warped_depth = np.load(self.samples["depth_paths"][random_frame_idx])
        warped_depth = torch.as_tensor(warped_depth, dtype=torch.float32, device=self.device)
        warped_transformation = np.load(self.samples["camera_transform_paths"][random_frame_idx])
        warped_transformation = self.axis_transform(warped_transformation)
        warped_camera_intrinsic_matrix = self.camera_intrinsic_matrix

        # Apply homographic augmentation
        if (self.action == "training") and (self.config["augmentation"]["homographic"]["apply"]):
            prob_h = np.random.rand()
            if prob_h < self.config["augmentation"]["homographic"]["p"]:
                warped_image, warped_depth, warped_transformation, scale = self.homographic_aug(warped_image, warped_depth, warped_transformation)
                warped_camera_intrinsic_matrix = torch.matmul(warped_camera_intrinsic_matrix, scale)

        warped_rotation, warped_translation = self.get_rotation_translation(warped_transformation)

        # Apply photometric augmentation
        if self.action == "training":
            input_image, warped_image = self.photometric_aug(input_image), self.photometric_aug(warped_image)
        
        input_image, warped_image = self.RGB_to_Grayscale(input_image), self.RGB_to_Grayscale(warped_image)
        input_image, warped_image = input_image.squeeze(), warped_image.squeeze()
        
        gt_relative_pose = self.relative_pose(input_transformation, warped_transformation)

        data = {"raw":{'image':input_image,
                       'input_depth':input_depth,
                       'input_rotation':input_rotation,
                       'input_translation':input_translation},
                "warp":{'image':warped_image,
                        'warped_depth':warped_depth,
                        'warped_rotation':warped_rotation,
                        'warped_translation':warped_translation},
                "name":input_name,
                "warped_name":warped_name,
                "gt_relative_pose":gt_relative_pose,
                "camera_intrinsic_matrix":self.camera_intrinsic_matrix,
                "warped_camera_intrinsic_matrix":warped_camera_intrinsic_matrix}

        return data
    
    def batch_collator(self, batch: list) -> dict:
        '''
        Collate batch of data.
        Input:
            batch: list of data
        Output:
            output: dict of batched data
        '''
    
        images = torch.stack([item['raw']['image'].unsqueeze(0) for item in batch]) # size=(batch_size,1,H,W)
        
        input_depths = torch.stack([item['raw']['input_depth'] for item in batch]) # size=(batch_size,H,W)
    
        input_rotations = torch.stack([item['raw']['input_rotation'] for item in batch]) # size=(batch_size,3,3)
    
        input_translations = torch.stack([item['raw']['input_translation'] for item in batch]) # size=(batch_size,3,1)
    
        warped_images = torch.stack([item['warp']['image'].unsqueeze(0) for item in batch]) # size=(batch_size,1,H,W)

        warped_depths = torch.stack([item['warp']['warped_depth'] for item in batch]) # size=(batch_size,H,W)

        warped_rotations = torch.stack([item['warp']['warped_rotation'] for item in batch]) # size=(batch_size,3,3)

        warped_translations = torch.stack([item['warp']['warped_translation'] for item in batch]) # size=(batch_size,3,1)

        input_names = [item['name'] for item in batch] 

        warped_names = [item["warped_name"] for item in batch] 
            
        intrinsic_matrix = torch.stack([item['camera_intrinsic_matrix'] for item in batch]) # size=(batch_size,3,3)

        warped_intrinsic_matrix = torch.stack([item['warped_camera_intrinsic_matrix'] for item in batch]) # size=(batch_size,3,3)

        gt_relative_poses = torch.stack([item['gt_relative_pose'] for item in batch]) # size=(batch_size,3,4)
            
        output = {"raw":{'image':images,
                       'input_depth':input_depths,
                       'input_rotation':input_rotations,
                       'input_translation':input_translations},
                 "warp":{'image':warped_images,
                        'warped_depth':warped_depths,
                        'warped_rotation':warped_rotations,
                        'warped_translation':warped_translations},
                 "name":input_names,
                 "warped_name":warped_names,
                 "gt_relative_pose": gt_relative_poses,
                 "camera_intrinsic_matrix":intrinsic_matrix,
                 "warped_camera_intrinsic_matrix":warped_intrinsic_matrix}
        
        return output