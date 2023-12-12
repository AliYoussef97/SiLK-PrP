import torch
from typing import Optional
from pointnerf.losses.silk_loss import positions_to_unidirectional_correspondence, keep_mutual_correspondences_only

def remove_pts_outside_shape(points: torch.Tensor,
                             shape: torch.Size,
                             device='cpu',
                             return_mask=False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
    """
    Removes keypoints that are outside the shape of the image.
    - Input:
        - points: (N,2) tensor
        - shape: (H,W) tensor
    - Output:
        - points: (N,2) tensor
        - mask: (N,) tensor
    """
    if len(points)!=0:
        H,W = shape
        mask  = (points[:,0] >= 0) & (points[:,0] < H-1) & (points[:,1] >= 0) & (points[:,1] < W-1)
        if return_mask:
            return points[mask], mask.to(device)
        return points[mask]
    else:
        return points

def compute_keypoint_map(points: torch.Tensor,
                         shape: torch.Size,
                         device='cpu'):
    """
    Computes a keypoint map from a set of keypoints.
    input:
        points: (N,2)
        shape: torch tensor (H,W)
    output:
        kmap: (H,W)
    """
    H, W = shape
    coord = torch.round(points).to(torch.int32)
    mask = (coord[:, 0] >= 0) & (coord[:, 0] < H-1) & (coord[:, 1] >= 0) & (coord[:, 1] < W-1)
    k_map = torch.zeros(shape, dtype=torch.int32, device=device)
    k_map[coord[mask, 0], coord[mask, 1]] = 1
    return k_map
    
def create_meshgrid(
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Generate a coordinate grid for an image."""
        if normalized:
            min_x = -1.0
            max_x = +1.0
            min_y = -1.0
            max_y = +1.0
        else:
            min_x = 0.5
            max_x = width - 0.5
            min_y = 0.5
            max_y = height - 0.5

        xs: torch.Tensor = torch.linspace(
            min_x,
            max_x,
            width,
            device=device,
            dtype=dtype,
        )
        ys: torch.Tensor = torch.linspace(
            min_y,
            max_y,
            height,
            device=device,
            dtype=dtype,
        )

        # generate grid by stacking coordinates
        base_grid: torch.Tensor = torch.stack(
            torch.meshgrid([ys, xs], indexing="ij"), dim=-1
        )  # WxHx2
        return base_grid.unsqueeze(0)  # 1xHxWx2

def warp_points_NeRF(points: torch.Tensor,
                     depth: torch.Tensor,
                     cam_intrinsic_matrix: torch.Tensor,
                     input_rotation: torch.Tensor,
                     input_translation: torch.Tensor, 
                     warp_rotation: torch.Tensor,
                     warp_translation: torch.Tensor, 
                     device='cpu') -> torch.Tensor:
    """
    Warp keypoints from the input frame to a different viewpoint frame.
    - Input
        - points: (N,2) tensor
        - depth: (H,W) tensor
        - cam_intrinsic_matrix: (3,3) tensor
        - input_rotation: (3,3) tensor
        - input_translation: (3,1) tensor
        - warp_rotation: (3,3) tensor
        - warp_translation: (3,1) tensor
    - Output
        - warped_points: (N,2) tensor
    """
    
    if len(points.shape)==0:
        return points

    depth_values_batch = []
    for dp in depth:     
        depth_value_sample = []
        for p in points:
            if int(p[0]) <= 2 or int(p[1]) <= 2 or int(p[0]) >= dp.shape[0]-2 or int(p[1]) >= dp.shape[1]-2:
                # Case where can not create a 5x5 depth patch, close to image border.
                depth_current = dp[int(p[0]),int(p[1])]
                depth_value_sample.append(depth_current)
                continue
            else:
                # Case where can create a 5x5 depth patch.
                depth_values = dp[int(p[0])-2:int(p[0])+3,int(p[1])-2:int(p[1])+3]
                depth_values_flattened = depth_values.flatten()
                min_depth, max_depth = torch.min(depth_values_flattened), torch.max(depth_values_flattened)
                if (max_depth - min_depth) >= 0.03:
                    # Case where there is a large difference in depth values in the 5x5 depth patch.
                    # most likely at the edge of an object.
                    depth_value_sample.append(min_depth)
                else:
                    # Case where there is a small difference in depth values in the 5x5 depth patch.
                    depth_current = dp[int(p[0]),int(p[1])]
                    depth_value_sample.append(depth_current)

        depth_values_batch.append(torch.tensor(depth_value_sample))
    
    depth_values = torch.stack(depth_values_batch).unsqueeze(1).to(device)
    
    points = torch.fliplr(points)
    
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)
    warped_points = torch.tensordot(torch.linalg.inv(cam_intrinsic_matrix), points, dims=([2], [1]))
    warped_points /= torch.linalg.norm(warped_points, dim=(1), keepdim=True)
    warped_points *= depth_values
    warped_points = input_rotation@warped_points + input_translation    
    warped_points = torch.linalg.inv(warp_rotation) @ warped_points - (torch.linalg.inv(warp_rotation) @ warp_translation)
    warped_points = cam_intrinsic_matrix @ warped_points

    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:,:, :2] / warped_points[:,:, 2:]
    warped_points = torch.flip(warped_points,dims=(2,))

    #warped_points = warped_points.squeeze(0)

    return warped_points

def get_correspondences(data: dict,
                        shape: list,
                        bias: float = 9.,
                        device: str = "cuda:0") -> torch.Tensor:
    """
    Get correspondences between images.
    Inputs:
        data: dict
        shape: list
        device: str
    Outputs:
        corr_0: (B,N)
        corr_1: (B,N)
    """
    input_rotation = data["raw"]["input_rotation"]
    input_translation = data["raw"]["input_translation"]
    input_depth = data["raw"]["input_depth"]

    warped_rotation = data["warp"]["warped_rotation"]
    warped_translation = data["warp"]["warped_translation"]
    warped_depth = data["warp"]["warped_depth"]

    K0 = K1 = data["camera_intrinsic_matrix"]

    H_d, W_d = shape

    B = data["raw"]["input_depth"].shape[0]

    positions = create_meshgrid(
                H_d,
                W_d,
                device = device,
                normalized=False,
                dtype=None,
            )
    positions = positions.expand(B, -1, -1, -1)  # add batch dim
    positions = positions.reshape(B, -1, 2)
    positions += bias

    
    positions_forward = warp_points_NeRF(points=positions.squeeze(0),
                                         depth=input_depth,
                                         cam_intrinsic_matrix=K0,
                                         input_rotation=input_rotation,
                                         input_translation=input_translation,
                                         warp_rotation=warped_rotation,
                                         warp_translation=warped_translation,
                                         device=device)
    positions_forward -= bias

    
    positions_backward = warp_points_NeRF(points=positions.squeeze(0),
                                          depth=warped_depth,
                                          cam_intrinsic_matrix=K1,
                                          input_rotation=warped_rotation,
                                          input_translation=warped_translation,
                                          warp_rotation=input_rotation,
                                          warp_translation=input_translation,
                                          device=device)
    positions_backward -= bias

    corr_forward = positions_to_unidirectional_correspondence(positions_forward, W_d, H_d, 1, ordering="yx")

    corr_backward = positions_to_unidirectional_correspondence(positions_backward, W_d, H_d, 1, ordering="yx")
    
    corr_0, corr_1 = keep_mutual_correspondences_only(corr_forward, corr_backward)

    return corr_0, corr_1