import torch
import poselib

# This is an experimental script, it is not used in the current version of the project.

def get_camera(K: torch.Tensor) -> dict:
    """
    Get the camera parameters from the intrinsics matrix.
    Inputs:
        K: (3, 3) torch.Tensor
    Outputs:
        camera: dict
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = int(2 * cx), int(2 * cy)
    camera = {"model": "PINHOLE", "width": W, "height": H, "params": [fx, fy, cx, cy]}
    return camera

def pose_estimation(mkpts_0: torch.Tensor,
                    mkpts_1: torch.Tensor,
                    K_0: torch.Tensor,
                    K_1: torch.Tensor,
                    P_GT: torch.Tensor = None,
                    ordering: str = "xy") -> torch.Tensor:
    """
    Estimate the relative pose between two images.
    Inputs:
        mkpts_0: (B, N, 2) torch.Tensor
        mkpts_1: (B, N, 2) torch.Tensor
        K_0: (B, 3, 3) torch.Tensor
        K_1: (B, 3, 3) torch.Tensor
        P_GT: (B, 3, 4) torch.Tensor
        ordering: str (xy or yx)
    Outputs:
        P_est: (B, 3, 4) torch.Tensor
    """
    assert ordering in ["xy", "yx"], "Ordering must be xy or yx"
    if  mkpts_0 is None or mkpts_0.shape[1] < 8:
        return None
    
    dev = mkpts_0.device

    if ordering != "xy":
        mkpts_0 = mkpts_0[:, :, [1,0]]
        mkpts_1 = mkpts_1[:, :, [1,0]]

    mkpts_0 = mkpts_0.squeeze(0).detach().cpu().numpy()
    mkpts_1 = mkpts_1.squeeze(0).detach().cpu().numpy()
    
    K_0 = K_0.squeeze(0).cpu().numpy()
    K_1 = K_1.squeeze(0).cpu().numpy()

    camera_0 = get_camera(K_0)
    camera_1 = get_camera(K_1)
    
    M, info = poselib.estimate_relative_pose(mkpts_0, mkpts_1, camera_0, camera_1, {"max_epipolar_error": 0.5, "success_prob": 0.99999})
    
    R = torch.tensor(M.R, requires_grad=True, dtype=torch.float32, device = dev).reshape(1, 3, 3)
    t = torch.tensor(M.t, requires_grad=True, dtype=torch.float32, device = dev).reshape(1, 3, 1)
    
    P_est = torch.cat([R, t], dim=-1)

    return P_est

def relative_pose_error(P_est: torch.Tensor,
                        P_GT: torch.Tensor,
                        train=False) -> torch.Tensor:
    
    """
    Computes the relative pose error between the estimated and ground truth poses.
    Inputs:
        P_est: (B, 3, 4) torch.Tensor
        P_GT: (B, 3, 4) torch.Tensor
        train: Boolean
    Outputs:
        err: (N) torch.Tensor
    """
    if P_est is None:
        return torch.abs(torch.arccos(torch.tensor(-.5, requires_grad=True if train else False))),\
               torch.abs(torch.arccos(torch.tensor(-.5, requires_grad=True if train else False))) 
    
    assert len(P_est.shape) == 3 and len(P_GT.shape) == 3, "Estimated and GT poses must have shape (B, 3, 4)"
    
    R_est, T_est = P_est[:, :3, :3], P_est[:, :3, 3]

    R_GT, T_GT = P_GT[:, :3, :3], P_GT[:, :3, 3]

    err_R = relative_rotation_error(R_est, R_GT)

    err_t, n_t = relative_translation_error(T_est, T_GT)

    if n_t < 1e-6:
        err_R = torch.zeros(R_GT.shape[0],requires_grad=True if train else False)
        err_t = torch.zeros(R_GT.shape[0],requires_grad=True if train else False)

    if train:
        err_R = torch.mean(err_R)
        err_t = torch.mean(err_t)
    
    return err_R, err_t


def relative_rotation_error(R_est: torch.Tensor,
                            R_GT: torch.Tensor) -> torch.Tensor:
    """
    Computes the relative rotation error between the estimated and ground truth poses.
    Inputs:
        P_est: (B, 3, 3) torch.Tensor
        P_GT: (B, 3, 3) torch.Tensor
        train: Boolean
    Outputs:
        err_R: (B, N) torch.Tensor
    """
    assert len(R_est.shape) == 3 and len(R_GT.shape) == 3, "R_est and R_GT must have shape (B, 3, 3)"
    
    err_R = torch.bmm(R_est.transpose(-1,-2), R_GT) 
    err_R = torch.einsum('bii->b', err_R) # Compute the trace of the dot product for a batch.
    err_R = (err_R - 1.) / 2. 
    err_R = torch.arccos(torch.clip(err_R, -1., 1.)) 
    err_R = torch.abs(err_R) 
    
    return err_R


def relative_translation_error(T_est: torch.Tensor,
                               T_GT: torch.Tensor) -> torch.Tensor:
    """
    Computes the relative translation error between the estimated and ground truth poses.
    Inputs:
    T_est: (B, 3, 1) torch.Tensor
    T_GT: (B, 3, 1) torch.Tensor
    train: Boolean
    Outputs:
        err_t: (B, N) torch.Tensor
    """
    assert len(T_est.shape) == 3 and len(T_GT.shape) == 3, "T_est and T_GT must have shape (B, 3, 1)"

    n = torch.linalg.norm(T_est, dim=-1) * torch.linalg.norm(T_GT, dim=-1)
    t_err = (T_est * T_GT).sum(dim=-1)
    t_err = torch.arccos(torch.clip(t_err/n, -1., 1.))
    t_err = torch.abs(t_err)
    
    return t_err, n