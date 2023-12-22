import torch
import kornia

def normalize_keypoints(kpts: torch.Tensor,
                        K: torch.Tensor) -> torch.Tensor:
    """
    Normalize the keypoints with the camera intrinsics.
    Inputs:
        kpts: (B, N, 2) torch.Tensor
        K: (B, 3, 3) torch.Tensor
    Returns:
        mkpts: (B, N, 2) torch.Tensor
    """
    assert len(kpts.shape) == 3 and len(K.shape) == 3, "keypoints and Intrinsics must have shape (B, N, 2) and (B, 3, 3)"
    
    f_x, f_y, c_x, c_y = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]

    kpts[:, :, 0] = (kpts[:, :, 0] - c_x[:, None]) / f_x[:, None]

    kpts[:, :, 1] = (kpts[:, :, 1] - c_y[:, None]) / f_y[:, None]

    return kpts
    

def pose_estimation(mkpts_0: torch.Tensor,
                    mkpts_1: torch.Tensor,
                    K_0: torch.Tensor,
                    K_1: torch.Tensor,
                    P_GT: torch.Tensor = None) -> torch.Tensor:
    """
    Estimate the relative pose between two images.
    Inputs:
        mkpts_0: (B, N, 2) torch.Tensor
        mkpts_1: (B, N, 2) torch.Tensor
        K_0: (B, 3, 3) torch.Tensor
        K_1: (B, 3, 3) torch.Tensor
    Outputs:
        P_est: (B, 3, 4) torch.Tensor
    """
    # Normalize the keypoints
    if  mkpts_0 is None or mkpts_0.shape[1] < 8:
        return None
    
    mkpts_0 = normalize_keypoints(mkpts_0, K_0)
    mkpts_1 = normalize_keypoints(mkpts_1, K_1)

    intr = torch.eye(3, device=K_0.device).unsqueeze(0)
    F = kornia.geometry.epipolar.find_fundamental(mkpts_0, mkpts_1, None, method='8POINT')
    
    # During training
    if P_GT is not None:
        
        # Compute the 4 possible solutions
        All_R, All_t = kornia.geometry.epipolar.motion_from_essential(F)

        # Compute the relative pose error for each solution and select the best one
        batched_err = torch.ones(size=(P_GT.shape[0], 1), device=P_GT.device) * torch.inf
        P_est = torch.zeros_like(P_GT, device=P_GT.device)

        for curr_R, curr_t in zip(All_R.permute(1, 0, 2, 3).contiguous(), All_t.permute(1, 0, 2, 3).contiguous()):
            
            # Compute the relative pose error for each batch using the current solution
            curr_P = torch.cat([curr_R, curr_t], dim=-1)
            curr_R_err, curr_t_err = relative_pose_error(curr_P, P_GT, train=False)
            current_err = curr_R_err + curr_t_err

            # Update the best solution for each batch if the current solution is better
            mask = (current_err < batched_err).squeeze(-1)
            batched_err[mask] = current_err[mask]
            P_est[mask] = curr_P[mask]

    else:
        R, t, _ = kornia.geometry.epipolar.motion_from_essential_choose_solution(F, intr, intr, mkpts_0, mkpts_1, None)
        P_est = torch.cat([R, t], dim=-1).to(K_0.device)

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
        return torch.tensor(0.0, device=P_GT.device), torch.tensor(0.0, device=P_GT.device)
    
    assert len(P_est.shape) == 3 and len(P_GT.shape) == 3, "Estimated and GT poses must have shape (B, 3, 4)"

    R_est, T_est = P_est[:, :3, :3], P_est[:, :3, -1:]

    R_GT, T_GT = P_GT[:, :3, :3], P_GT[:, :3, -1:]

    err_R = relative_rotation_error(R_est, R_GT)

    err_t = relative_translation_error(T_est, T_GT)

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

    n = torch.linalg.norm(T_est, dim=1) * torch.linalg.norm(T_GT, dim=1)
    t_err = (T_est * T_GT).sum(dim=1)
    t_err = torch.arccos(torch.clip(t_err/n, -1., 1.))
    t_err = torch.abs(t_err)
    
    return t_err