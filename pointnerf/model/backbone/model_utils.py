import torch
import torch.nn as nn

def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities using sigmoid function.
    Inputs:
        logits: (B, 1, H, W) torch.Tensor
    Outputs:
        prob_map: (B, 1, H, W) torch.Tensor
    """
    return torch.sigmoid(logits)
    
def probabilities_top_k(prob_map: torch.Tensor, 
                        top_k: int, 
                        threshold: float = 1.) -> torch.Tensor:
    """
    Select the top k probabilities.
    Inputs:
        prob_map: (B, 1, H, W) torch.Tensor
        k: int
        threshold: float (if set to 0., then dense porbability map is returned)
    Outputs:
        prob_map: (B, H, W) torch.Tensor
    """
    if len(prob_map.shape) == 4: prob_map = prob_map.squeeze(1) # (B, H, W)

    B, H, W = prob_map.shape

    threshold = torch.tensor(threshold, device=prob_map.device).repeat(B)

    if top_k > H * W:
        k_threshold = torch.zeros_like(threshold)
    else:
        top_k = torch.tensor(top_k, device=prob_map.device)

        prob_map_1D = prob_map.reshape(B, -1) # (B, H*W)

        One_D = prob_map_1D.shape[1] # H*W

        top_k_percentage = (One_D - top_k - 1) / One_D # Percentage of top k

        k_threshold = prob_map_1D.quantile(top_k_percentage, 
                                            dim=1, 
                                            interpolation='midpoint') # Quantile cut-off value at which all values above threshold are top k
    
    prob_theshold = torch.minimum(threshold, k_threshold)[:,None][:,None]

    prob_map = torch.where(prob_map > prob_theshold, 
                           prob_map, 
                           torch.tensor(0., device=prob_map.device)) # (B, H, W)
    
    return prob_map
    
def prob_map_to_points_scores(prob_map: torch.Tensor, 
                              threshold: float = 0.) -> torch.Tensor:
    """
    Convert a probability map to a set of points and scores.
    Inputs:
        prob_map: (B, H, W) torch.Tensor
        threshold: float
    Outputs:
        points_with_scores: (B, N, 3) torch.Tensor (y, x, score)
    """
    if len(prob_map.shape) == 4: prob_map = prob_map.squeeze(1)

    B = prob_map.shape[0]
        
    points_with_scores = [torch.cat((torch.nonzero(prob_map[i] > threshold, as_tuple=False) + 0.5, 
                                        prob_map[i][torch.nonzero(prob_map[i] > threshold, as_tuple=True)][:, None]),
                                        dim=1) for i in range(B)] # [B, N, 3] 
    points_with_scores = torch.stack(points_with_scores, dim=0) # (B, N, 3) (y, x, score)
    return points_with_scores

def normalise_raw_descriptors(raw_descriptors: torch.Tensor, 
                              scale: float = 1.41) -> torch.Tensor:
    """
    Normalise raw descriptors to unit length.
    Inputs:
        raw_descriptors: (B, C, H, W) torch.Tensor
    Outputs:
        descriptors: (B, C, H, W) torch.Tensor
    """
    return nn.functional.normalize(raw_descriptors, dim=1, p=2) * scale

def dense_normalised_descriptors(raw_descriptors: torch.Tensor,
                                 scale_factor: float = 1.41) -> torch.Tensor:
    """
    Convert normalised descriptors to dense descriptors.
    Inputs:
        normalised_descriptors: (B, C, H, W) torch.Tensor
    Outputs:
        dense_descriptors: (B, H*W, C) torch.Tensor
    """
    B, C, = raw_descriptors.shape[:2]
    dense_descriptors = normalise_raw_descriptors(raw_descriptors, scale_factor)
    dense_descriptors = dense_descriptors.reshape(B, C, -1)
    dense_descriptors = dense_descriptors.permute(0, 2, 1)
    return dense_descriptors
    
def sparse_normalised_descriptors(raw_descriptors: torch.Tensor, 
                                  points_with_scores: torch.Tensor,
                                  scale_factor: float = 1.41) -> torch.Tensor:
    """
    Convert normalised descriptors to sparse descriptors (Only for batch size of 1)
    Inputs:
        normalised_descriptors: (C, H, W) torch.Tensor
        points_with_scores: (N, 3) torch.Tensor (y, x, score)
    Outputs:
        sparse_descriptors: (N, C) torch.Tensor
    """
    assert raw_descriptors.shape[0] == 1, "normalised_descriptors must have batch size of 1"
    if len(raw_descriptors.shape) == 4: raw_descriptors = raw_descriptors.squeeze(0)
    if len(points_with_scores.shape) == 3: points_with_scores = points_with_scores.squeeze(0)

    points = points_with_scores[:, :2].floor().long() # (N, 2)
    sparse_raw_descriptors = raw_descriptors[:, points[:, 0], points[:, 1]].T # (N, C)
    normalised_descriptors = normalise_raw_descriptors(sparse_raw_descriptors, scale_factor) # (N, C)

    return normalised_descriptors