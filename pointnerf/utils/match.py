import torch

def double_softmax_sim(desc_0: torch.Tensor,
                       desc_1: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
    """
    Compute the double softmax similarity between two sets of descriptors.
    Inputs:
        desc_0: (N0, D) torch.Tensor
        desc_1: (N1, D) torch.Tensor
        temperature: float
    Returns:
        sim: (N0,N1) torch.Tensor
    """
    sim = torch.divide(torch.matmul(desc_0, desc_1.T), temperature)
    match_prob = torch.softmax(sim, dim=0) * torch.softmax(sim, dim=1)
    return  1 - match_prob


def match_nn_double_softmax(match_prob: torch.Tensor,
                            dist_thresh: float = 0.6,
                            cross_check: bool = True) -> torch.Tensor:
    """
    Compute the nearest neighbor matches from the match probabilities.
    Inputs:
        match_prob: (N0, N1) torch.Tensor
        dist_thresh: float
        cross_check: bool
    Outputs:
        matches: (N, 2) torch.Tensor
        confidence: (N, 1) torch.Tensor
    """
    indices_0 = torch.arange(match_prob.shape[0], device=match_prob.device)
    indices_1 = torch.argmin(match_prob, dim=1) # each index in desc_0 has a corresponding index in desc_1

    if cross_check:
        matches_1 = torch.argmin(match_prob, dim=0) # each index in desc_1 has a corresponding index in desc_0
        mask = torch.eq(indices_0, matches_1[indices_1]) # cross-checking
        indices_0 = indices_0[mask] 
        indices_1 = indices_1[mask]
    
    mask = torch.lt(match_prob[indices_0, indices_1], dist_thresh) # distance thresholding
    indices_0 = indices_0[mask]
    indices_1 = indices_1[mask]

    matches = torch.vstack((indices_0,indices_1)).T

    confidence = match_prob[indices_0,indices_1]

    return matches, confidence