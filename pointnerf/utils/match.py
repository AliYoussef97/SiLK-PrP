import torch
from typing import Literal

# Parts of this script is taken from SiLK - Simple Learned Keypoints [https://github.com/facebookresearch/silk]

def ratio_test_sim(desc_0: torch.Tensor,
                   desc_1: torch.Tensor) -> torch.Tensor:
    """
    Compute the ratio test similarity between two sets of descriptors.
    Inputs:
        desc_0: (N0, D) torch.Tensor
        desc_1: (N1, D) torch.Tensor
    Returns:
        match_prob_dist: (N0,N1) torch.Tensor
    """
    desc_0 = torch.nn.functional.normalize(desc_0, p=2, dim=1)
    desc_1 = torch.nn.functional.normalize(desc_1, p=2, dim=1)
    match_prob_dist = torch.matmul(desc_0, desc_1.T)
    return 1 - match_prob_dist
    
def double_softmax_sim(desc_0: torch.Tensor,
                       desc_1: torch.Tensor,
                       temperature: float = 0.1) -> torch.Tensor:
    """
    Compute the double softmax similarity between two sets of descriptors.
    Inputs:
        desc_0: (N0, D) torch.Tensor
        desc_1: (N1, D) torch.Tensor
        temperature: float
    Returns:
        match_prob_dist: (N0, N1) torch.Tensor
    """
    sim = torch.divide(torch.matmul(desc_0, desc_1.T), temperature)
    match_prob_dist = torch.softmax(sim, dim=0) * torch.softmax(sim, dim=1)
    return  1 - match_prob_dist


def match_nn_double(match_prob_dist: torch.Tensor,
                    dist_thresh: float = torch.inf,
                    max_ratio: float = torch.inf,
                    cross_check: bool = True) -> torch.Tensor:
    """
    Compute the nearest neighbor matches from the match probabilities.
    Inputs:
        match_prob_dist: (N0, N1) torch.Tensor
        dist_thresh: float
        cross_check: bool
    Outputs:
        matches: (N, 2) torch.Tensor
    """
    indices_0 = torch.arange(match_prob_dist.shape[0], device=match_prob_dist.device)
    indices_1 = torch.argmin(match_prob_dist, dim=1) # each index in desc_0 has a corresponding index in desc_1

    if cross_check:
        matches_0 = torch.argmin(match_prob_dist, dim=0) # each index in desc_1 has a corresponding index in desc_0
        mask = torch.eq(indices_0, matches_0[indices_1]) # cross-checking
        indices_0 = indices_0[mask] 
        indices_1 = indices_1[mask]
    
    if dist_thresh < torch.inf:
        mask = torch.lt(match_prob_dist[indices_0, indices_1], dist_thresh) # distance thresholding
        indices_0 = indices_0[mask]
        indices_1 = indices_1[mask]

    if max_ratio < torch.inf:
        best_distances = match_prob_dist[indices_0, indices_1]
        match_prob_dist[indices_0, indices_1] = torch.inf
        second_best_indices2 = torch.argmin(match_prob_dist[indices_0], axis=1)
        second_best_distances = match_prob_dist[indices_0, second_best_indices2]
        second_best_distances[second_best_distances == 0] = torch.finfo(
            torch.double
        ).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices_0 = indices_0[mask]
        indices_1 = indices_1[mask]

    matches = torch.vstack((indices_0, indices_1)).T

    return matches


def matcher(desc_0: torch.Tensor,
            desc_1: torch.Tensor,
            matching_method: Literal["double_softmax", "ratio_test", "mnn"] = "double_softmax",
            dist_thresh: float = torch.inf,
            max_ratio: float = torch.inf,
            temperature: float = 0.1,
            cross_check: bool = True) -> torch.Tensor:
    """
    Wrapper function for the matching methods.
    Inputs:
        matching_method: Literal["double_softmax","ratio_test"]
        dist_thresh: float
        cross_check: bool
    Outputs:
        matches: (N, 2) torch.Tensor
        confidence: (N, 1) torch.Tensor
    """
    if matching_method == "double_softmax":
        distance = double_softmax_sim(desc_0, desc_1, temperature)
    
    elif matching_method == "ratio_test":
        distance = ratio_test_sim(desc_0, desc_1)
    
    elif matching_method == "mnn":
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    
    else:
        raise NotImplementedError(f"Matching method {matching_method} not implemented.")

    matches = match_nn_double(distance, dist_thresh, max_ratio, cross_check)

    return matches, distance[matches[:, 0], matches[:, 1]]