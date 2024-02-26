import torch
from typing import Literal
from silkprp.model.backbone.model_utils import *

# Parts of this script is taken from SiLK - Simple Learned Keypoints [https://github.com/facebookresearch/silk]

    
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
    return  1.0 - match_prob_dist


def match_nn_double(match_prob_dist: torch.Tensor,
                    dist_thresh: float = None,
                    max_ratio: float = None,
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
    
    if dist_thresh is not None:
        mask = torch.lt(match_prob_dist[indices_0, indices_1], dist_thresh) # distance thresholding
        indices_0 = indices_0[mask]
        indices_1 = indices_1[mask]

    if max_ratio is not None:
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


def match(desc_0: torch.Tensor,
          desc_1: torch.Tensor,
          matching_method: Literal["double_softmax", "ratio_test", "mnn"] = "double_softmax",
          dist_thresh: float = None,
          max_ratio: float = None,
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
        desc_0 = torch.nn.functional.normalize(desc_0, dim=1, p=2)
        desc_1 = torch.nn.functional.normalize(desc_1, dim=1, p=2)
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    
    elif matching_method == "mnn":
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    
    else:
        raise NotImplementedError(f"Matching method {matching_method} not implemented.")

    matches = match_nn_double(distance, dist_thresh, max_ratio, cross_check)

    return matches, distance[matches[:, 0], matches[:, 1]]


def extract_matched_points(kpts: torch.Tensor,
                           matches: torch.Tensor,
                           bias: float) -> torch.Tensor:
    """
    Extract the keypoints from the matches.
    Inputs:
        kpts: (N, 2) torch.Tensor
        matches: (M) torch.Tensor
    Outputs:
        m_kpts: (M, 2) torch.Tensor
    """
    if len(kpts.shape) == 3:
        kpts = kpts.squeeze()
    kpts[:,:2] = kpts[:,:2] + bias
    m_kpts = kpts[matches]
    matched_kpts = m_kpts[:, :2]
    matched_kpts = matched_kpts[:, [1, 0]]
    return matched_kpts


def matcher(desc_0: torch.Tensor,
            desc_1: torch.Tensor,
            logits_0: torch.Tensor,
            logits_1: torch.Tensor,
            matching_method: Literal["double_softmax", "ratio_test", "mnn"] = "double_softmax",
            top_k: int = 10000,
            dist_thresh: float = None,
            max_ratio: float = None,
            temperature: float = 0.1,
            scale_factor: float = 1.41,
            bias: float = 9.0,
            cross_check: bool = True):
    """
    Wrapper function for the matching methods.
    Inputs:
        matching_method: Literal["double_softmax", "ratio_test" , "mnn"]
        dist_thresh: float
        cross_check: bool
    Outputs:
        dict: {"kpts_0": (N, 3) torch.Tensor,
                "kpts_1": (N, 3) torch.Tensor,
                "m_kpts_0": (M, 2) torch.Tensor,
                "m_kpts_1": (M, 2) torch.Tensor,
                "confidence": (M) torch.Tensor}
    """
    
    prob_0, prob_1 = logits_to_probabilities(logits_0), logits_to_probabilities(logits_1)

    prob_0_top_k, prob_1_top_k = probabilities_top_k(prob_0, top_k), probabilities_top_k(prob_1, top_k)

    kpts_0, kpts_1 = prob_map_to_points_scores(prob_0_top_k), prob_map_to_points_scores(prob_1_top_k)
    
    sparse_desc_0, sparse_desc_1 = sparse_normalised_descriptors(desc_0, kpts_0, scale_factor), sparse_normalised_descriptors(desc_1, kpts_1, scale_factor)
    
    matches, confidence = match(sparse_desc_0, sparse_desc_1, matching_method, dist_thresh, max_ratio, temperature, cross_check)

    m_kpts0, m_kpts1 = extract_matched_points(kpts_0, matches[:, 0], bias), extract_matched_points(kpts_1, matches[:, 1], bias)
    
    return {"kpts_0": kpts_0[:, [1, 0, 2]].squeeze().detach().cpu().numpy(),
            "kpts_1": kpts_1[:, [1, 0, 2]].squeeze().detach().cpu().numpy(),
            "m_kpts_0": m_kpts0.detach().cpu().numpy(),
            "m_kpts_1": m_kpts1.detach().cpu().numpy(),
            "confidence": confidence.detach().cpu().numpy()}