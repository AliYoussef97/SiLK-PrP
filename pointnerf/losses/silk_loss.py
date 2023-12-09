# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This script is taken from SiLK - Simple Learned Keypoints [https://github.com/facebookresearch/silk]


import math
from typing import Optional
import torch
import jax
import pointnerf.losses.jax_loss as jax_loss
from pointnerf.losses.jax_functions import jax2torch
from pointnerf.model.backbone.model_utils import logits_to_probabilities, probabilities_top_k, prob_map_to_points_scores

positions_to_unidirectional_correspondence = jax2torch(
    jax.jit(
        jax.vmap(
            jax_loss.positions_to_unidirectional_correspondence,
            in_axes=(0, None, None, None, None),
            out_axes=0,
        ),
        static_argnames=["ordering"],
    ),
    backward_pass=False,
)

keep_mutual_correspondences_only = jax2torch(
    jax.jit(
        jax.vmap(
            jax_loss.keep_mutual_correspondences_only,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )
    ),
    backward_pass=False,
)


def total_loss_reduction(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    block_size=None,
):
    batched_total_loss = jax.vmap(
        jax_loss.total_loss,
        in_axes=(0, 0, 0, 0, 0, 0, None),
        out_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )

    loss_0, loss_1, precision, recall, correct_mask_0, correct_mask_1, similarity_0, similarity_1 = batched_total_loss(desc_0,
                                                                                                                       desc_1,
                                                                                                                       corr_0,
                                                                                                                       corr_1,
                                                                                                                       logits_0,
                                                                                                                       logits_1,
                                                                                                                       block_size,
                                                                                                                    )

    return loss_0.mean(), loss_1.mean(), precision.mean(), recall.mean(), correct_mask_0, correct_mask_1, similarity_0, similarity_1


total_loss = jax2torch(
    jax.jit(
        total_loss_reduction,
        static_argnames=("block_size"),
    )
)


class Loss(torch.nn.Module):
    def __init__(
        self,
        block_size: Optional[int] = None,
        jax_device: str = "cuda:1",
        temperature: float = 0.1,
        shape: tuple = (462, 622)
    ) -> None:
        super().__init__()

        self._block_size = block_size
        self._jax_device = jax_device
        self._temperature_sqrt_inv = 1.0 / math.sqrt(temperature)
        self._shape = shape
    
    @staticmethod
    def flatten(x):
        B,C,_,_ = x.shape
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1).contiguous()
        return x
    
    @staticmethod
    def unflatten(x, shape):
        B, _ = x.shape
        H, W = shape
        x = x.reshape(B, H, W)
        x = x.unsqueeze(1)
        return x

    def __call__(
        self,
        desc_0,
        desc_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
    ):
        
        desc_0, desc_1 = self.flatten(desc_0), self.flatten(desc_1)
        desc_0 = desc_0 * self._temperature_sqrt_inv
        desc_1 = desc_1 * self._temperature_sqrt_inv

        # Sigmoid on logits
        prob_map_0, prob_map_1 = logits_to_probabilities(logits_0), logits_to_probabilities(logits_1)

        prob_map_0, prob_map_1 = self.flatten(prob_map_0), self.flatten(prob_map_1)
        prob_map_0, prob_map_1 = prob_map_0.squeeze(-1), prob_map_1.squeeze(-1)

        logits_0, logits_1 = self.flatten(logits_0), self.flatten(logits_1)
        logits_0, logits_1 = logits_1.squeeze(-1), logits_1.squeeze(-1)
        
        desc_loss, keypoint_loss, precision, recall, correct_mask_0, correct_mask_1, similarity_0, similarity_1 = total_loss(desc_0,
                                                                                                                             desc_1,
                                                                                                                             corr_0,
                                                                                                                             corr_1,
                                                                                                                             logits_0,
                                                                                                                             logits_1,
                                                                                                                             block_size=self._block_size,
                                                                                                                             jax_device=self._jax_device)
        # All indicies of shape (H, W) flattened
        flat_indexes = torch.arange(start=0, end=int(self._shape[0]*self._shape[1]), 
                                    dtype = torch.long, device=prob_map_0.device).unsqueeze(0)
        
        # Only keep the indices of the correct matches
        correct_mask_0 = torch.where(correct_mask_0.bool(), flat_indexes, torch.tensor(-1, dtype=torch.long, device=prob_map_0.device))
        correct_mask_1 = torch.where(correct_mask_1.bool(), flat_indexes, torch.tensor(-1, dtype=torch.long, device=prob_map_1.device))
        
        # Keep mutual correct matches only
        correct_mask_0, correct_mask_1 = keep_mutual_correspondences_only(correct_mask_0, correct_mask_1)

        # If no mutual correct matches, return None
        if torch.all(correct_mask_0 == -1):
            m_points_0 = m_points_1 = None
            confidence = None
            return desc_loss, keypoint_loss, precision, recall, m_points_0, m_points_1, confidence

        else:
            # Keep points from flattened probability map with correct matches
            prob_map_0, prob_map_1 = torch.where(correct_mask_0 != -1, prob_map_0, torch.tensor(0., dtype=torch.float32, device=prob_map_0.device)),\
                                     torch.where(correct_mask_1 != -1, prob_map_1, torch.tensor(0., dtype=torch.float32, device=prob_map_1.device))   

            # Unflatten probability to shape (B, 1, H, W)
            prob_map_0, prob_map_1 = self.unflatten(prob_map_0, self._shape), self.unflatten(prob_map_1, self._shape)

            # Convert probability map to points and scores (B, N, 3)
            m_points_0, m_points_1 = prob_map_to_points_scores(prob_map_0), prob_map_to_points_scores(prob_map_1)
            
            # Keep points only
            m_points_0, m_points_1 = m_points_0[:,:,:2].floor().long(), m_points_1[:,:,:2].floor().long()

            # Unflatten similarity to shape (B, 1, H, W)
            similarity_0, similarity_1 = self.unflatten(similarity_0, self._shape), self.unflatten(similarity_1, self._shape)

            # Keep similarity where matches are located
            similarity_0 = similarity_0[:,:,m_points_0[:,:,0], m_points_0[:,:,1]].squeeze()

            similarity_1 = similarity_1[:,:,m_points_1[:,:,0], m_points_1[:,:,1]].squeeze()

            confidence = (similarity_0 * similarity_1).unsqueeze(0)

            return desc_loss, keypoint_loss, precision, recall, m_points_0.to(torch.float32), m_points_1.to(torch.float32), confidence