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
        out_axes=(0, 0, 0, 0, 0, 0),
    )

    loss_0, loss_1, precision, recall, correct_mask_0, correct_mask_1 = batched_total_loss(
        desc_0,
        desc_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
        block_size,
    )

    return loss_0.mean(), loss_1.mean(), precision.mean(), recall.mean(), correct_mask_0, correct_mask_1


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
        jax_device: str = "cuda:0",
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

        prob_map_0, prob_map_1 = logits_to_probabilities(logits_0), logits_to_probabilities(logits_1)

        prob_map_0, prob_map_1 = self.flatten(prob_map_0), self.flatten(prob_map_1)

        logits_0, logits_1 = self.flatten(logits_0), self.flatten(logits_1)

        desc_loss, keypoint_loss, precision, recall, correct_mask_0, correct_mask_1 = total_loss(desc_0,
                                                                                                 desc_1,
                                                                                                 corr_0,
                                                                                                 corr_1,
                                                                                                 logits_0,
                                                                                                 logits_1,
                                                                                                 block_size=self._block_size,
                                                                                                 jax_device=self._jax_device)
        
        m_points_0, m_points_1 = torch.where(correct_mask_0.bool(), prob_map_0, torch.tensor(0., device=prob_map_0.device)),\
                                 torch.where(correct_mask_1.bool(), prob_map_1, torch.tensor(0., device=prob_map_1.device))
        
        m_points_0, m_points_1 = self.unflatten(m_points_0, self._shape), self.unflatten(m_points_1, self._shape)

        m_points_0, m_points_1 = prob_map_to_points_scores(m_points_0), prob_map_to_points_scores(m_points_1)
        
        m_points_0, m_points_1 = m_points_0[:,:,:2].floor().long(), m_points_1[:,:,:2].floor().long()

        return desc_loss, keypoint_loss, precision, recall, m_points_0, m_points_1