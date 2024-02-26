# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This script is taken from SiLK - Simple Learned Keypoints [https://github.com/facebookresearch/silk]


import jax
from jax import numpy as jnp
from silkprp.losses.jax_functions import delayed_vjp

def positions_to_unidirectional_correspondence(
    positions,
    width,
    height,
    cell_size,
    ordering="yx",
):
    assert ordering in {"xy", "yx"}

    # positions : Nx2
    cell_size = jnp.array(cell_size, dtype=positions.dtype)
    cell_size = jnp.expand_dims(cell_size, (0, 1))

    floored_positions = jnp.floor(positions / cell_size).astype(jnp.int32)

    if ordering == "yx":
        desc_shape = jnp.array([[height, width]])
    elif ordering == "xy":
        desc_shape = jnp.array([[width, height]])

    mask = jnp.logical_and(floored_positions >= 0, floored_positions < desc_shape)
    mask = mask.all(axis=1)

    if ordering == "yx":
        floored_positions = (
            floored_positions[..., 0] * width + floored_positions[..., 1]
        )
    elif ordering == "xy":
        floored_positions = (
            floored_positions[..., 1] * width + floored_positions[..., 0]
        )
    floored_positions = jnp.where(mask, floored_positions, -1)

    return floored_positions


def asym_keep_mutual_correspondences_only(corr_0, corr_1):
    idx = jnp.arange(corr_0.size)
    is_bidir = corr_1[corr_0] == idx
    return jnp.where(is_bidir, corr_0, -1)


def keep_mutual_correspondences_only(corr_0, corr_1):
    corr_0 = asym_keep_mutual_correspondences_only(corr_0, corr_1)
    corr_1 = asym_keep_mutual_correspondences_only(corr_1, corr_0)
    return corr_0, corr_1


def _scan_reduce(x0, x1, reducer, block_size):
    x0_shape0 = x0.shape[0]
    n = x0.shape[0] // block_size

    if x0.shape[0] % block_size > 0:
        r = block_size - x0.shape[0] % block_size
        _0 = jnp.array(0, dtype=x0.dtype)
        x0 = jax.lax.pad(x0, _0, ((0, r, 0), (0, 0, 0)))
        n += 1

    x0 = jax.lax.reshape(x0, (n, block_size, x0.shape[1]))
    xs = x0

    def fun(_, x0):
        return None, reducer(x0, x1)

    _, accu = jax.lax.scan(fun, None, xs, length=n, unroll=1)

    def reshape(x):
        return jnp.ravel(x)[:x0_shape0]

    return jax.tree_map(reshape, accu)


def asym_corr_cross_entropy(
    lse,
    corr,
    desc_0,
    desc_1,
):

    # get mask of valid correspondences
    query_corr = corr >= 0
    n_corr = query_corr.sum()

    # make -1 correspondences out-of-bound (for the next get fille)
    corr = jnp.where(query_corr, corr, desc_1.shape[0])

    # align all descriptors from 1 to descriptors from 0
    # set unmatched descriptors to 0 (those that are out-of-bound)
    _desc_1 = desc_1.at[corr].get(
        mode="fill",
        fill_value=0,
    )

    # aligned dot product
    log_num = jax.vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(desc_0, _desc_1)

    # compute log of denominator
    log_den = lse

    log_p_corr = jnp.sum(log_num, where=query_corr) - jnp.sum(log_den, where=query_corr)

    log_p_corr /= n_corr

    return -log_p_corr


def sym_corr_cross_entropy(
    lse_0,
    lse_1,
    desc_0,
    desc_1,
    corr_0,
    corr_1,
):
    loss_0 = asym_corr_cross_entropy(
        lse_0,
        corr_0,
        desc_0,
        desc_1,
    )
    loss_1 = asym_corr_cross_entropy(
        lse_1,
        corr_1,
        desc_1,
        desc_0,
    )
    return loss_0 + loss_1


def corr_matching_binary_cross_entropy(
    best_idx_0,
    best_idx_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
):

    best_idx_0, best_idx_1 = keep_mutual_correspondences_only(best_idx_0, best_idx_1)

    # gt positives mask
    gt_mask_0 = corr_0 >= 0
    gt_mask_1 = corr_1 >= 0

    # pred positives mask
    pr_mask_0 = best_idx_0 >= 0
    pr_mask_1 = best_idx_1 >= 0

    # true positives
    tp_mask_0 = jnp.logical_and(gt_mask_0, pr_mask_0)
    tp_mask_1 = jnp.logical_and(gt_mask_1, pr_mask_1)

    # correct matches
    correct_mask_0 = corr_0 == best_idx_0
    correct_mask_1 = corr_1 == best_idx_1

    loss_0 = correct_mask_0 * jax.nn.softplus(-logits_0) + (
        ~correct_mask_0
    ) * jax.nn.softplus(+logits_0)
    loss_1 = correct_mask_1 * jax.nn.softplus(-logits_1) + (
        ~correct_mask_1
    ) * jax.nn.softplus(+logits_1)

    m0 = tp_mask_0
    m1 = tp_mask_1

    m0 = jnp.logical_or(m0, gt_mask_0)
    m1 = jnp.logical_or(m1, gt_mask_1)

    n0 = m0.sum()
    n1 = m1.sum()

    loss_0 = loss_0.sum(where=m0)
    loss_1 = loss_1.sum(where=m1)

    loss = (loss_0 + loss_1) / (n0 + n1)

    precision = tp_mask_0.sum() / pr_mask_0.sum()
    recall = tp_mask_0.sum() / gt_mask_0.sum()

    correct_mask_0 = correct_mask_0 * m0
    correct_mask_1 = correct_mask_1 * m1

    return loss, precision, recall, correct_mask_0, correct_mask_1


def total_loss(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    block_size,
):
    
    if block_size is None:  # reduction on full similarity matrix
        x0x1 = desc_0 @ desc_1.T

        lse_0 = jax.nn.logsumexp(x0x1, axis=1)
        lse_1 = jax.nn.logsumexp(x0x1, axis=0)
        argmax_0 = jnp.argmax(x0x1, axis=1)
        argmax_1 = jnp.argmax(x0x1, axis=0)
    else:  # reduction by scanning blocks of similarity matrix

        @delayed_vjp
        def reducer(x0, x1):
            x0x1 = x0 @ x1.T
            output = (
                jax.nn.logsumexp(x0x1, axis=1),
                jnp.argmax(x0x1, axis=1),
            )
            return output

        lse_0, argmax_0 = _scan_reduce(
            desc_0,
            desc_1,
            reducer,
            block_size,
        )
        lse_1, argmax_1 = _scan_reduce(
            desc_1,
            desc_0,
            reducer,
            block_size,
        )

    # info nce loss
    loss_0 = sym_corr_cross_entropy(
        lse_0,
        lse_1,
        desc_0,
        desc_1,
        corr_0,
        corr_1,
    )

    # matching loss
    loss_1, precision, recall, correct_mask_0, correct_mask_1 = corr_matching_binary_cross_entropy(
        argmax_0,
        argmax_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
    )
    
    return loss_0, loss_1, precision, recall, correct_mask_0, correct_mask_1