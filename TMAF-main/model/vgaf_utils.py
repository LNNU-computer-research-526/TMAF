# -*- coding: utf-8 -*-
"""VGAF (visual–audio graph attention fusion) helpers.

Eq.(11) cross-modal alignment: for each video b, frame i attends only over
audio frames of the same video (T×T), not over the flattened batch×time axis.
"""
import torch


def vgaf_cross_modal_refine(vis_feat_encode, audio_feat_encode):
    """Per-video VGAF (Eq. 11).

    Args:
        vis_feat_encode: (B, T, C) visual temporal features.
        audio_feat_encode: (B, T, C) audio temporal features.

    Returns:
        Refined audio features (B, T, C), residual added.

    Complexity (per forward):
        O(B · T² · C) for score matrix and weighted sum.
        Memory for attention weights: O(B · T²).

    Note:
        Flattening batch and time to (B·T, C) yields a (B·T)×(B·T) matrix
        (O(B²T²C)) and lets unrelated videos in the same mini-batch interact;
        this implementation avoids that leakage.
    """
    scale = vis_feat_encode.size(-1) ** 0.5
    attn_scores = torch.matmul(
        vis_feat_encode, audio_feat_encode.transpose(-2, -1)
    ) / scale
    attn_scores = torch.softmax(attn_scores, dim=-1)
    aligned = torch.matmul(attn_scores, audio_feat_encode)
    return aligned + audio_feat_encode


