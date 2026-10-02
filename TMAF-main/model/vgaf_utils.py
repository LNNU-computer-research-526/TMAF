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


# Text block for manuscript / rebuttal (复杂度分析，式(11)修正后).
VGAF_COMPLEXITY_ANALYSIS_ZH = """
VGAF 跨模态对齐（式(11)）在实现中对每个视频独立计算注意力。设 batch 大小为 B，
每个视频片段长度为 T，特征维度为 C。视觉与音频编码特征记为 F_v, F_a ∈ R^{B×T×C}。
对第 b 个视频，注意力权重为
  A^{(b)} = softmax( F_v^{(b)} (F_a^{(b)})^T / sqrt(C) ) ∈ R^{T×T}，
精炼后的音频特征为 F̃_a^{(b)} = A^{(b)} F_a^{(b)} + F_a^{(b)}。
时间复杂度为 O(B·T²·C)，额外显存 O(B·T²)。该形式与 mini-batch 中样本顺序无关，
且同一 batch 内不同视频之间无注意力耦合。

（修正说明）若将 batch 维与时间维展平为 (B·T, C) 再计算 (B·T)×(B·T) 的全局注意力，
复杂度为 O(B²·T²·C)，且会引入跨视频交互，与「仅在单视频片段内对齐帧级视听特征」
的设计不符；正文与代码已统一为按视频独立的 T×T 注意力。
""".strip()
