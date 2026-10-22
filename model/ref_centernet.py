# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026: trimmed to inference (forward and decode), no mmcv/mmdet dependency.
"""Vendored subset of mmdetection 2.x CenterNetHead (Apache-2.0, OpenMMLab).

Only what CYWS inference uses: the three conv branches (forward_single) and the
heatmap decode (get_local_maximum / get_topk_from_heatmap / transpose_and_gather_feat /
decode_heatmap), following mmdet v2.28.2 mmdet/models/dense_heads/centernet_head.py and
mmdet/models/utils/gaussian_target.py. Used as the numeric reference so the
reference check does not need mmcv/mmdet installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefCenterNetHead(nn.Module):
    def __init__(self, in_channel: int = 64, feat_channel: int = 64, num_classes: int = 1):
        super().__init__()
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

    @staticmethod
    def _build_head(in_channel: int, feat_channel: int, out_channel: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
                             nn.ReLU(inplace=True), nn.Conv2d(feat_channel, out_channel, kernel_size=1))

    def forward(self, feat: torch.Tensor):
        # forward_single
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred


def get_local_maximum(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores: torch.Tensor, k: int = 20):
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    return feat.gather(1, ind)


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return gather_feat(feat, ind)


def decode_heatmap(center_heatmap_pred, wh_pred, offset_pred, k: int = 100, kernel: int = 3,
                   img_shape: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """mmdet CenterNetHead.decode_heatmap; returns (B, k, 5) [x1, y1, x2, y2, score]."""
    height, width = center_heatmap_pred.shape[2:]
    inp_h, inp_w = img_shape
    center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)
    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets
    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
    batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
    return torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
