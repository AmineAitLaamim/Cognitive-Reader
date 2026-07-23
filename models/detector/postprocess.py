"""
models/detector/postprocess.py
Heatmap decoding, NMS, sub-pixel refinement, and box estimation.

Contains everything related to CONSUMING the heatmap:
  1. NMS — 3x3 max-pool local maxima suppression.
  2. Sub-pixel refinement — 2D parabolic fitting for <1px accuracy.
  3. Box estimation — fixed-size boxes from detected centers.
  4. Coordinate utilities — format conversions, clamping, normalization.
  5. DigitDetector — full pipeline assembling heatmap → detections → boxes.

Does NOT contain the heatmap neural network or training logic.
Those live in heatmap.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

@dataclass
class PostProcessConfig:
    """Configuration for heatmap post-processing."""
    # Detection
    stride: int = 8
    score_threshold: float = 0.3
    top_k: int = 200
    nms_kernel_size: int = 3
    
    # Sub-pixel refinement
    enable_subpixel: bool = True
    max_offset: float = 0.5
    
    # Box estimation
    estimated_digit_w: float = 20.0
    estimated_digit_h: float = 30.0
    box_padding_factor: float = 1.3
    
    # Image
    img_width: int = 640
    img_height: int = 640


@dataclass
class DetectionResult:
    """Result of digit detection on a single image."""
    centers_px: torch.Tensor       # [N, 2] — (x, y) in pixels
    boxes: torch.Tensor            # [N, 4] — (x1, y1, x2, y2)
    scores: torch.Tensor           # [N] — confidence scores
    heatmap: torch.Tensor          # [H/stride, W/stride] — predicted heatmap
    num_detections: int


# ==============================================================
# 2. COORDINATE UTILITIES
# ==============================================================

def centers_to_boxes(
    centers: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor
) -> torch.Tensor:
    """
    Convert centers + sizes to (x1, y1, x2, y2) boxes.
    
    Args:
        centers: [N, 2] — (cx, cy).
        widths: [N] or float — box widths.
        heights: [N] or float — box heights.
    Returns:
        [N, 4] — (x1, y1, x2, y2).
    """
    if isinstance(widths, (int, float)):
        widths = torch.full((centers.shape[0],), widths, device=centers.device, dtype=centers.dtype)
    if isinstance(heights, (int, float)):
        heights = torch.full((centers.shape[0],), heights, device=centers.device, dtype=centers.dtype)
    
    return torch.stack([
        centers[:, 0] - widths / 2,
        centers[:, 1] - heights / 2,
        centers[:, 0] + widths / 2,
        centers[:, 1] + heights / 2,
    ], dim=1)


def boxes_to_centers(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert (x1, y1, x2, y2) to centers + sizes.
    
    Returns:
        centers: [N, 2], sizes: [N, 2] (w, h).
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy], dim=1), torch.stack([w, h], dim=1)


def clamp_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Clamp boxes to image boundaries."""
    b = boxes.clone()
    b[:, 0].clamp_(min=0, max=img_w)
    b[:, 1].clamp_(min=0, max=img_h)
    b[:, 2].clamp_(min=0, max=img_w)
    b[:, 3].clamp_(min=0, max=img_h)
    return b


def normalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Normalize box coords to [0, 1]."""
    b = boxes.clone().float()
    b[:, 0] /= img_w
    b[:, 1] /= img_h
    b[:, 2] /= img_w
    b[:, 3] /= img_h
    return b


def denormalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Denormalize box coords from [0, 1] to pixels."""
    b = boxes.clone().float()
    b[:, 0] *= img_w
    b[:, 1] *= img_h
    b[:, 2] *= img_w
    b[:, 3] *= img_h
    return b


# ==============================================================
# 3. NMS (Non-Maximum Suppression)
# ==============================================================

def heatmap_nms(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    3x3 max-pool NMS: keep only local maxima.
    
    Every pixel that is NOT the maximum in its 3x3 neighborhood
    is suppressed to 0. This ensures each digit produces exactly
    one detection peak.
    
    Args:
        heatmap: [H, W] — heatmap after sigmoid.
        kernel_size: NMS window size (3 = standard).
    
    Returns:
        [H, W] — heatmap with non-maxima set to 0.
    """
    hm_4d = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pad = kernel_size // 2
    hm_max = F.max_pool2d(hm_4d, kernel_size=kernel_size, stride=1, padding=pad)
    hm_max = hm_max.squeeze(0).squeeze(0)
    
    keep = (heatmap == hm_max).float()
    return heatmap * keep


# ==============================================================
# 4. SUB-PIXEL REFINEMENT
# ==============================================================

def subpixel_refine(
    heatmap: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    max_offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Refine integer peak locations to sub-pixel accuracy via
    2D parabolic fitting.
    
    For each peak at integer (x, y), fit a parabola to the
    horizontal neighbors (x-1, x, x+1) and vertical neighbors
    (y-1, y, y+1) independently:
    
        offset = (f(x+1) - f(x-1)) / (2 * (2*f(x) - f(x-1) - f(x+1)))
    
    This gives the true peak location of the underlying continuous
    function, reducing quantization error from ±4px (stride 8)
    to <1px.
    
    Args:
        heatmap: [H, W] — original heatmap (before NMS).
        xs: [N] — integer x coordinates of peaks.
        ys: [N] — integer y coordinates of peaks.
        max_offset: Clamp offsets to [-max_offset, max_offset].
    
    Returns:
        xs_refined: [N] — refined x coordinates.
        ys_refined: [N] — refined y coordinates.
    """
    H, W = heatmap.shape
    xs_ref = xs.clone()
    ys_ref = ys.clone()
    
    for i in range(len(xs)):
        xi = int(xs[i].item())
        yi = int(ys[i].item())
        
        # Horizontal fit
        if 0 < xi < W - 1:
            left = heatmap[yi, xi - 1].item()
            center = heatmap[yi, xi].item()
            right = heatmap[yi, xi + 1].item()
            denom = 2.0 * (2.0 * center - left - right)
            if abs(denom) > 1e-6:
                dx = (right - left) / denom
                xs_ref[i] = xs[i] + max(-max_offset, min(max_offset, dx))
        
        # Vertical fit
        if 0 < yi < H - 1:
            up = heatmap[yi - 1, xi].item()
            center = heatmap[yi, xi].item()
            down = heatmap[yi + 1, xi].item()
            denom = 2.0 * (2.0 * center - up - down)
            if abs(denom) > 1e-6:
                dy = (down - up) / denom
                ys_ref[i] = ys[i] + max(-max_offset, min(max_offset, dy))
    
    return xs_ref, ys_ref


# ==============================================================
# 5. HEATMAP DECODER
# ==============================================================

class HeatmapDecoder:
    """
    Decodes a predicted heatmap into digit center detections.
    
    Full pipeline:
      1. Sigmoid activation (if logits provided).
      2. 3x3 NMS (suppress non-maxima).
      3. Top-K selection.
      4. Confidence thresholding.
      5. Sub-pixel refinement.
      6. Feature-map → pixel coordinate conversion.
    """
    
    def __init__(self, config: PostProcessConfig):
        self.cfg = config
    
    @torch.no_grad()
    def decode(
        self,
        heatmap: torch.Tensor,
        is_logits: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a heatmap into detections.
        
        Args:
            heatmap: [B, 1, H, W] or [B, H, W] — predicted heatmap.
            is_logits: If True, apply sigmoid first.
        
        Returns:
            all_centers: [N_total, 2] — pixel coordinates (x, y).
            all_scores: [N_total] — confidence scores.
        """
        if heatmap.dim() == 4:
            heatmap = heatmap.squeeze(1)  # [B, H, W]
        
        if is_logits:
            heatmap = torch.sigmoid(heatmap)
        
        B, H, W = heatmap.shape
        all_centers = []
        all_scores = []
        
        for b in range(B):
            centers, scores = self._decode_single(heatmap[b])
            all_centers.append(centers)
            all_scores.append(scores)
        
        if not all_centers:
            device = heatmap.device
            return torch.zeros(0, 2, device=device), torch.zeros(0, device=device)
        
        return torch.cat(all_centers, dim=0), torch.cat(all_scores, dim=0)
    
    def _decode_single(
        self, hm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode one heatmap [H, W]."""
        H, W = hm.shape
        device = hm.device
        
        # NMS
        hm_nms = heatmap_nms(hm, self.cfg.nms_kernel_size)
        
        # Top-K
        scores_flat, indices_flat = torch.topk(
            hm_nms.view(-1), min(self.cfg.top_k, H * W)
        )
        
        # Threshold
        mask = scores_flat > self.cfg.score_threshold
        scores_flat = scores_flat[mask]
        indices_flat = indices_flat[mask]
        
        if len(scores_flat) == 0:
            return torch.zeros(0, 2, device=device), torch.zeros(0, device=device)
        
        # Flat index → (y, x)
        ys = (indices_flat // W).float()
        xs = (indices_flat % W).float()
        
        # Sub-pixel refinement
        if self.cfg.enable_subpixel:
            xs, ys = subpixel_refine(hm, xs, ys, self.cfg.max_offset)
        
        # Feature-map → pixel coordinates
        centers = torch.stack([
            xs * self.cfg.stride,
            ys * self.cfg.stride
        ], dim=1)
        
        return centers, scores_flat


# ==============================================================
# 6. BOX ESTIMATOR
# ==============================================================

class BoxEstimator:
    """
    Estimates bounding boxes from detected center points.
    
    Uses fixed-size boxes centered on each detection.
    The box size is configurable and expanded by box_padding_factor
    to include local context for RoI Align.
    
    For real-world data with variable digit sizes, replace this
    with a learned regression head.
    """
    
    def __init__(self, config: PostProcessConfig):
        self.cfg = config
    
    def estimate(
        self,
        centers: torch.Tensor,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None
    ) -> torch.Tensor:
        """
        Estimate boxes from centers.
        
        Args:
            centers: [N, 2] — (x, y) in pixels.
            img_width: For clamping.
            img_height: For clamping.
        
        Returns:
            [N, 4] — (x1, y1, x2, y2).
        """
        if centers.shape[0] == 0:
            return torch.zeros(0, 4, device=centers.device)
        
        w = img_width or self.cfg.img_width
        h = img_height or self.cfg.img_height
        
        half_w = self.cfg.estimated_digit_w * self.cfg.box_padding_factor / 2.0
        half_h = self.cfg.estimated_digit_h * self.cfg.box_padding_factor / 2.0
        
        boxes = torch.stack([
            centers[:, 0] - half_w,
            centers[:, 1] - half_h,
            centers[:, 0] + half_w,
            centers[:, 1] + half_h,
        ], dim=1)
        
        return clamp_boxes(boxes, w, h)


# ==============================================================
# 7. FULL DETECTION PIPELINE
# ==============================================================

class DigitDetector:
    """
    Complete digit detection pipeline.
    
    Assembles: heatmap head (from heatmap.py) + decoder + box estimator.
    
    Usage:
        from models.detector.heatmap import HeatmapHead
        from models.detector.postprocess import DigitDetector, PostProcessConfig
        
        head = HeatmapHead(512, 128)
        detector = DigitDetector(PostProcessConfig(), heatmap_head=head)
        
        result = detector.detect_from_logits(heatmap_logits)
        # result.centers_px: [N, 2]
        # result.boxes: [N, 4]
        # result.scores: [N]
    """
    
    def __init__(
        self,
        config: PostProcessConfig,
        heatmap_head: Optional[nn.Module] = None
    ):
        self.cfg = config
        self.heatmap_head = heatmap_head
        self.decoder = HeatmapDecoder(config)
        self.box_estimator = BoxEstimator(config)
    
    @torch.no_grad()
    def detect_from_logits(
        self,
        heatmap_logits: torch.Tensor
    ) -> DetectionResult:
        """
        Run post-processing on raw heatmap logits.
        
        Args:
            heatmap_logits: [B, 1, H, W] — raw logits from HeatmapHead.
        
        Returns:
            DetectionResult.
        """
        # Decode
        centers, scores = self.decoder.decode(heatmap_logits, is_logits=True)
        
        # Estimate boxes
        boxes = self.box_estimator.estimate(
            centers, self.cfg.img_width, self.cfg.img_height
        )
        
        # Heatmap for visualization
        heatmap = torch.sigmoid(heatmap_logits)
        if heatmap.dim() == 4:
            heatmap = heatmap.squeeze(0).squeeze(0)
        
        return DetectionResult(
            centers_px=centers,
            boxes=boxes,
            scores=scores,
            heatmap=heatmap,
            num_detections=centers.shape[0]
        )
    
    @torch.no_grad()
    def detect_from_feature_map(
        self,
        feature_map: torch.Tensor
    ) -> DetectionResult:
        """
        Run full pipeline: feature map → heatmap head → decode → boxes.
        
        Requires heatmap_head to be set.
        
        Args:
            feature_map: [B, C, H/8, W/8]
        
        Returns:
            DetectionResult.
        """
        if self.heatmap_head is None:
            raise ValueError("heatmap_head not set. Provide it in constructor.")
        
        logits = self.heatmap_head(feature_map)
        return self.detect_from_logits(logits)
    
    @torch.no_grad()
    def detect_from_backbone(
        self,
        backbone: nn.Module,
        image: torch.Tensor
    ) -> DetectionResult:
        """
        Run full pipeline: image → backbone → heatmap → decode → boxes.
        
        Args:
            backbone: Stride-8 backbone (returns feature map or dict).
            image: [1, 3, H, W] — preprocessed image.
        
        Returns:
            DetectionResult.
        """
        fm = backbone(image)
        if isinstance(fm, dict):
            fm = fm['feature_map']
        return self.detect_from_feature_map(fm)


if __name__ == "__main__":
    print("=" * 60)
    print("  models/detector/postprocess.py — Unit Test")
    print("=" * 60)
    
    config = PostProcessConfig(score_threshold=0.1, top_k=50)
    
    # Test 1: Coordinate utilities
    print("\n[Test 1] Coordinate utilities")
    centers = torch.tensor([[100.0, 100.0], [200.0, 150.0]])
    boxes = centers_to_boxes(centers, 20.0, 30.0)
    assert boxes.shape == (2, 4)
    assert boxes[0, 0] == 90.0 and boxes[0, 2] == 110.0
    
    rec_centers, rec_sizes = boxes_to_centers(boxes)
    assert torch.allclose(rec_centers, centers)
    
    oob = clamp_boxes(torch.tensor([[-10.0, -5.0, 700.0, 700.0]]), 640, 640)
    assert oob[0, 0] == 0.0 and oob[0, 2] == 640.0
    
    nb = normalize_boxes(torch.tensor([[320.0, 320.0, 340.0, 340.0]]), 640, 640)
    db = denormalize_boxes(nb, 640, 640)
    assert torch.allclose(db, torch.tensor([[320.0, 320.0, 340.0, 340.0]]), atol=1e-4)
    print("  ✓ All coordinate utilities passed")
    
    # Test 2: NMS
    print("\n[Test 2] NMS")
    hm = torch.zeros(20, 20)
    hm[5, 5] = 0.9
    hm[5, 6] = 0.7  # Neighbor — should be suppressed
    hm[15, 15] = 0.8
    
    hm_nms = heatmap_nms(hm, kernel_size=3)
    assert hm_nms[5, 5] == 0.9   # Local max — kept
    assert hm_nms[5, 6] == 0.0   # Not local max — suppressed
    assert hm_nms[15, 15] == 0.8  # Local max — kept
    print("  ✓ NMS correctly suppresses non-maxima")
    
    # Test 3: Sub-pixel refinement
    print("\n[Test 3] Sub-pixel refinement")
    hm_sub = torch.zeros(20, 20)
    hm_sub[10, 10] = 1.0
    hm_sub[10, 11] = 0.6  # Asymmetric → peak shifts right
    
    xs, ys = torch.tensor([10.0]), torch.tensor([10.0])
    xs_r, ys_r = subpixel_refine(hm_sub, xs, ys)
    print(f"  Original: ({xs[0]:.2f}, {ys[0]:.2f})")
    print(f"  Refined:  ({xs_r[0]:.2f}, {ys_r[0]:.2f})")
    assert xs_r[0] > 10.0, "Should shift right"
    print("  ✓ Sub-pixel refinement works")
    
    # Test 4: HeatmapDecoder
    print("\n[Test 4] HeatmapDecoder")
    decoder = HeatmapDecoder(config)
    
    fake_logits = torch.full((1, 1, 80, 80), -5.0)
    fake_logits[0, 0, 12, 12] = 5.0
    fake_logits[0, 0, 40, 50] = 4.0
    fake_logits[0, 0, 65, 70] = 3.0
    
    centers, scores = decoder.decode(fake_logits, is_logits=True)
    print(f"  Detections: {centers.shape[0]}")
    for i in range(centers.shape[0]):
        print(f"    ({centers[i,0]:.1f}, {centers[i,1]:.1f}) score={scores[i]:.4f}")
    assert centers.shape[0] == 3
    print("  ✓ Decoder found all 3 peaks")
    
    # Test 5: BoxEstimator
    print("\n[Test 5] BoxEstimator")
    estimator = BoxEstimator(config)
    test_centers = torch.tensor([[100.0, 100.0], [5.0, 5.0]])
    boxes = estimator.estimate(test_centers)
    assert boxes.shape == (2, 4)
    assert boxes[1, 0] >= 0  # Clamped
    w = boxes[0, 2] - boxes[0, 0]
    h = boxes[0, 3] - boxes[0, 1]
    print(f"  Box size: {w:.1f}x{h:.1f}")
    print("  ✓ Box estimation passed")
    
    # Test 6: DigitDetector
    print("\n[Test 6] DigitDetector (logits input)")
    detector = DigitDetector(config)
    result = detector.detect_from_logits(fake_logits)
    print(f"  Detections: {result.num_detections}")
    print(f"  Centers: {result.centers_px.shape}")
    print(f"  Boxes: {result.boxes.shape}")
    print(f"  Heatmap: {result.heatmap.shape}")
    assert result.num_detections == 3
    print("  ✓ Full pipeline passed")
    
    # Test 7: Empty detection
    print("\n[Test 7] Empty detection")
    strict_config = PostProcessConfig(score_threshold=0.99)
    strict_detector = DigitDetector(strict_config)
    low_logits = torch.full((1, 1, 80, 80), -10.0)
    empty_result = strict_detector.detect_from_logits(low_logits)
    assert empty_result.num_detections == 0
    assert empty_result.centers_px.shape == (0, 2)
    assert empty_result.boxes.shape == (0, 4)
    print("  ✓ Empty detection handled")
    
    # Test 8: DigitDetector with heatmap head
    print("\n[Test 8] DigitDetector with HeatmapHead")
    from models.detector.heatmap import HeatmapHead
    head = HeatmapHead(512, 128)
    detector_with_head = DigitDetector(config, heatmap_head=head)
    
    fm = torch.randn(1, 512, 80, 80)
    result_fm = detector_with_head.detect_from_feature_map(fm)
    print(f"  Detections from feature map: {result_fm.num_detections}")
    print("  ✓ Feature map pipeline passed")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)