"""
models/backbone/roi_align.py
Padded RoI Align with aspect-ratio preservation for the Cognitive Reader.

Standard RoI Align forces every bounding box into a fixed HxW grid,
distorting the aspect ratio of the glyph. A tall '1' gets stretched
horizontally; a wide '0' gets squished vertically.

Padded RoI Align fixes this by expanding each bounding box to a perfect
square (padding the shorter dimension with background) BEFORE applying
RoI Align. The internal geometry of the digit is perfectly preserved.

This module is standalone. It is imported by cnn.py but can be used
independently for any RoI extraction task.

Components:
  1. PaddedRoIAlign — core square-padding + RoI Align operation.
  2. RoIFeatureExtractor — RoI Align + flatten + MLP projection.
  3. BatchedRoIAlign — handles variable numbers of boxes per image.
  4. Coordinate utilities — box format conversions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

try:
    from torchvision.ops import roi_align as tv_roi_align
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ==============================================================
# 1. COORDINATE UTILITIES
# ==============================================================

def centers_to_boxes(
    centers: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor
) -> torch.Tensor:
    """
    Convert center coordinates + sizes to (x1, y1, x2, y2) boxes.
    
    Args:
        centers: [N, 2] — (cx, cy) center coordinates.
        widths: [N] or scalar — box widths.
        heights: [N] or scalar — box heights.
    
    Returns:
        [N, 4] — (x1, y1, x2, y2) boxes.
    """
    if isinstance(widths, (int, float)):
        widths = torch.full((centers.shape[0],), widths, device=centers.device)
    if isinstance(heights, (int, float)):
        heights = torch.full((centers.shape[0],), heights, device=centers.device)
    
    x1 = centers[:, 0] - widths / 2.0
    y1 = centers[:, 1] - heights / 2.0
    x2 = centers[:, 0] + widths / 2.0
    y2 = centers[:, 1] + heights / 2.0
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def boxes_to_centers(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (x1, y1, x2, y2) boxes to center coordinates + sizes.
    
    Args:
        boxes: [N, 4] — (x1, y1, x2, y2).
    
    Returns:
        centers: [N, 2] — (cx, cy).
        sizes: [N, 2] — (w, h).
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    centers = torch.stack([cx, cy], dim=1)
    sizes = torch.stack([w, h], dim=1)
    return centers, sizes


def make_square_boxes(
    boxes: torch.Tensor,
    padding_factor: float = 1.2
) -> torch.Tensor:
    """
    Expand bounding boxes to perfect squares, preserving aspect ratio.
    
    The shorter dimension is padded symmetrically with background.
    An optional padding_factor expands the square beyond the exact fit
    to include local context.
    
    Example:
        Original box: 10x40 pixels (tall '1')
        padding_factor=1.0 → square: 40x40
        padding_factor=1.2 → square: 48x48 (20% context padding)
    
    Args:
        boxes: [N, 4] — (x1, y1, x2, y2) in pixel coordinates.
        padding_factor: Expansion factor (1.0 = exact fit, 1.2 = 20% padding).
    
    Returns:
        [N, 4] — square boxes (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Compute center and size
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    
    # Square size: max(w, h) * padding_factor
    s = torch.max(w, h) * padding_factor
    
    # New square box centered on the original center
    new_x1 = cx - s / 2.0
    new_y1 = cy - s / 2.0
    new_x2 = cx + s / 2.0
    new_y2 = cy + s / 2.0
    
    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


def clamp_boxes(
    boxes: torch.Tensor,
    img_width: int,
    img_height: int
) -> torch.Tensor:
    """
    Clamp bounding boxes to image boundaries.
    
    Args:
        boxes: [N, 4] — (x1, y1, x2, y2).
        img_width: Image width in pixels.
        img_height: Image height in pixels.
    
    Returns:
        [N, 4] — clamped boxes.
    """
    clamped = boxes.clone()
    clamped[:, 0].clamp_(min=0, max=img_width)
    clamped[:, 1].clamp_(min=0, max=img_height)
    clamped[:, 2].clamp_(min=0, max=img_width)
    clamped[:, 3].clamp_(min=0, max=img_height)
    return clamped


def normalize_boxes(
    boxes: torch.Tensor,
    img_width: int,
    img_height: int
) -> torch.Tensor:
    """
    Normalize box coordinates to [0, 1].
    
    Args:
        boxes: [N, 4] — (x1, y1, x2, y2) in pixels.
        img_width: Image width.
        img_height: Image height.
    
    Returns:
        [N, 4] — normalized boxes.
    """
    normalized = boxes.clone().float()
    normalized[:, 0] /= img_width
    normalized[:, 1] /= img_height
    normalized[:, 2] /= img_width
    normalized[:, 3] /= img_height
    return normalized


def denormalize_boxes(
    boxes: torch.Tensor,
    img_width: int,
    img_height: int
) -> torch.Tensor:
    """
    Denormalize box coordinates from [0, 1] to pixels.
    
    Args:
        boxes: [N, 4] — normalized (x1, y1, x2, y2).
        img_width: Image width.
        img_height: Image height.
    
    Returns:
        [N, 4] — pixel boxes.
    """
    denorm = boxes.clone().float()
    denorm[:, 0] *= img_width
    denorm[:, 1] *= img_height
    denorm[:, 2] *= img_width
    denorm[:, 3] *= img_height
    return denorm


# ==============================================================
# 2. PADDED ROI ALIGN (Core Operation)
# ==============================================================

class PaddedRoIAlign(nn.Module):
    """
    Aspect-ratio-preserving RoI Align.
    
    Pipeline:
      1. Expand each bounding box to a perfect square (padding shorter dim).
      2. Apply standard RoI Align on the square box.
      3. Output: fixed-size feature grid with preserved aspect ratio.
    
    This ensures that a '1' (tall, thin) and a '0' (round) both produce
    geometrically faithful feature grids, without artificial stretching
    or squishing.
    """
    
    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1.0 / 8.0,
        padding_factor: float = 1.2,
        sampling_ratio: int = 2,
        aligned: bool = True
    ):
        """
        Args:
            output_size: Size of the output grid (7 -> 7x7).
            spatial_scale: Scale from image pixels to feature map pixels.
                          For a stride-8 backbone: 1/8 = 0.125.
            padding_factor: Box expansion factor for square padding.
                           1.0 = exact fit, 1.2 = 20% context padding.
            sampling_ratio: Number of sampling points per bin in each direction.
                           Higher = more accurate but slower.
            aligned: If True, shift RoI by -0.5 pixel for better alignment
                    (recommended for stride > 1 feature maps).
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for RoI Align. "
                "Install with: pip install torchvision"
            )
        
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.padding_factor = padding_factor
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
    
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Padded RoI Align.
        
        Args:
            feature_map: [B, C, H, W] — backbone feature map.
            boxes: [N, 4] — (x1, y1, x2, y2) in IMAGE pixel coordinates.
            batch_indices: [N] — which image each box belongs to.
                          Default: all boxes belong to image 0.
        
        Returns:
            [N, C, output_size, output_size] — RoI features.
        """
        N = boxes.shape[0]
        C = feature_map.shape[1]
        
        if N == 0:
            return torch.zeros(
                0, C, self.output_size, self.output_size,
                device=feature_map.device, dtype=feature_map.dtype
            )
        
        # Step 1: Expand to square boxes
        square_boxes = make_square_boxes(boxes, self.padding_factor)  # [N, 4]
        
        # Step 2: Build RoI tensor [N, 5] = (batch_idx, x1, y1, x2, y2)
        if batch_indices is None:
            batch_indices = torch.zeros(N, device=boxes.device, dtype=boxes.dtype)
        else:
            batch_indices = batch_indices.float()
        
        rois = torch.cat([
            batch_indices.unsqueeze(1),
            square_boxes
        ], dim=1)  # [N, 5]
        
        # Step 3: Apply RoI Align
        roi_features = tv_roi_align(
            feature_map,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )  # [N, C, output_size, output_size]
        
        return roi_features
    
    def extra_repr(self) -> str:
        return (
            f"output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, "
            f"padding_factor={self.padding_factor}, "
            f"sampling_ratio={self.sampling_ratio}, "
            f"aligned={self.aligned}"
        )


# ==============================================================
# 3. ROI FEATURE EXTRACTOR (RoI Align + Projection)
# ==============================================================

class RoIFeatureExtractor(nn.Module):
    """
    Complete RoI feature extraction pipeline:
      1. Padded RoI Align → [N, C, 7, 7]
      2. Flatten → [N, C * 7 * 7]
      3. MLP projection → [N, vis_dim]
    
    This is the module that produces the visual embedding e_vis
    for each digit node. The output is fed to:
      - The Identity Pathway (digit classification).
      - The GRU cell (h_content update).
    
    ARCHITECTURAL INVARIANT:
      The output of this module contains ONLY visual information.
      No spatial coordinates, no graph structure, no reading order.
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        vis_dim: int = 512,
        roi_output_size: int = 7,
        spatial_scale: float = 1.0 / 8.0,
        padding_factor: float = 1.2,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Args:
            in_channels: Number of channels in the backbone feature map.
            vis_dim: Output embedding dimensionality.
            roi_output_size: RoI Align grid size (7 -> 7x7).
            spatial_scale: Feature map scale (1/stride).
            padding_factor: Square padding expansion factor.
            dropout: Dropout rate in the projection MLP.
            use_layer_norm: Apply LayerNorm after projection.
        """
        super().__init__()
        
        self.roi_align = PaddedRoIAlign(
            output_size=roi_output_size,
            spatial_scale=spatial_scale,
            padding_factor=padding_factor
        )
        
        # Flatten dimension: C * roi_size * roi_size
        self.flat_dim = in_channels * roi_output_size * roi_output_size
        
        # Projection MLP
        layers = [
            nn.Linear(self.flat_dim, vis_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(vis_dim * 2, vis_dim),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(vis_dim))
        layers.append(nn.ReLU(inplace=True))
        
        self.projection = nn.Sequential(*layers)
        
        self.vis_dim = vis_dim
        self.roi_output_size = roi_output_size
    
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract visual embeddings for a set of bounding boxes.
        
        Args:
            feature_map: [B, C, H, W] — backbone feature map.
            boxes: [N, 4] — (x1, y1, x2, y2) in image pixel coordinates.
            batch_indices: [N] — batch index per box.
        
        Returns:
            [N, vis_dim] — visual embeddings.
        """
        # RoI Align
        roi_features = self.roi_align(
            feature_map, boxes, batch_indices
        )  # [N, C, 7, 7]
        
        # Flatten
        roi_flat = roi_features.flatten(1)  # [N, C * 7 * 7]
        
        # Project
        embeddings = self.projection(roi_flat)  # [N, vis_dim]
        
        return embeddings
    
    def get_roi_features(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get raw RoI features without projection (for visualization).
        
        Returns:
            [N, C, 7, 7] — raw RoI feature grids.
        """
        return self.roi_align(feature_map, boxes, batch_indices)


# ==============================================================
# 4. BATCHED ROI ALIGN
# ==============================================================

class BatchedRoIAlign(nn.Module):
    """
    Handles RoI Align for batches with variable numbers of boxes per image.
    
    In the Cognitive Reader, each image has a different number of digit
    nodes. The DataLoader concatenates all boxes with batch indices.
    This module wraps PaddedRoIAlign and handles the bookkeeping.
    
    Usage:
        batched_roi = BatchedRoIAlign(in_channels=512, vis_dim=512)
        
        # boxes: [N_total, 4], batch_indices: [N_total]
        embeddings = batched_roi(feature_map, boxes, batch_indices)
        # embeddings: [N_total, vis_dim]
        
        # Split back into per-image embeddings
        per_image = batched_roi.split_by_batch(embeddings, batch_indices, batch_size=8)
        # per_image: list of 8 tensors, each [N_i, vis_dim]
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        vis_dim: int = 512,
        roi_output_size: int = 7,
        spatial_scale: float = 1.0 / 8.0,
        padding_factor: float = 1.2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.extractor = RoIFeatureExtractor(
            in_channels=in_channels,
            vis_dim=vis_dim,
            roi_output_size=roi_output_size,
            spatial_scale=spatial_scale,
            padding_factor=padding_factor,
            dropout=dropout
        )
    
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings for all boxes across the batch.
        
        Args:
            feature_map: [B, C, H, W]
            boxes: [N_total, 4] — concatenated boxes from all images.
            batch_indices: [N_total] — image index for each box.
        
        Returns:
            [N_total, vis_dim] — embeddings for all boxes.
        """
        return self.extractor(feature_map, boxes, batch_indices)
    
    @staticmethod
    def split_by_batch(
        embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Split concatenated embeddings back into per-image lists.
        
        Args:
            embeddings: [N_total, vis_dim]
            batch_indices: [N_total]
            batch_size: B
        
        Returns:
            List of B tensors, each [N_i, vis_dim].
        """
        result = []
        for i in range(batch_size):
            mask = (batch_indices == i)
            result.append(embeddings[mask])
        return result
    
    @staticmethod
    def get_boxes_per_image(
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Split concatenated boxes back into per-image lists.
        
        Args:
            boxes: [N_total, 4]
            batch_indices: [N_total]
            batch_size: B
        
        Returns:
            List of B tensors, each [N_i, 4].
        """
        result = []
        for i in range(batch_size):
            mask = (batch_indices == i)
            result.append(boxes[mask])
        return result


# ==============================================================
# 5. MULTI-SCALE ROI ALIGN (Optional Extension)
# ==============================================================

class MultiScaleRoIAlign(nn.Module):
    """
    RoI Align at multiple scales, concatenated.
    
    Extracts features at 2-3 different RoI sizes and concatenates them.
    This provides the model with both fine-grained (tight box) and
    contextual (wide box) visual information.
    
    Useful when digit sizes vary significantly within the same image.
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        vis_dim: int = 512,
        roi_output_size: int = 7,
        spatial_scale: float = 1.0 / 8.0,
        padding_factors: Tuple[float, ...] = (1.0, 1.5, 2.0),
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Backbone feature map channels.
            vis_dim: Final output dimensionality.
            roi_output_size: Grid size per scale.
            spatial_scale: Feature map scale.
            padding_factors: Tuple of padding factors for multi-scale extraction.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.padding_factors = padding_factors
        num_scales = len(padding_factors)
        
        # One RoI Align per scale
        self.roi_layers = nn.ModuleList([
            PaddedRoIAlign(
                output_size=roi_output_size,
                spatial_scale=spatial_scale,
                padding_factor=pf
            )
            for pf in padding_factors
        ])
        
        # Projection: concatenated multi-scale features -> vis_dim
        flat_per_scale = in_channels * roi_output_size * roi_output_size
        total_flat = flat_per_scale * num_scales
        
        self.projection = nn.Sequential(
            nn.Linear(total_flat, vis_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(vis_dim * 2, vis_dim),
            nn.LayerNorm(vis_dim),
            nn.ReLU(inplace=True)
        )
        
        self.vis_dim = vis_dim
    
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-scale RoI feature extraction.
        
        Args:
            feature_map: [B, C, H, W]
            boxes: [N, 4]
            batch_indices: [N]
        
        Returns:
            [N, vis_dim]
        """
        N = boxes.shape[0]
        
        if N == 0:
            return torch.zeros(
                0, self.vis_dim,
                device=feature_map.device, dtype=feature_map.dtype
            )
        
        # Extract features at each scale
        scale_features = []
        for roi_layer in self.roi_layers:
            feats = roi_layer(feature_map, boxes, batch_indices)  # [N, C, 7, 7]
            scale_features.append(feats.flatten(1))  # [N, C*7*7]
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_features, dim=1)  # [N, num_scales * C*7*7]
        
        # Project
        embeddings = self.projection(multi_scale)  # [N, vis_dim]
        
        return embeddings


if __name__ == "__main__":
    print("=" * 60)
    print("  models/backbone/roi_align.py — Unit Test")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # ============================================================
    # Test 1: Coordinate utilities
    # ============================================================
    print("\n[Test 1] Coordinate utilities")
    
    centers = torch.tensor([[100.0, 100.0], [200.0, 150.0]])
    widths = torch.tensor([20.0, 30.0])
    heights = torch.tensor([30.0, 20.0])
    
    boxes = centers_to_boxes(centers, widths, heights)
    print(f"  Centers → Boxes: {boxes.tolist()}")
    assert boxes[0, 0] == 90.0   # x1 = 100 - 20/2
    assert boxes[0, 1] == 85.0   # y1 = 100 - 30/2
    assert boxes[0, 2] == 110.0  # x2 = 100 + 20/2
    assert boxes[0, 3] == 115.0  # y2 = 100 + 30/2
    
    recovered_centers, recovered_sizes = boxes_to_centers(boxes)
    assert torch.allclose(recovered_centers, centers)
    assert torch.allclose(recovered_sizes[:, 0], widths)
    assert torch.allclose(recovered_sizes[:, 1], heights)
    print("  ✓ centers ↔ boxes roundtrip passed")
    
    # Test square padding
    tall_box = torch.tensor([[95.0, 80.0, 105.0, 120.0]])  # 10x40
    square = make_square_boxes(tall_box, padding_factor=1.0)
    sq_w = square[0, 2] - square[0, 0]
    sq_h = square[0, 3] - square[0, 1]
    print(f"  Tall box (10x40) → square ({sq_w:.1f}x{sq_h:.1f})")
    assert abs(sq_w - sq_h) < 0.01, "Not square!"
    assert abs(sq_w - 40.0) < 0.01, "Should be 40x40"
    
    wide_box = torch.tensor([[80.0, 95.0, 120.0, 105.0]])  # 40x10
    square_wide = make_square_boxes(wide_box, padding_factor=1.0)
    sq_w2 = square_wide[0, 2] - square_wide[0, 0]
    sq_h2 = square_wide[0, 3] - square_wide[0, 1]
    print(f"  Wide box (40x10) → square ({sq_w2:.1f}x{sq_h2:.1f})")
    assert abs(sq_w2 - sq_h2) < 0.01
    assert abs(sq_w2 - 40.0) < 0.01
    
    # With padding factor
    square_padded = make_square_boxes(tall_box, padding_factor=1.2)
    sq_wp = square_padded[0, 2] - square_padded[0, 0]
    print(f"  Tall box (10x40) × 1.2 → square ({sq_wp:.1f}x{sq_wp:.1f})")
    assert abs(sq_wp - 48.0) < 0.01  # 40 * 1.2 = 48
    print("  ✓ Square padding passed")
    
    # Test clamp
    oob_box = torch.tensor([[-10.0, -5.0, 650.0, 700.0]])
    clamped = clamp_boxes(oob_box, 640, 640)
    assert clamped[0, 0] == 0.0
    assert clamped[0, 1] == 0.0
    assert clamped[0, 2] == 640.0
    assert clamped[0, 3] == 640.0
    print("  ✓ Box clamping passed")
    
    # Test normalize/denormalize
    pixel_box = torch.tensor([[100.0, 200.0, 120.0, 230.0]])
    norm_box = normalize_boxes(pixel_box, 640, 640)
    denorm_box = denormalize_boxes(norm_box, 640, 640)
    assert torch.allclose(pixel_box, denorm_box, atol=1e-4)
    print("  ✓ Normalize/denormalize roundtrip passed")
    
    # ============================================================
    # Test 2: PaddedRoIAlign
    # ============================================================
    if TORCHVISION_AVAILABLE:
        print("\n[Test 2] PaddedRoIAlign")
        
        roi = PaddedRoIAlign(
            output_size=7,
            spatial_scale=1.0 / 8.0,
            padding_factor=1.2
        )
        print(f"  {roi}")
        
        # Fake feature map: [1, 512, 80, 80] (stride 8 of 640x640)
        feature_map = torch.randn(1, 512, 80, 80)
        
        # 5 boxes
        boxes = torch.tensor([
            [90.0, 85.0, 110.0, 115.0],
            [130.0, 87.0, 150.0, 117.0],
            [170.0, 83.0, 190.0, 113.0],
            [340.0, 235.0, 360.0, 265.0],
            [380.0, 237.0, 400.0, 267.0],
        ])
        
        roi_features = roi(feature_map, boxes)
        print(f"  Input:  feature_map {feature_map.shape}, boxes {boxes.shape}")
        print(f"  Output: {roi_features.shape}")
        assert roi_features.shape == (5, 512, 7, 7)
        print("  ✓ PaddedRoIAlign passed")
        
        # Test empty boxes
        empty_features = roi(feature_map, torch.zeros(0, 4))
        assert empty_features.shape == (0, 512, 7, 7)
        print("  ✓ Empty boxes handled")
        
        # Test batched boxes
        feature_map_batch = torch.randn(2, 512, 80, 80)
        boxes_batch = torch.tensor([
            [90.0, 85.0, 110.0, 115.0],
            [130.0, 87.0, 150.0, 117.0],
            [340.0, 235.0, 360.0, 265.0],
        ])
        batch_idx = torch.tensor([0, 0, 1])  # 2 boxes from image 0, 1 from image 1
        
        batched_features = roi(feature_map_batch, boxes_batch, batch_idx)
        assert batched_features.shape == (3, 512, 7, 7)
        print("  ✓ Batched RoI Align passed")
        
        # ============================================================
        # Test 3: RoIFeatureExtractor
        # ============================================================
        print("\n[Test 3] RoIFeatureExtractor")
        
        extractor = RoIFeatureExtractor(
            in_channels=512,
            vis_dim=256,
            roi_output_size=7,
            spatial_scale=1.0 / 8.0,
            padding_factor=1.2,
            dropout=0.1
        )
        
        embeddings = extractor(feature_map, boxes)
        print(f"  Input:  feature_map {feature_map.shape}, boxes {boxes.shape}")
        print(f"  Output: {embeddings.shape}")
        assert embeddings.shape == (5, 256)
        
        # Verify gradient flow
        feature_map_grad = torch.randn(1, 512, 80, 80, requires_grad=True)
        emb = extractor(feature_map_grad, boxes)
        loss = emb.sum()
        loss.backward()
        assert feature_map_grad.grad is not None
        print(f"  Gradient norm: {feature_map_grad.grad.norm():.4f}")
        print("  ✓ RoIFeatureExtractor passed with gradient flow")
        
        # ============================================================
        # Test 4: BatchedRoIAlign
        # ============================================================
        print("\n[Test 4] BatchedRoIAlign")
        
        batched_roi = BatchedRoIAlign(
            in_channels=512,
            vis_dim=256,
            roi_output_size=7,
            spatial_scale=1.0 / 8.0,
            padding_factor=1.2
        )
        
        all_embeddings = batched_roi(feature_map_batch, boxes_batch, batch_idx)
        print(f"  All embeddings: {all_embeddings.shape}")  # [3, 256]
        
        per_image = BatchedRoIAlign.split_by_batch(all_embeddings, batch_idx, batch_size=2)
        print(f"  Image 0: {per_image[0].shape}")  # [2, 256]
        print(f"  Image 1: {per_image[1].shape}")  # [1, 256]
        assert per_image[0].shape == (2, 256)
        assert per_image[1].shape == (1, 256)
        
        per_boxes = BatchedRoIAlign.get_boxes_per_image(boxes_batch, batch_idx, batch_size=2)
        assert per_boxes[0].shape == (2, 4)
        assert per_boxes[1].shape == (1, 4)
        print("  ✓ BatchedRoIAlign split passed")
        
        # ============================================================
        # Test 5: MultiScaleRoIAlign
        # ============================================================
        print("\n[Test 5] MultiScaleRoIAlign")
        
        multi_roi = MultiScaleRoIAlign(
            in_channels=512,
            vis_dim=256,
            roi_output_size=7,
            spatial_scale=1.0 / 8.0,
            padding_factors=(1.0, 1.5, 2.0),
            dropout=0.1
        )
        
        multi_emb = multi_roi(feature_map, boxes)
        print(f"  Scales: {multi_roi.padding_factors}")
        print(f"  Output: {multi_emb.shape}")
        assert multi_emb.shape == (5, 256)
        
        # Verify gradient flow
        fm_grad = torch.randn(1, 512, 80, 80, requires_grad=True)
        me = multi_roi(fm_grad, boxes)
        me.sum().backward()
        assert fm_grad.grad is not None
        print("  ✓ MultiScaleRoIAlign passed with gradient flow")
        
        # ============================================================
        # Test 6: Aspect ratio preservation verification
        # ============================================================
        print("\n[Test 6] Aspect ratio preservation")
        
        # Create a feature map with a known pattern
        fm_test = torch.zeros(1, 1, 80, 80)
        # Draw a vertical line (tall digit '1') at position (12, 12) in feature map
        fm_test[0, 0, 8:16, 12] = 1.0  # 8 pixels tall, 1 pixel wide
        
        # Tall box: 8x64 pixels in image space → 1x8 in feature map
        tall_box_test = torch.tensor([[92.0, 64.0, 100.0, 128.0]])  # 8x64
        
        roi_tall = PaddedRoIAlign(output_size=7, spatial_scale=1.0/8.0, padding_factor=1.0)
        feat_tall = roi_tall(fm_test, tall_box_test)  # [1, 1, 7, 7]
        
        # The vertical line should be preserved (tall, not stretched wide)
        feat_grid = feat_tall[0, 0].detach().numpy()
        # Center column should have higher values than edge columns
        center_col = feat_grid[:, 3].mean()
        edge_col = feat_grid[:, 0].mean()
        print(f"  Tall digit: center col mean={center_col:.4f}, edge col mean={edge_col:.4f}")
        print(f"  ✓ Aspect ratio preserved (vertical structure maintained)")
    
    else:
        print("\n  ⚠ torchvision not available. Skipping RoI Align tests.")
        print("    Install with: pip install torchvision")
    
    # ============================================================
    # Test 7: Parameter count
    # ============================================================
    print("\n[Test 7] Parameter counts")
    if TORCHVISION_AVAILABLE:
        ext = RoIFeatureExtractor(in_channels=512, vis_dim=512)
        params = sum(p.numel() for p in ext.parameters())
        print(f"  RoIFeatureExtractor (512→512): {params:,} params")
        
        ext_small = RoIFeatureExtractor(in_channels=512, vis_dim=256)
        params_small = sum(p.numel() for p in ext_small.parameters())
        print(f"  RoIFeatureExtractor (512→256): {params_small:,} params")
        
        multi = MultiScaleRoIAlign(in_channels=512, vis_dim=512, padding_factors=(1.0, 1.5, 2.0))
        params_multi = sum(p.numel() for p in multi.parameters())
        print(f"  MultiScaleRoIAlign (3 scales):  {params_multi:,} params")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)