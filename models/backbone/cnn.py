"""
models/backbone/cnn.py
Stride-8 ResNet-18 Visual Backbone with Padded RoI Align and Heatmap Head.

Responsibilities:
  1. Extract a stride-8 feature map from the input image.
  2. Produce a global CLS token via Global Average Pooling.
  3. Extract per-node visual embeddings via Padded RoI Align (aspect-ratio preserving).
  4. Predict a digit center heatmap for the detector (optional).

ARCHITECTURAL INVARIANT:
  The backbone produces ONLY visual features. It knows nothing about
  the graph, the controller, or the reading order. It is a pure
  vision module that maps pixels to embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from torchvision.ops import roi_align
from torchvision.models import resnet18, ResNet18_Weights


# ==============================================================
# STRIDE-8 RESNET-18
# ==============================================================

class Stride8ResNet18(nn.Module):
    """
    Modified ResNet-18 that outputs a stride-8 feature map.
    
    Standard ResNet-18 stride progression:
      conv1 (stride 2) → maxpool (stride 2) → layer1 (stride 1)
      → layer2 (stride 2) → layer3 (stride 2) → layer4 (stride 2)
      Total stride: 32
    
    Modified stride progression:
      conv1 (stride 2) → maxpool (stride 2) → layer1 (stride 1)
      → layer2 (stride 2) → layer3 (stride 1*) → layer4 (stride 1*)
      Total stride: 8
    
    *layer3 and layer4 downsampling is removed (stride changed from 2 to 1).
    This preserves spatial resolution while maintaining the full depth
    and receptive field of ResNet-18.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load base ResNet-18
        if pretrained:
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            base = resnet18(weights=None)
        
        # Extract layers
        self.conv1 = base.conv1       # 7x7, stride 2, 64 channels
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool   # 3x3, stride 2
        
        self.layer1 = base.layer1     # stride 1, 64 channels
        self.layer2 = base.layer2     # stride 2, 128 channels
        
        # Modify layer3: remove downsampling (stride 2 → 1)
        self.layer3 = base.layer3
        self._remove_downsample(self.layer3)
        # Output: 256 channels, stride 8
        
        # Modify layer4: remove downsampling (stride 2 → 1)
        self.layer4 = base.layer4
        self._remove_downsample(self.layer4)
        # Output: 512 channels, stride 8
        
        self.out_channels = 512
        self.stride = 8
    
    def _remove_downsample(self, layer: nn.Sequential) -> None:
        """
        Remove downsampling from the first block of a ResNet layer.
        Changes the stride of the first block's conv layers from 2 to 1.
        """
        first_block = layer[0]
        
        # Modify conv1 stride (the 3x3 conv in BasicBlock)
        if hasattr(first_block, 'conv1'):
            first_block.conv1.stride = (1, 1)
        
        # Modify conv2 stride (the 3x3 conv in BasicBlock)
        if hasattr(first_block, 'conv2'):
            first_block.conv2.stride = (1, 1)
        
        # Fix the downsample layer (1x1 conv with stride 2 → stride 1)
        if first_block.downsample is not None:
            for module in first_block.downsample.modules():
                if isinstance(module, nn.Conv2d):
                    module.stride = (1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image.
        
        Returns:
            [B, 512, H/8, W/8] stride-8 feature map.
        """
        x = self.conv1(x)       # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, H/4, W/4]
        
        x = self.layer1(x)      # [B, 64, H/4, W/4]
        x = self.layer2(x)      # [B, 128, H/8, W/8]
        x = self.layer3(x)      # [B, 256, H/8, W/8]
        x = self.layer4(x)      # [B, 512, H/8, W/8]
        
        return x


# ==============================================================
# PADDED ROI ALIGN
# ==============================================================

class PaddedRoIAlign(nn.Module):
    """
    Aspect-ratio-preserving RoI Align.
    
    Standard RoI Align forces every bounding box into a fixed HxW grid,
    distorting the aspect ratio. A tall '1' gets stretched horizontally;
    a wide '0' gets squished vertically.
    
    Padded RoI Align fixes this by expanding each bounding box to a
    perfect square (padding the shorter dimension with background)
    BEFORE applying RoI Align. The internal geometry of the digit
    is perfectly preserved within the square grid.
    
    Example:
      Original box: 10x40 pixels (tall '1')
      Square box:   40x40 pixels (padded left/right with background)
      RoI Align:    40x40 → 7x7 grid (aspect ratio preserved)
    """
    
    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1.0 / 8.0,
        padding_factor: float = 1.2
    ):
        """
        Args:
            output_size: Size of the RoI Align output grid (7 → 7x7).
            spatial_scale: Scale from image pixels to feature map pixels (1/stride).
            padding_factor: How much to expand the square box (1.0 = exact fit,
                           1.2 = 20% padding for context).
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.padding_factor = padding_factor
    
    def make_square_boxes(
        self, boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand bounding boxes to perfect squares, preserving aspect ratio.
        
        Args:
            boxes: [N, 4] — (x1, y1, x2, y2) in image pixel coordinates.
        
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
        s = torch.max(w, h) * self.padding_factor
        
        # New square box
        new_x1 = cx - s / 2.0
        new_y1 = cy - s / 2.0
        new_x2 = cx + s / 2.0
        new_y2 = cy + s / 2.0
        
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
    
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Padded RoI Align.
        
        Args:
            feature_map: [B, C, H, W] — stride-8 feature map.
            boxes: [N, 4] — (x1, y1, x2, y2) in image pixel coordinates.
            batch_indices: [N] — batch index for each box (default: all 0).
        
        Returns:
            [N, C, output_size, output_size] — RoI features.
        """
        N = boxes.shape[0]
        if N == 0:
            C = feature_map.shape[1]
            return torch.zeros(
                0, C, self.output_size, self.output_size,
                device=feature_map.device
            )
        
        # Expand to square boxes
        square_boxes = self.make_square_boxes(boxes)  # [N, 4]
        
        # Add batch indices: [N, 5] — (batch_idx, x1, y1, x2, y2)
        if batch_indices is None:
            batch_indices = torch.zeros(N, device=boxes.device)
        
        rois = torch.cat([
            batch_indices.unsqueeze(1).float(),
            square_boxes
        ], dim=1)  # [N, 5]
        
        # Apply RoI Align
        roi_features = roi_align(
            feature_map,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            aligned=True  # Use aligned mode for better gradient flow
        )  # [N, C, output_size, output_size]
        
        return roi_features


# ==============================================================
# HEATMAP HEAD (Digit Center Detector)
# ==============================================================

class HeatmapHead(nn.Module):
    """
    Lightweight heatmap predictor for digit center detection.
    
    Predicts a single-channel heatmap where each pixel value represents
    the probability of a digit center being at that location.
    
    Trained with Focal Loss (CornerNet/CenterNet variant) to handle
    the extreme foreground/background imbalance (99.5% background).
    """
    
    def __init__(self, in_channels: int = 512, hidden_channels: int = 128):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)  # Single channel output
        )
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] — stride-8 feature map.
        
        Returns:
            [B, 1, H, W] — raw logits (before sigmoid).
        """
        return self.head(feature_map)
    
    @staticmethod
    def focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0
    ) -> torch.Tensor:
        """
        Heatmap Focal Loss (CornerNet/CenterNet variant).
        
        Args:
            pred: [B, 1, H, W] — raw logits (before sigmoid).
            target: [B, 1, H, W] — ground-truth Gaussian heatmap [0, 1].
            alpha: Focusing parameter for positive samples.
            beta: Down-weighting parameter for easy negatives.
        
        Returns:
            Scalar loss.
        """
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        
        # Positive locations: where target == 1 (exact centers)
        pos_mask = target.eq(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        
        # Negative locations: where target < 1
        neg_mask = target.lt(1).float()
        neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
        
        # Normalize by number of positive peaks
        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss
    
    @staticmethod
    def generate_heatmap_target(
        centers_px: torch.Tensor,
        img_height: int,
        img_width: int,
        stride: int = 8,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Generate ground-truth heatmap with Gaussian blobs.
        
        Args:
            centers_px: [N, 2] — digit center coordinates in pixels (x, y).
            img_height: Original image height.
            img_width: Original image width.
            stride: Feature map stride.
            sigma: Gaussian standard deviation in feature map pixels.
        
        Returns:
            [1, H/stride, W/stride] — ground-truth heatmap.
        """
        h = img_height // stride
        w = img_width // stride
        
        heatmap = torch.zeros(1, h, w)
        
        for i in range(centers_px.shape[0]):
            cx = centers_px[i, 0].item() / stride
            cy = centers_px[i, 1].item() / stride
            
            # Integer center
            cx_int = int(round(cx))
            cy_int = int(round(cy))
            
            # Gaussian radius (3 sigma)
            radius = int(3 * sigma)
            
            # Generate Gaussian blob
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y = cy_int + dy
                    x = cx_int + dx
                    if 0 <= y < h and 0 <= x < w:
                        val = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                        heatmap[0, y, x] = max(heatmap[0, y, x].item(), val)
        
        return heatmap
    
    @staticmethod
    def decode_heatmap(
        heatmap_logits: torch.Tensor,
        stride: int = 8,
        top_k: int = 100,
        threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode heatmap predictions into digit center coordinates.
        Uses sub-pixel refinement via 2D parabolic fitting.
        
        Args:
            heatmap_logits: [B, 1, H, W] — raw logits.
            stride: Feature map stride.
            top_k: Maximum number of detections.
            threshold: Minimum probability threshold.
        
        Returns:
            centers_px: [N, 2] — detected center coordinates in pixels (x, y).
            scores: [N] — detection confidence scores.
        """
        heatmap = torch.sigmoid(heatmap_logits.squeeze(1))  # [B, H, W]
        B, H, W = heatmap.shape
        
        all_centers = []
        all_scores = []
        
        for b in range(B):
            hm = heatmap[b]  # [H, W]
            
            # Simple NMS: 3x3 max pooling
            hm_max = F.max_pool2d(
                hm.unsqueeze(0).unsqueeze(0),
                kernel_size=3, stride=1, padding=1
            ).squeeze(0).squeeze(0)
            
            # Keep only local maxima
            keep = (hm == hm_max).float()
            hm = hm * keep
            
            # Flatten and get top-k
            scores, indices = torch.topk(hm.view(-1), min(top_k, H * W))
            
            # Filter by threshold
            mask = scores > threshold
            scores = scores[mask]
            indices = indices[mask]
            
            if len(scores) == 0:
                continue
            
            # Convert flat indices to (y, x) coordinates
            ys = (indices // W).float()
            xs = (indices % W).float()
            
            # Sub-pixel refinement via 2D parabolic fitting
            for i in range(len(scores)):
                y_int = int(ys[i].item())
                x_int = int(xs[i].item())
                
                # Parabolic fit in x direction
                if 0 < x_int < W - 1:
                    left = hm[y_int, x_int - 1].item()
                    center = hm[y_int, x_int].item()
                    right = hm[y_int, x_int + 1].item()
                    denom = 2.0 * (2.0 * center - left - right)
                    if abs(denom) > 1e-6:
                        dx = (right - left) / denom
                    else:
                        dx = 0.0
                else:
                    dx = 0.0
                
                # Parabolic fit in y direction
                if 0 < y_int < H - 1:
                    up = hm[y_int - 1, x_int].item()
                    center = hm[y_int, x_int].item()
                    down = hm[y_int + 1, x_int].item()
                    denom = 2.0 * (2.0 * center - up - down)
                    if abs(denom) > 1e-6:
                        dy = (down - up) / denom
                    else:
                        dy = 0.0
                else:
                    dy = 0.0
                
                # Clamp offsets to [-0.5, 0.5]
                dx = max(-0.5, min(0.5, dx))
                dy = max(-0.5, min(0.5, dy))
                
                xs[i] = xs[i] + dx
                ys[i] = ys[i] + dy
            
            # Convert to pixel coordinates
            centers_x = xs * stride
            centers_y = ys * stride
            centers = torch.stack([centers_x, centers_y], dim=1)  # [N, 2]
            
            all_centers.append(centers)
            all_scores.append(scores)
        
        if len(all_centers) == 0:
            device = heatmap.device
            return torch.zeros(0, 2, device=device), torch.zeros(0, device=device)
        
        return torch.cat(all_centers, dim=0), torch.cat(all_scores, dim=0)


# ==============================================================
# COMPLETE VISUAL BACKBONE
# ==============================================================

class VisualBackbone(nn.Module):
    """
    Complete visual backbone: Stride-8 ResNet-18 + Padded RoI Align + Heatmap Head.
    
    One forward pass produces:
      1. feature_map: [B, 512, H/8, W/8] — for RoI Align and heatmap.
      2. cls_token: [B, 512] — global image feature (for Mode 2 initialization).
      3. node_embeddings: [N, vis_dim] — per-digit visual embeddings.
      4. heatmap_logits: [B, 1, H/8, W/8] — digit center heatmap (optional).
    """
    
    def __init__(
        self,
        vis_dim: int = 512,
        roi_output_size: int = 7,
        pretrained: bool = True,
        enable_heatmap: bool = True,
        padding_factor: float = 1.2,
        dropout: float = 0.1
    ):
        """
        Args:
            vis_dim: Output dimensionality of per-node visual embeddings.
            roi_output_size: RoI Align output grid size (7 → 7x7).
            pretrained: Use ImageNet-pretrained ResNet-18 weights.
            enable_heatmap: If True, include the heatmap detection head.
            padding_factor: Bounding box expansion factor for Padded RoI Align.
            dropout: Dropout rate for the embedding projection.
        """
        super().__init__()
        
        self.vis_dim = vis_dim
        self.stride = 8
        
        # Stride-8 ResNet-18
        self.backbone = Stride8ResNet18(pretrained=pretrained)
        backbone_channels = self.backbone.out_channels  # 512
        
        # Padded RoI Align
        self.roi_align = PaddedRoIAlign(
            output_size=roi_output_size,
            spatial_scale=1.0 / self.stride,
            padding_factor=padding_factor
        )
        
        # Embedding projection: flatten RoI features → vis_dim
        roi_flat_dim = backbone_channels * roi_output_size * roi_output_size  # 512 * 7 * 7 = 25088
        self.embedding_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, vis_dim * 2),
            nn.LayerNorm(vis_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vis_dim * 2, vis_dim),
            nn.LayerNorm(vis_dim),
            nn.ReLU()
        )
        
        # Global Average Pooling → CLS token
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_proj = nn.Sequential(
            nn.Linear(backbone_channels, vis_dim),
            nn.LayerNorm(vis_dim),
            nn.ReLU()
        )
        
        # Heatmap Head (optional)
        self.enable_heatmap = enable_heatmap
        if enable_heatmap:
            self.heatmap_head = HeatmapHead(
                in_channels=backbone_channels,
                hidden_channels=128
            )
    
    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            image: [B, 3, H, W] — input image (normalized).
            boxes: [N, 4] — (x1, y1, x2, y2) bounding boxes in pixel coords.
                   If None, only feature_map, cls_token, and heatmap are returned.
            batch_indices: [N] — batch index for each box (default: all 0).
        
        Returns:
            Dict with:
              'feature_map': [B, 512, H/8, W/8]
              'cls_token': [B, vis_dim]
              'node_embeddings': [N, vis_dim] (if boxes provided)
              'heatmap_logits': [B, 1, H/8, W/8] (if enable_heatmap)
        """
        output = {}
        
        # 1. Extract stride-8 feature map
        feature_map = self.backbone(image)  # [B, 512, H/8, W/8]
        output['feature_map'] = feature_map
        
        # 2. Global CLS token
        gap = self.gap(feature_map).flatten(1)  # [B, 512]
        cls_token = self.cls_proj(gap)           # [B, vis_dim]
        output['cls_token'] = cls_token
        
        # 3. Per-node visual embeddings (if boxes provided)
        if boxes is not None and boxes.shape[0] > 0:
            roi_features = self.roi_align(
                feature_map, boxes, batch_indices
            )  # [N, 512, 7, 7]
            
            roi_flat = roi_features.flatten(1)  # [N, 512*7*7]
            node_embeddings = self.embedding_proj(roi_flat)  # [N, vis_dim]
            output['node_embeddings'] = node_embeddings
        
        # 4. Heatmap (if enabled)
        if self.enable_heatmap:
            heatmap_logits = self.heatmap_head(feature_map)  # [B, 1, H/8, W/8]
            output['heatmap_logits'] = heatmap_logits
        
        return output
    
    def extract_embeddings(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: extract node embeddings and CLS token only.
        Used by the DualModeController during training and inference.
        
        Args:
            image: [B, 3, H, W]
            boxes: [N, 4] — (x1, y1, x2, y2)
            batch_indices: [N]
        
        Returns:
            node_embeddings: [N, vis_dim]
            cls_token: [vis_dim] (squeezed from batch dim 1)
        """
        output = self.forward(image, boxes, batch_indices)
        cls_token = output['cls_token'].squeeze(0)  # [vis_dim]
        node_embeddings = output['node_embeddings']  # [N, vis_dim]
        return node_embeddings, cls_token


if __name__ == "__main__":
    print("=" * 60)
    print("  VisualBackbone Unit Test")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create backbone
    backbone = VisualBackbone(
        vis_dim=512,
        roi_output_size=7,
        pretrained=False,  # No download for testing
        enable_heatmap=True,
        padding_factor=1.2
    )
    backbone.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    
    # Simulate input
    B = 1
    H, W = 640, 640
    image = torch.randn(B, 3, H, W)
    
    # Simulate bounding boxes (5 digits)
    boxes = torch.tensor([
        [90.0, 85.0, 110.0, 115.0],   # digit 0: 20x30
        [130.0, 87.0, 150.0, 117.0],   # digit 1: 20x30
        [170.0, 83.0, 190.0, 113.0],   # digit 2: 20x30
        [340.0, 235.0, 360.0, 265.0],  # digit 3: 20x30
        [380.0, 237.0, 400.0, 267.0],  # digit 4: 20x30
    ])
    
    # Forward pass
    with torch.no_grad():
        output = backbone(image, boxes)
    
    print(f"\n  Input image:  {image.shape}")
    print(f"  Feature map:  {output['feature_map'].shape}")   # [1, 512, 80, 80]
    print(f"  CLS token:    {output['cls_token'].shape}")      # [1, 512]
    print(f"  Embeddings:   {output['node_embeddings'].shape}") # [5, 512]
    print(f"  Heatmap:      {output['heatmap_logits'].shape}")  # [1, 1, 80, 80]
    
    # Verify stride
    expected_h = H // 8
    expected_w = W // 8
    assert output['feature_map'].shape[2] == expected_h
    assert output['feature_map'].shape[3] == expected_w
    print(f"\n  ✓ Stride-8 verified: {H}x{W} → {expected_h}x{expected_w}")
    
    # Verify embedding dimensionality
    assert output['node_embeddings'].shape == (5, 512)
    print(f"  ✓ Embedding dim verified: [5, 512]")
    
    # Test Padded RoI Align aspect ratio preservation
    print(f"\n  Padded RoI Align Test:")
    roi_module = backbone.roi_align
    
    # Tall digit (10x40)
    tall_box = torch.tensor([[100.0, 80.0, 110.0, 120.0]])  # 10x40
    square_tall = roi_module.make_square_boxes(tall_box)
    tall_w = square_tall[0, 2] - square_tall[0, 0]
    tall_h = square_tall[0, 3] - square_tall[0, 1]
    print(f"    Tall box (10x40) → square ({tall_w:.1f}x{tall_h:.1f})")
    assert abs(tall_w - tall_h) < 0.01, "Not square!"
    
    # Wide digit (40x10)
    wide_box = torch.tensor([[100.0, 100.0, 140.0, 110.0]])  # 40x10
    square_wide = roi_module.make_square_boxes(wide_box)
    wide_w = square_wide[0, 2] - square_wide[0, 0]
    wide_h = square_wide[0, 3] - square_wide[0, 1]
    print(f"    Wide box (40x10) → square ({wide_w:.1f}x{wide_h:.1f})")
    assert abs(wide_w - wide_h) < 0.01, "Not square!"
    print(f"    ✓ Aspect ratio preserved via square padding")
    
    # Test heatmap target generation
    print(f"\n  Heatmap Target Generation Test:")
    centers = torch.tensor([[100.0, 100.0], [300.0, 250.0]])
    target = HeatmapHead.generate_heatmap_target(
        centers, img_height=640, img_width=640, stride=8, sigma=1.0
    )
    print(f"    Target shape: {target.shape}")  # [1, 80, 80]
    print(f"    Max value: {target.max():.4f}")  # Should be 1.0
    print(f"    Non-zero pixels: {(target > 0).sum().item()}")
    print(f"    ✓ Heatmap target generated")
    
    # Test heatmap decoding
    print(f"\n  Heatmap Decoding Test:")
    fake_logits = torch.randn(1, 1, 80, 80) * 0.1
    # Place strong peaks at known locations
    fake_logits[0, 0, 12, 12] = 5.0   # pixel (96, 96)
    fake_logits[0, 0, 31, 37] = 5.0   # pixel (296, 248)
    
    decoded_centers, decoded_scores = HeatmapHead.decode_heatmap(
        fake_logits, stride=8, top_k=10, threshold=0.3
    )
    print(f"    Detected {len(decoded_scores)} centers")
    for i in range(len(decoded_scores)):
        print(f"      Center {i}: ({decoded_centers[i, 0]:.1f}, {decoded_centers[i, 1]:.1f}), "
              f"score={decoded_scores[i]:.4f}")
    print(f"    ✓ Heatmap decoding with sub-pixel refinement")
    
    # Test focal loss
    print(f"\n  Focal Loss Test:")
    pred = torch.randn(1, 1, 80, 80)
    loss = HeatmapHead.focal_loss(pred, target)
    print(f"    Loss: {loss.item():.4f}")
    loss.backward()
    print(f"    ✓ Focal loss computed and backpropagated")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)