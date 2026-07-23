"""
models/detector/heatmap.py
Heatmap prediction head, target generation, focal loss, and pre-training.

Contains everything related to PRODUCING and TRAINING the heatmap:
  1. HeatmapHead — the neural network that predicts digit center heatmaps.
  2. HeatmapTargetGenerator — creates Gaussian blob ground-truth targets.
  3. heatmap_focal_loss — CornerNet/CenterNet focal loss for training.
  4. DetectorTrainer — pre-training loop for the detector.

Does NOT contain decoding, NMS, or box estimation.
Those live in postprocess.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# ==============================================================
# 1. HEATMAP HEAD (Neural Network)
# ==============================================================

class HeatmapHead(nn.Module):
    """
    Predicts a single-channel heatmap of digit center probabilities.
    
    Architecture:
      Conv2d(C, 128, 3x3) → BN → ReLU →
      Conv2d(128, 128, 3x3) → BN → ReLU →
      Conv2d(128, 1, 1x1)
    
    Input:  [B, C, H/8, W/8] — backbone feature map (stride 8).
    Output: [B, 1, H/8, W/8] — raw logits (before sigmoid).
    
    The final conv bias is initialized to -2.0 so that
    sigmoid(-2) ≈ 0.12, preventing the network from starting
    with all pixels at 0.5 (which would produce huge focal loss).
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 128
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        )
        
        # Initialize final bias to -2.0 (sigmoid(-2) ≈ 0.12)
        nn.init.constant_(self.head[-1].bias, -2.0)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] — backbone feature map.
        Returns:
            [B, 1, H, W] — raw logits.
        """
        return self.head(feature_map)
    
    def predict(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Inference convenience: returns probabilities (after sigmoid).
        
        Args:
            feature_map: [B, C, H, W]
        Returns:
            [B, 1, H, W] — probabilities in [0, 1].
        """
        return torch.sigmoid(self.forward(feature_map))


# ==============================================================
# 2. HEATMAP TARGET GENERATOR
# ==============================================================

class HeatmapTargetGenerator:
    """
    Generates ground-truth heatmaps with Gaussian blobs at digit centers.
    
    Each digit center produces a Gaussian blob on the downsampled grid:
        G(x, y) = exp(-(dx² + dy²) / (2σ²))
    
    where (dx, dy) is the offset from the center in feature-map pixels.
    
    The peak value is 1.0 (exact center). Values decay to ~0 at 3σ.
    Overlapping blobs are merged via element-wise max (not sum),
    so the peak remains 1.0 even if two digits are very close.
    """
    
    def __init__(self, stride: int = 8, sigma: float = 1.0):
        """
        Args:
            stride: Feature map stride (8 for stride-8 backbone).
            sigma: Gaussian standard deviation in feature-map pixels.
                   σ=1.0 → blob covers ~7x7 feature-map pixels (3σ radius).
        """
        self.stride = stride
        self.sigma = sigma
        self.radius = int(3 * sigma)
    
    def generate(
        self,
        centers_px: torch.Tensor,
        img_height: int,
        img_width: int
    ) -> torch.Tensor:
        """
        Generate a single heatmap target.
        
        Args:
            centers_px: [N, 2] — digit centers in pixels (x, y).
            img_height: Original image height.
            img_width: Original image width.
        
        Returns:
            [1, H/stride, W/stride] — heatmap target with values in [0, 1].
        """
        h = img_height // self.stride
        w = img_width // self.stride
        heatmap = torch.zeros(1, h, w)
        
        N = centers_px.shape[0]
        for i in range(N):
            cx = centers_px[i, 0].item() / self.stride
            cy = centers_px[i, 1].item() / self.stride
            cx_int = int(round(cx))
            cy_int = int(round(cy))
            
            for dy in range(-self.radius, self.radius + 1):
                for dx in range(-self.radius, self.radius + 1):
                    y = cy_int + dy
                    x = cx_int + dx
                    if 0 <= y < h and 0 <= x < w:
                        val = math.exp(-(dx**2 + dy**2) / (2 * self.sigma**2))
                        heatmap[0, y, x] = max(heatmap[0, y, x].item(), val)
        
        return heatmap
    
    def generate_batch(
        self,
        centers_list: List[torch.Tensor],
        img_height: int,
        img_width: int
    ) -> torch.Tensor:
        """
        Generate heatmap targets for a batch.
        
        Args:
            centers_list: List of [N_i, 2] tensors (one per image).
            img_height: Image height.
            img_width: Image width.
        
        Returns:
            [B, 1, H/stride, W/stride] — batched targets.
        """
        return torch.stack([
            self.generate(c, img_height, img_width) for c in centers_list
        ], dim=0)


# ==============================================================
# 3. FOCAL LOSS
# ==============================================================

def heatmap_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0
) -> torch.Tensor:
    """
    Heatmap Focal Loss (CornerNet / CenterNet variant).
    
    Solves the extreme foreground/background imbalance (99.5% background)
    by down-weighting easy negatives near digit centers.
    
    For positive pixels (target == 1):
        L_pos = -(1 - p)^α * log(p)
    
    For negative pixels (target < 1):
        L_neg = -(1 - t)^β * p^α * log(1 - p)
    
    The (1 - t)^β term is the key: pixels near the Gaussian peak
    (t ≈ 0.9) get down-weighted by (0.1)^4 = 0.0001. Pixels far
    from any digit (t ≈ 0) get full weight. This prevents the
    massive number of background pixels from overwhelming the
    gradient signal from the few positive pixels.
    
    Args:
        pred: [B, 1, H, W] — raw logits (before sigmoid).
        target: [B, 1, H, W] — Gaussian heatmap targets in [0, 1].
        alpha: Focusing parameter for positives (higher = harder focus).
        beta: Down-weighting exponent for easy negatives.
    
    Returns:
        Scalar loss, normalized by number of positive peaks.
    """
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    
    pos_mask = target.eq(1).float()
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    
    neg_mask = target.lt(1).float()
    neg_loss = (
        -((1 - target) ** beta)
        * (pred ** alpha)
        * torch.log(1 - pred)
        * neg_mask
    )
    
    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


# ==============================================================
# 4. DETECTOR PRE-TRAINER
# ==============================================================

@dataclass
class DetectorTrainerConfig:
    """Configuration for detector pre-training."""
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_epochs: int = 30
    warmup_epochs: int = 3
    batch_size: int = 16
    num_workers: int = 4
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    freeze_backbone_epochs: int = 5
    checkpoint_dir: str = './checkpoints/detector'
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


class DetectorTrainer:
    """
    Pre-trains the heatmap head (and optionally the backbone)
    on synthetic data before joint training with the controller.
    
    Usage:
        trainer = DetectorTrainer(config, backbone)
        trainer.fit(train_loader, val_loader)
        trainer.load_into_backbone(full_model.backbone)
    """
    
    def __init__(
        self,
        config: DetectorTrainerConfig,
        backbone: nn.Module,
        in_channels: int = 512,
        hidden_channels: int = 128,
        stride: int = 8
    ):
        self.cfg = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        
        self.backbone = backbone.to(self.device)
        self.heatmap_head = HeatmapHead(in_channels, hidden_channels).to(self.device)
        self.target_gen = HeatmapTargetGenerator(stride=stride, sigma=1.0)
        
        self.optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': config.backbone_lr, 'weight_decay': config.weight_decay},
            {'params': self.heatmap_head.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        ])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs - config.warmup_epochs, eta_min=1e-7
        )
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        bp = sum(p.numel() for p in self.backbone.parameters())
        hp = sum(p.numel() for p in self.heatmap_head.parameters())
        print(f"[DetectorTrainer] Backbone: {bp:,} | Head: {hp:,} | Device: {self.device}")
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """Main pre-training loop."""
        print(f"\n{'='*60}")
        print(f"  Detector Pre-Training: {self.cfg.num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            self.current_epoch = epoch
            self._update_freeze(epoch)
            
            train_loss = self._train_epoch(epoch, train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(epoch, val_loader)
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save('best')
                    print(f"  ★ Best val loss: {val_loss:.4f}")
            
            if (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                self._save(f'epoch_{epoch+1}')
            
            if epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()
        
        self._save('final')
        print(f"\n  Pre-training complete. Best: {self.best_val_loss:.4f}")
    
    def _update_freeze(self, epoch: int) -> None:
        if epoch < self.cfg.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if epoch == 0:
                print(f"  Backbone frozen for {self.cfg.freeze_backbone_epochs} epochs")
        elif epoch == self.cfg.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            print(f"  Backbone unfrozen at epoch {epoch}")
    
    def _train_epoch(self, epoch: int, loader: DataLoader) -> float:
        self.backbone.train()
        self.heatmap_head.train()
        total, count = 0.0, 0
        t0 = time.time()
        
        for batch in loader:
            images = batch['image'].to(self.device)
            targets = batch['heatmap_target'].to(self.device)
            
            fm = self.backbone(images)
            if isinstance(fm, dict):
                fm = fm['feature_map']
            
            logits = self.heatmap_head(fm)
            loss = heatmap_focal_loss(logits, targets, self.cfg.focal_alpha, self.cfg.focal_beta)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.heatmap_head.parameters()),
                self.cfg.max_grad_norm
            )
            self.optimizer.step()
            
            total += loss.item()
            count += 1
            self.global_step += 1
            
            if self.global_step % self.cfg.log_every_n_steps == 0:
                print(f"  [Ep {epoch+1} | Step {self.global_step}] loss={loss.item():.4f} ({time.time()-t0:.1f}s)")
        
        avg = total / max(count, 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f} ({time.time()-t0:.1f}s)")
        return avg
    
    @torch.no_grad()
    def _validate(self, epoch: int, loader: DataLoader) -> float:
        self.backbone.eval()
        self.heatmap_head.eval()
        total, count = 0.0, 0
        
        for batch in loader:
            images = batch['image'].to(self.device)
            targets = batch['heatmap_target'].to(self.device)
            fm = self.backbone(images)
            if isinstance(fm, dict):
                fm = fm['feature_map']
            logits = self.heatmap_head(fm)
            total += heatmap_focal_loss(logits, targets).item()
            count += 1
        
        avg = total / max(count, 1)
        print(f"  [Val Ep {epoch+1}] loss={avg:.4f}")
        return avg
    
    def _save(self, tag: str) -> str:
        path = os.path.join(self.cfg.checkpoint_dir, f'detector_{tag}.pt')
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'backbone_state_dict': self.backbone.state_dict(),
            'heatmap_head_state_dict': self.heatmap_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"  Saved: {path}")
        return path
    
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt['backbone_state_dict'])
        self.heatmap_head.load_state_dict(ckpt['heatmap_head_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"  Loaded: {path} (epoch {self.current_epoch})")
    
    def load_into_backbone(self, target_backbone: nn.Module) -> None:
        """Copy pre-trained backbone weights into the full model."""
        src = self.backbone.state_dict()
        tgt = target_backbone.state_dict()
        loaded = 0
        for k in src:
            if k in tgt and src[k].shape == tgt[k].shape:
                tgt[k] = src[k]
                loaded += 1
        target_backbone.load_state_dict(tgt)
        print(f"  Loaded {loaded} backbone weights into full model")
    
    def load_head_into(self, target_head: HeatmapHead) -> None:
        """Copy pre-trained heatmap head weights."""
        target_head.load_state_dict(self.heatmap_head.state_dict())
        print("  Loaded heatmap head weights")


if __name__ == "__main__":
    print("=" * 60)
    print("  models/detector/heatmap.py — Unit Test")
    print("=" * 60)
    
    # Test 1: HeatmapHead
    print("\n[Test 1] HeatmapHead")
    head = HeatmapHead(512, 128)
    fm = torch.randn(2, 512, 80, 80)
    logits = head(fm)
    probs = head.predict(fm)
    assert logits.shape == (2, 1, 80, 80)
    assert probs.shape == (2, 1, 80, 80)
    assert probs.min() >= 0 and probs.max() <= 1
    print(f"  Output: {logits.shape}, range=[{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Params: {sum(p.numel() for p in head.parameters()):,}")
    print("  ✓ Passed")
    
    # Test 2: Target generator
    print("\n[Test 2] HeatmapTargetGenerator")
    gen = HeatmapTargetGenerator(stride=8, sigma=1.0)
    centers = torch.tensor([[100.0, 100.0], [300.0, 250.0]])
    target = gen.generate(centers, 640, 640)
    assert target.shape == (1, 80, 80)
    assert target.max() > 0.99
    print(f"  Shape: {target.shape}, max={target.max():.4f}, nonzero={int((target>0).sum())}")
    
    batch_target = gen.generate_batch([centers, centers[:1]], 640, 640)
    assert batch_target.shape == (2, 1, 80, 80)
    print("  ✓ Passed")
    
    # Test 3: Focal loss
    print("\n[Test 3] Focal Loss")
    pred = torch.randn(2, 1, 80, 80)
    loss = heatmap_focal_loss(pred, batch_target)
    assert not torch.isnan(loss) and loss.item() > 0
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    
    # All-background edge case
    bg_target = torch.zeros(1, 1, 80, 80)
    bg_loss = heatmap_focal_loss(torch.randn(1, 1, 80, 80), bg_target)
    assert not torch.isnan(bg_loss)
    print(f"  All-bg loss: {bg_loss.item():.4f}")
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)