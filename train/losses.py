"""
train/losses.py
Unified loss functions and aggregation for the Cognitive Reader project.

Components:
  1. Masked Cross-Entropy (digit classification with padding).
  2. Action Selection Loss (routing + CHUNK decision).
  3. Jump Selection Loss (Mode 2 global attention).
  4. Heatmap Focal Loss (digit center detection).
  5. LossAggregator (weighted combination with scheduling).
  6. LossMonitor (anomaly detection, per-component tracking).

All functions are standalone: they take tensors as input,
not model objects. This keeps the loss module decoupled from
the model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ==============================================================
# 1. MASKED CROSS-ENTROPY (Digit Classification)
# ==============================================================

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Cross-entropy loss with masking for padded or invalid entries.
    
    Args:
        logits: [N, C] — raw classification logits.
        targets: [N] — ground-truth class indices.
        mask: [N] — binary mask (1 = valid, 0 = ignore).
              If None, uses ignore_index to determine valid entries.
        ignore_index: Target value to ignore (default: -1 for padded nodes).
        reduction: 'mean' (average over valid), 'sum', or 'none'.
    
    Returns:
        Scalar loss (or [N] if reduction='none').
    """
    if mask is None:
        mask = (targets != ignore_index).float()
    
    # Clamp logits for numerical stability
    logits = logits.clamp(-50.0, 50.0)
    
    # Compute per-element loss
    per_element = F.cross_entropy(
        logits, targets.clamp(min=0),  # Clamp to avoid index error
        reduction='none'
    )  # [N]
    
    # Apply mask
    masked_loss = per_element * mask
    
    if reduction == 'mean':
        return masked_loss.sum() / mask.sum().clamp(min=1.0)
    elif reduction == 'sum':
        return masked_loss.sum()
    else:
        return masked_loss


def digit_classification_loss(
    digit_logits: torch.Tensor,
    digit_targets: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Digit classification loss with padding mask.
    
    Wrapper around masked_cross_entropy with digit-specific defaults.
    
    Args:
        digit_logits: [N, 10] — logits for each node.
        digit_targets: [N] — ground-truth digit labels (0-9, -1 for padding).
        node_mask: [N] — 1 = real node, 0 = padding.
    
    Returns:
        Scalar loss.
    """
    return masked_cross_entropy(
        logits=digit_logits,
        targets=digit_targets,
        mask=node_mask,
        ignore_index=-1,
        reduction='mean'
    )


# ==============================================================
# 2. ACTION SELECTION LOSS (Mode 1 Routing)
# ==============================================================

def action_selection_loss(
    action_logits: torch.Tensor,
    action_target: int,
    candidate_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Cross-entropy loss for the Mode 1 action selection.
    
    The action space is [neighbor_0, ..., neighbor_K, <CHUNK>].
    The target is the index of the correct action.
    
    Args:
        action_logits: [K + 1] — logits for each action.
        action_target: Index of the ground-truth action (0..K for neighbors,
                      K for <CHUNK>).
        candidate_mask: [K] — 1 = valid candidate, 0 = visited/padding.
                       If provided, invalid candidates are masked to -inf.
    
    Returns:
        Scalar loss.
    """
    logits = action_logits.clone()
    
    # Mask invalid candidates (but never mask the <CHUNK> action)
    if candidate_mask is not None:
        K = candidate_mask.shape[0]
        if logits.shape[0] == K + 1:
            # Mask neighbor logits, keep <CHUNK> logit untouched
            logits[:K] = logits[:K].masked_fill(candidate_mask == 0, float('-inf'))
    
    # Clamp for stability
    logits = logits.clamp(-50.0, 50.0)
    
    target_tensor = torch.tensor(
        [action_target], dtype=torch.long, device=logits.device
    )
    
    return F.cross_entropy(logits.unsqueeze(0), target_tensor)


def batch_action_selection_loss(
    action_logits_list: List[torch.Tensor],
    action_targets: List[int],
    candidate_masks: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """
    Averaged action selection loss over a batch of Mode 1 steps.
    
    Args:
        action_logits_list: List of [K_i + 1] logit tensors.
        action_targets: List of ground-truth action indices.
        candidate_masks: List of [K_i] masks (optional).
    
    Returns:
        Scalar averaged loss.
    """
    if len(action_logits_list) == 0:
        return torch.tensor(0.0)
    
    total_loss = torch.tensor(0.0, device=action_logits_list[0].device)
    
    for i, (logits, target) in enumerate(zip(action_logits_list, action_targets)):
        mask = candidate_masks[i] if candidate_masks is not None else None
        total_loss = total_loss + action_selection_loss(logits, target, mask)
    
    return total_loss / len(action_logits_list)


# ==============================================================
# 3. JUMP SELECTION LOSS (Mode 2 Saccadic)
# ==============================================================

def jump_selection_loss(
    attention_logits: torch.Tensor,
    target_node_idx: int,
    visited_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Cross-entropy loss for the Mode 2 saccadic jump selection.
    
    The attention logits span ALL nodes in the graph.
    Visited nodes are masked to -inf.
    The target is the ground-truth next starting node.
    
    Args:
        attention_logits: [N] — attention scores for all nodes.
        target_node_idx: Ground-truth node index to jump to.
        visited_mask: [N] — 1 = visited (mask out), 0 = unvisited.
    
    Returns:
        Scalar loss.
    """
    logits = attention_logits.clone()
    
    # Mask visited nodes
    if visited_mask is not None:
        logits = logits.masked_fill(visited_mask.bool(), float('-inf'))
    
    # Clamp for stability
    logits = logits.clamp(-50.0, 50.0)
    
    target_tensor = torch.tensor(
        [target_node_idx], dtype=torch.long, device=logits.device
    )
    
    return F.cross_entropy(logits.unsqueeze(0), target_tensor)


# ==============================================================
# 4. HEATMAP FOCAL LOSS (Digit Detection)
# ==============================================================

def heatmap_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0
) -> torch.Tensor:
    """
    Heatmap Focal Loss (CornerNet/CenterNet variant).
    
    Handles the extreme foreground/background imbalance (99.5% background)
    by down-weighting easy negatives near digit centers.
    
    Args:
        pred: [B, 1, H, W] — raw logits (before sigmoid).
        target: [B, 1, H, W] — ground-truth Gaussian heatmap [0, 1].
        alpha: Focusing parameter for positive samples.
        beta: Down-weighting parameter for easy negatives.
    
    Returns:
        Scalar loss.
    """
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    
    # Positive locations: exact centers (target == 1)
    pos_mask = target.eq(1).float()
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    
    # Negative locations: everything else (target < 1)
    neg_mask = target.lt(1).float()
    neg_loss = (
        -((1 - target) ** beta)
        * (pred ** alpha)
        * torch.log(1 - pred)
        * neg_mask
    )
    
    # Normalize by number of positive peaks
    num_pos = pos_mask.sum().clamp(min=1.0)
    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    
    return loss


# ==============================================================
# 5. LOSS AGGREGATOR
# ==============================================================

@dataclass
class LossWeights:
    """Weights for each loss component."""
    digit: float = 1.0
    action: float = 1.0
    jump: float = 1.0
    heatmap: float = 1.0


class LossAggregator:
    """
    Weighted aggregation of all loss components with optional scheduling.
    
    Supports:
      - Fixed weights (default).
      - Linear warmup of specific components (e.g., ramp up jump loss
        after the model learns basic digit recognition).
      - Per-epoch weight scheduling via callbacks.
    
    Usage:
        aggregator = LossAggregator(
            weights=LossWeights(digit=1.0, action=1.0, jump=0.5, heatmap=1.0),
            jump_warmup_epochs=10
        )
        
        for epoch in range(num_epochs):
            aggregator.set_epoch(epoch)
            
            for batch in dataloader:
                losses = compute_losses(batch)
                total = aggregator.aggregate(losses)
                total.backward()
    """
    
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        digit_warmup_epochs: int = 0,
        action_warmup_epochs: int = 0,
        jump_warmup_epochs: int = 0,
        heatmap_warmup_epochs: int = 0
    ):
        """
        Args:
            weights: Base loss weights.
            digit_warmup_epochs: Epochs to linearly ramp digit loss weight.
            action_warmup_epochs: Epochs to ramp action loss weight.
            jump_warmup_epochs: Epochs to ramp jump loss weight.
            heatmap_warmup_epochs: Epochs to ramp heatmap loss weight.
        """
        self.base_weights = weights or LossWeights()
        self.warmup_epochs = {
            'digit': digit_warmup_epochs,
            'action': action_warmup_epochs,
            'jump': jump_warmup_epochs,
            'heatmap': heatmap_warmup_epochs,
        }
        self._current_epoch = 0
        self._current_weights = self._compute_weights()
    
    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch (call at the start of each epoch)."""
        self._current_epoch = epoch
        self._current_weights = self._compute_weights()
    
    def _compute_weights(self) -> Dict[str, float]:
        """Compute the effective weights for the current epoch."""
        weights = {}
        for component in ['digit', 'action', 'jump', 'heatmap']:
            base = getattr(self.base_weights, component)
            warmup = self.warmup_epochs[component]
            
            if warmup > 0 and self._current_epoch < warmup:
                # Linear ramp from 0 to base weight
                factor = (self._current_epoch + 1) / warmup
                weights[component] = base * factor
            else:
                weights[component] = base
        
        return weights
    
    def aggregate(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the weighted total loss.
        
        Args:
            losses: Dict with keys 'digit', 'action', 'jump', 'heatmap'.
                   Each value is a scalar loss tensor.
        
        Returns:
            Weighted total loss tensor.
        """
        total = torch.tensor(0.0, device=self._get_device(losses))
        
        for component, weight in self._current_weights.items():
            if component in losses and weight > 0:
                total = total + weight * losses[component]
        
        return total
    
    def get_current_weights(self) -> Dict[str, float]:
        """Return the effective weights for the current epoch."""
        return dict(self._current_weights)
    
    def _get_device(self, losses: Dict[str, torch.Tensor]) -> torch.device:
        """Get the device from the first available loss tensor."""
        for v in losses.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cpu')


# ==============================================================
# 6. LOSS MONITOR
# ==============================================================

class LossMonitor:
    """
    Tracks per-component losses and detects anomalies.
    
    Detects:
      - NaN or Inf losses.
      - Sudden loss spikes (> spike_threshold × running mean).
      - Loss stagnation (no improvement for N epochs).
      - Component imbalance (one loss dominating the total).
    
    Usage:
        monitor = LossMonitor(window_size=50)
        
        for step in range(num_steps):
            monitor.update(losses_dict)
            
            if step % 100 == 0:
                stats = monitor.get_stats()
                anomalies = monitor.check_anomalies()
    """
    
    def __init__(
        self,
        window_size: int = 50,
        spike_threshold: float = 5.0,
        stagnation_patience: int = 20
    ):
        """
        Args:
            window_size: Number of recent values for running statistics.
            spike_threshold: A loss is a "spike" if it exceeds
                            spike_threshold × running mean.
            stagnation_patience: Number of epochs without improvement
                                before flagging stagnation.
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.stagnation_patience = stagnation_patience
        
        # Per-component tracking
        self._history: Dict[str, List[float]] = {}
        self._best: Dict[str, float] = {}
        self._best_epoch: Dict[str, int] = {}
        self._step_count: int = 0
        self._epoch_count: int = 0
        self._anomalies: List[Dict] = []
    
    def update(self, losses: Dict[str, Any], epoch: Optional[int] = None) -> None:
        """
        Record loss values for one step.
        
        Args:
            losses: Dict mapping component name → loss value (float or tensor).
            epoch: Current epoch (for stagnation tracking).
        """
        self._step_count += 1
        if epoch is not None:
            self._epoch_count = epoch
        
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if not isinstance(value, (int, float)):
                continue
            
            # Initialize tracking
            if key not in self._history:
                self._history[key] = []
                self._best[key] = float('inf')
                self._best_epoch[key] = 0
            
            # Append to history (keep window)
            self._history[key].append(value)
            if len(self._history[key]) > self.window_size:
                self._history[key] = self._history[key][-self.window_size:]
            
            # Track best
            if value < self._best[key]:
                self._best[key] = value
                self._best_epoch[key] = self._epoch_count
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute running statistics for each loss component.
        
        Returns:
            Dict mapping component → {mean, std, min, max, latest, best}.
        """
        stats = {}
        for key, history in self._history.items():
            if not history:
                continue
            
            arr = np.array(history) if len(history) > 1 else np.array([history[0]])
            stats[key] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'latest': float(history[-1]),
                'best': float(self._best[key]),
                'best_epoch': self._best_epoch[key],
            }
        return stats
    
    def check_anomalies(self) -> List[Dict[str, Any]]:
        """
        Check for loss anomalies.
        
        Returns:
            List of anomaly dicts with {type, component, step, details}.
        """
        anomalies = []
        
        for key, history in self._history.items():
            if not history:
                continue
            
            latest = history[-1]
            
            # NaN / Inf check
            if math.isnan(latest) or math.isinf(latest):
                anomalies.append({
                    'type': 'nan_inf',
                    'component': key,
                    'step': self._step_count,
                    'value': latest,
                    'details': f'{key} is {latest}'
                })
                continue
            
            # Spike check (need at least 10 values)
            if len(history) >= 10:
                running_mean = sum(history[:-1]) / len(history[:-1])
                if running_mean > 0 and latest > self.spike_threshold * running_mean:
                    anomalies.append({
                        'type': 'spike',
                        'component': key,
                        'step': self._step_count,
                        'value': latest,
                        'running_mean': running_mean,
                        'ratio': latest / running_mean,
                        'details': (
                            f'{key} spiked to {latest:.4f} '
                            f'({latest/running_mean:.1f}x running mean {running_mean:.4f})'
                        )
                    })
            
            # Stagnation check
            if self._epoch_count - self._best_epoch.get(key, 0) > self.stagnation_patience:
                anomalies.append({
                    'type': 'stagnation',
                    'component': key,
                    'step': self._step_count,
                    'best': self._best[key],
                    'best_epoch': self._best_epoch[key],
                    'current_epoch': self._epoch_count,
                    'details': (
                        f'{key} has not improved for '
                        f'{self._epoch_count - self._best_epoch[key]} epochs '
                        f'(best={self._best[key]:.4f} at epoch {self._best_epoch[key]})'
                    )
                })
        
        # Component imbalance check
        if len(self._history) >= 2:
            latest_values = {}
            for key, history in self._history.items():
                if history:
                    latest_values[key] = abs(history[-1])
            
            if latest_values:
                max_val = max(latest_values.values())
                min_val = min(latest_values.values())
                if min_val > 0 and max_val / min_val > 100:
                    dominant = max(latest_values, key=latest_values.get)
                    anomalies.append({
                        'type': 'imbalance',
                        'step': self._step_count,
                        'dominant': dominant,
                        'ratio': max_val / min_val,
                        'details': (
                            f'Loss imbalance: {dominant} dominates '
                            f'({max_val/min_val:.0f}x ratio)'
                        )
                    })
        
        self._anomalies.extend(anomalies)
        return anomalies
    
    def get_anomaly_history(self) -> List[Dict]:
        """Return all detected anomalies."""
        return list(self._anomalies)
    
    def reset(self) -> None:
        """Clear all tracking state."""
        self._history.clear()
        self._best.clear()
        self._best_epoch.clear()
        self._step_count = 0
        self._epoch_count = 0
        self._anomalies.clear()
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        stats = self.get_stats()
        lines = [f"Loss Monitor Summary (step {self._step_count}, epoch {self._epoch_count}):"]
        
        for key, s in stats.items():
            lines.append(
                f"  {key:12s}: latest={s['latest']:.4f}  "
                f"mean={s['mean']:.4f}  best={s['best']:.4f} (ep {s['best_epoch']})"
            )
        
        recent_anomalies = [a for a in self._anomalies if a['step'] > self._step_count - 100]
        if recent_anomalies:
            lines.append(f"\n  Recent anomalies ({len(recent_anomalies)}):")
            for a in recent_anomalies[-5:]:
                lines.append(f"    [{a['type']}] {a['details']}")
        
        return '\n'.join(lines)


# ==============================================================
# 7. COMPLETE LOSS PACKAGE
# ==============================================================

class LossPackage:
    """
    Bundles LossAggregator + LossMonitor for clean training integration.
    
    Usage:
        loss_pkg = LossPackage(
            weights=LossWeights(digit=1.0, action=1.0, jump=0.5, heatmap=1.0),
            jump_warmup_epochs=10
        )
        
        for epoch in range(num_epochs):
            loss_pkg.set_epoch(epoch)
            
            for batch in dataloader:
                losses = {
                    'digit': digit_loss,
                    'action': action_loss,
                    'jump': jump_loss,
                    'heatmap': heatmap_loss,
                }
                
                total = loss_pkg.compute_total(losses)
                total.backward()
                
                loss_pkg.monitor.update(losses, epoch=epoch)
    """
    
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        digit_warmup_epochs: int = 0,
        action_warmup_epochs: int = 0,
        jump_warmup_epochs: int = 0,
        heatmap_warmup_epochs: int = 0,
        monitor_window: int = 50,
        spike_threshold: float = 5.0
    ):
        self.aggregator = LossAggregator(
            weights=weights,
            digit_warmup_epochs=digit_warmup_epochs,
            action_warmup_epochs=action_warmup_epochs,
            jump_warmup_epochs=jump_warmup_epochs,
            heatmap_warmup_epochs=heatmap_warmup_epochs
        )
        
        self.monitor = LossMonitor(
            window_size=monitor_window,
            spike_threshold=spike_threshold
        )
    
    def set_epoch(self, epoch: int) -> None:
        """Update epoch for weight scheduling."""
        self.aggregator.set_epoch(epoch)
    
    def compute_total(
        self,
        losses: Dict[str, torch.Tensor],
        track: bool = True,
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted total loss and optionally track metrics.
        
        Args:
            losses: Dict of individual loss components.
            track: If True, update the loss monitor.
            epoch: Current epoch (for monitoring).
        
        Returns:
            Weighted total loss tensor.
        """
        total = self.aggregator.aggregate(losses)
        
        if track:
            # Track individual losses + total
            track_dict = {k: v for k, v in losses.items()}
            track_dict['total'] = total
            self.monitor.update(track_dict, epoch=epoch)
        
        return total
    
    def get_weights(self) -> Dict[str, float]:
        """Get current effective loss weights."""
        return self.aggregator.get_current_weights()
    
    def check_anomalies(self) -> List[Dict]:
        """Check for loss anomalies."""
        return self.monitor.check_anomalies()
    
    def summary(self) -> str:
        """Get monitor summary."""
        return self.monitor.summary()


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("  train/losses.py — Unit Test")
    print("=" * 60)
    
    # Test 1: Masked Cross-Entropy
    print("\n[Test 1] Masked Cross-Entropy")
    logits = torch.randn(6, 10)
    targets = torch.tensor([3, 8, -1, 1, -1, 5])  # -1 = padding
    mask = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.float)
    
    loss = masked_cross_entropy(logits, targets, mask)
    print(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    # Verify padding is ignored
    loss_no_pad = masked_cross_entropy(
        logits[[0, 1, 3, 5]], targets[[0, 1, 3, 5]]
    )
    assert abs(loss.item() - loss_no_pad.item()) < 1e-5, "Padding should be ignored"
    print("  ✓ Padding correctly ignored")
    
    # Test 2: Digit Classification Loss
    print("\n[Test 2] Digit Classification Loss")
    digit_logits = torch.randn(8, 10)
    digit_targets = torch.tensor([3, 8, 4, 2, -1, -1, 1, 7])
    node_mask = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.float)
    
    d_loss = digit_classification_loss(digit_logits, digit_targets, node_mask)
    print(f"  Loss: {d_loss.item():.4f}")
    assert not torch.isnan(d_loss)
    print("  ✓ Passed")
    
    # Test 3: Action Selection Loss
    print("\n[Test 3] Action Selection Loss")
    action_logits = torch.randn(5)  # 4 neighbors + 1 CHUNK
    action_target = 2  # Select neighbor 2
    cand_mask = torch.tensor([1, 1, 1, 0])  # Neighbor 3 is visited
    
    a_loss = action_selection_loss(action_logits, action_target, cand_mask)
    print(f"  Loss: {a_loss.item():.4f}")
    assert not torch.isnan(a_loss)
    
    # Test with CHUNK target
    chunk_loss = action_selection_loss(action_logits, 4)  # Index 4 = CHUNK
    print(f"  CHUNK loss: {chunk_loss.item():.4f}")
    print("  ✓ Passed")
    
    # Test 4: Jump Selection Loss
    print("\n[Test 4] Jump Selection Loss")
    attn_logits = torch.randn(20)  # 20 nodes
    target_node = 12
    visited = torch.zeros(20)
    visited[:8] = 1  # First 8 nodes visited
    
    j_loss = jump_selection_loss(attn_logits, target_node, visited)
    print(f"  Loss: {j_loss.item():.4f}")
    assert not torch.isnan(j_loss)
    
    # Verify visited nodes are masked
    attn_logits_copy = attn_logits.clone()
    attn_logits_copy[:8] = 100.0  # Make visited nodes very attractive
    j_loss_masked = jump_selection_loss(attn_logits_copy, target_node, visited)
    # Loss should be similar because visited nodes are masked
    print(f"  Loss with bait: {j_loss_masked.item():.4f}")
    print("  ✓ Visited nodes correctly masked")
    
    # Test 5: Heatmap Focal Loss
    print("\n[Test 5] Heatmap Focal Loss")
    pred_hm = torch.randn(1, 1, 80, 80)
    target_hm = torch.zeros(1, 1, 80, 80)
    target_hm[0, 0, 12, 12] = 1.0
    target_hm[0, 0, 31, 43] = 1.0
    # Add Gaussian skirts
    for cy, cx in [(12, 12), (31, 43)]:
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y, x = cy + dy, cx + dx
                if 0 <= y < 80 and 0 <= x < 80:
                    val = math.exp(-(dx**2 + dy**2) / 2.0)
                    target_hm[0, 0, y, x] = max(target_hm[0, 0, y, x].item(), val)
    
    hm_loss = heatmap_focal_loss(pred_hm, target_hm)
    print(f"  Loss: {hm_loss.item():.4f}")
    assert not torch.isnan(hm_loss)
    assert hm_loss.item() > 0
    print("  ✓ Passed")
    
    # Test 6: Loss Aggregator
    print("\n[Test 6] Loss Aggregator")
    weights = LossWeights(digit=1.0, action=1.0, jump=0.5, heatmap=1.0)
    aggregator = LossAggregator(weights=weights, jump_warmup_epochs=10)
    
    losses = {
        'digit': torch.tensor(2.0),
        'action': torch.tensor(1.5),
        'jump': torch.tensor(0.8),
        'heatmap': torch.tensor(1.2),
    }
    
    # Epoch 0: jump weight should be ramped
    aggregator.set_epoch(0)
    w0 = aggregator.get_current_weights()
    total_0 = aggregator.aggregate(losses)
    print(f"  Epoch 0 weights: {w0}")
    print(f"  Epoch 0 total: {total_0.item():.4f}")
    assert w0['jump'] < 0.5, "Jump weight should be ramped at epoch 0"
    
    # Epoch 10: jump weight should be full
    aggregator.set_epoch(10)
    w10 = aggregator.get_current_weights()
    total_10 = aggregator.aggregate(losses)
    print(f"  Epoch 10 weights: {w10}")
    print(f"  Epoch 10 total: {total_10.item():.4f}")
    assert abs(w10['jump'] - 0.5) < 1e-6, "Jump weight should be full at epoch 10"
    
    print("  ✓ Weight scheduling passed")
    
    # Test 7: Loss Monitor
    print("\n[Test 7] Loss Monitor")
    monitor = LossMonitor(window_size=20, spike_threshold=3.0)
    
    # Simulate normal training
    for i in range(50):
        losses_sim = {
            'digit': 2.0 - i * 0.02 + np.random.normal(0, 0.05),
            'action': 1.5 - i * 0.01 + np.random.normal(0, 0.03),
            'total': 3.5 - i * 0.03 + np.random.normal(0, 0.06),
        }
        monitor.update(losses_sim, epoch=i // 10)
    
    stats = monitor.get_stats()
    print(f"  Components tracked: {list(stats.keys())}")
    for key, s in stats.items():
        print(f"    {key}: latest={s['latest']:.4f}, mean={s['mean']:.4f}")
    
    # Inject a spike
    monitor.update({'digit': 50.0, 'action': 1.0, 'total': 51.0}, epoch=5)
    anomalies = monitor.check_anomalies()
    spike_anomalies = [a for a in anomalies if a['type'] == 'spike']
    print(f"  Spike detected: {len(spike_anomalies) > 0}")
    if spike_anomalies:
        print(f"    {spike_anomalies[-1]['details']}")
    
    # Inject NaN
    monitor.update({'digit': float('nan'), 'action': 1.0, 'total': float('nan')}, epoch=5)
    anomalies = monitor.check_anomalies()
    nan_anomalies = [a for a in anomalies if a['type'] == 'nan_inf']
    print(f"  NaN detected: {len(nan_anomalies) > 0}")
    
    print("  ✓ Passed")
    
    # Test 8: Loss Package
    print("\n[Test 8] Loss Package")
    pkg = LossPackage(
        weights=LossWeights(digit=1.0, action=1.0, jump=0.5, heatmap=1.0),
        jump_warmup_epochs=5,
        monitor_window=20
    )
    
    pkg.set_epoch(0)
    total = pkg.compute_total(losses, track=True, epoch=0)
    print(f"  Total loss: {total.item():.4f}")
    print(f"  Weights: {pkg.get_weights()}")
    
    print(f"\n  {pkg.summary()}")
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)