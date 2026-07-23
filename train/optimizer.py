"""
train/optimizer.py
Optimizer construction, learning rate scheduling, gradient management,
and Exponential Moving Average for the Cognitive Reader project.

Components:
  1. Parameter group extraction (backbone vs controller vs detector).
  2. Optimizer construction (AdamW with per-group LR and weight decay).
  3. Learning rate schedulers (cosine with warmup, step, constant).
  4. Gradient clipping with norm logging.
  5. Exponential Moving Average (EMA) of model weights.
  6. Layer freezing/unfreezing utilities.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field


# ==============================================================
# 1. PARAMETER GROUP EXTRACTION
# ==============================================================

@dataclass
class ParamGroupConfig:
    """Configuration for a single parameter group."""
    name: str
    lr: float
    weight_decay: float = 0.0
    freeze: bool = False


def get_parameter_groups(
    backbone: nn.Module,
    controller: nn.Module,
    backbone_lr: float = 1e-5,
    controller_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone_epochs: int = 0,
    no_decay_keywords: Tuple[str, ...] = ('bias', 'LayerNorm', 'layernorm', 'bn')
) -> List[Dict[str, Any]]:
    """
    Extract parameter groups with separate learning rates and weight decay.
    
    Strategy:
      - Backbone parameters: lower LR (pretrained, fine-tuning).
      - Controller parameters: higher LR (randomly initialized).
      - Bias and LayerNorm parameters: no weight decay (standard practice).
      - Optionally freeze backbone for the first N epochs.
    
    Args:
        backbone: Visual backbone module.
        controller: Dual-mode controller module.
        backbone_lr: Learning rate for backbone parameters.
        controller_lr: Learning rate for controller parameters.
        weight_decay: Weight decay for non-exempt parameters.
        freeze_backbone_epochs: If > 0, freeze backbone initially.
        no_decay_keywords: Parameter name keywords that exempt from weight decay.
    
    Returns:
        List of parameter group dicts for torch.optim.
    """
    groups = []
    
    # --- Backbone parameters ---
    backbone_decay = []
    backbone_no_decay = []
    
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)
    
    if backbone_decay:
        groups.append({
            'params': backbone_decay,
            'lr': backbone_lr,
            'weight_decay': weight_decay,
            'group_name': 'backbone_decay'
        })
    if backbone_no_decay:
        groups.append({
            'params': backbone_no_decay,
            'lr': backbone_lr,
            'weight_decay': 0.0,
            'group_name': 'backbone_no_decay'
        })
    
    # --- Controller parameters ---
    controller_decay = []
    controller_no_decay = []
    
    for name, param in controller.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            controller_no_decay.append(param)
        else:
            controller_decay.append(param)
    
    if controller_decay:
        groups.append({
            'params': controller_decay,
            'lr': controller_lr,
            'weight_decay': weight_decay,
            'group_name': 'controller_decay'
        })
    if controller_no_decay:
        groups.append({
            'params': controller_no_decay,
            'lr': controller_lr,
            'weight_decay': 0.0,
            'group_name': 'controller_no_decay'
        })
    
    return groups


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }


# ==============================================================
# 2. OPTIMIZER CONSTRUCTION
# ==============================================================

def build_optimizer(
    backbone: nn.Module,
    controller: nn.Module,
    backbone_lr: float = 1e-5,
    controller_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer_type: str = 'adamw'
) -> torch.optim.Optimizer:
    """
    Build the optimizer with per-group learning rates.
    
    Args:
        backbone: Visual backbone.
        controller: Dual-mode controller.
        backbone_lr: LR for backbone (pretrained).
        controller_lr: LR for controller (random init).
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon.
        optimizer_type: 'adamw' | 'adam' | 'sgd'.
    
    Returns:
        Configured optimizer.
    """
    param_groups = get_parameter_groups(
        backbone=backbone,
        controller=controller,
        backbone_lr=backbone_lr,
        controller_lr=controller_lr,
        weight_decay=weight_decay
    )
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Log parameter group summary
    print(f"[Optimizer] Type: {optimizer_type}")
    for group in optimizer.param_groups:
        name = group.get('group_name', 'unnamed')
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {name}: {num_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']:.2e}")
    
    return optimizer


# ==============================================================
# 3. LEARNING RATE SCHEDULERS
# ==============================================================

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing.
    
    LR schedule:
      - Epochs [0, warmup_epochs): linear ramp from lr_init to lr_peak.
      - Epochs [warmup_epochs, max_epochs): cosine decay from lr_peak to lr_min.
    
    Applied per parameter group (each group has its own lr_peak).
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        lr_init_factor: float = 0.01,
        lr_min_factor: float = 0.01,
        steps_per_epoch: int = 1
    ):
        """
        Args:
            optimizer: The optimizer to schedule.
            warmup_epochs: Number of warmup epochs.
            max_epochs: Total number of training epochs.
            lr_init_factor: Initial LR as a fraction of peak LR.
            lr_min_factor: Minimum LR as a fraction of peak LR.
            steps_per_epoch: Number of optimizer steps per epoch
                            (for step-level scheduling).
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.max_steps = max_epochs * steps_per_epoch
        self.lr_init_factor = lr_init_factor
        self.lr_min_factor = lr_min_factor
        
        # Store peak LRs for each parameter group
        self.peak_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self._step_count = 0
    
    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._step_count += 1
        self._update_lrs()
    
    def _update_lrs(self) -> None:
        """Compute and set the current LR for each parameter group."""
        for i, group in enumerate(self.optimizer.param_groups):
            peak_lr = self.peak_lrs[i]
            lr = self._compute_lr(peak_lr)
            group['lr'] = lr
    
    def _compute_lr(self, peak_lr: float) -> float:
        """Compute the current LR given the peak LR."""
        t = self._step_count
        
        if t < self.warmup_steps:
            # Linear warmup
            lr_init = peak_lr * self.lr_init_factor
            progress = t / max(self.warmup_steps, 1)
            return lr_init + (peak_lr - lr_init) * progress
        else:
            # Cosine annealing
            lr_min = peak_lr * self.lr_min_factor
            progress = (t - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            return lr_min + 0.5 * (peak_lr - lr_min) * (1 + math.cos(math.pi * progress))
    
    def get_last_lr(self) -> List[float]:
        """Return the current LR for each parameter group."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        """Save scheduler state for checkpointing."""
        return {
            'step_count': self._step_count,
            'peak_lrs': self.peak_lrs,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load scheduler state from checkpoint."""
        self._step_count = state['step_count']
        self.peak_lrs = state['peak_lrs']
        self.warmup_steps = state['warmup_steps']
        self.max_steps = state['max_steps']


class WarmupStepScheduler:
    """
    Linear warmup followed by step decay.
    
    LR drops by gamma every step_size epochs after warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        step_size: int,
        gamma: float = 0.1,
        lr_init_factor: float = 0.01,
        steps_per_epoch: int = 1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.step_size_steps = step_size * steps_per_epoch
        self.gamma = gamma
        self.lr_init_factor = lr_init_factor
        self.peak_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step_count = 0
    
    def step(self) -> None:
        self._step_count += 1
        for i, group in enumerate(self.optimizer.param_groups):
            peak_lr = self.peak_lrs[i]
            
            if self._step_count < self.warmup_steps:
                lr_init = peak_lr * self.lr_init_factor
                progress = self._step_count / max(self.warmup_steps, 1)
                group['lr'] = lr_init + (peak_lr - lr_init) * progress
            else:
                steps_after_warmup = self._step_count - self.warmup_steps
                num_decays = steps_after_warmup // self.step_size_steps
                group['lr'] = peak_lr * (self.gamma ** num_decays)
    
    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        return {'step_count': self._step_count, 'peak_lrs': self.peak_lrs}
    
    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state['step_count']
        self.peak_lrs = state['peak_lrs']


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    warmup_epochs: int = 5,
    max_epochs: int = 100,
    step_size: int = 30,
    gamma: float = 0.1,
    steps_per_epoch: int = 1
) -> Any:
    """
    Build a learning rate scheduler.
    
    Args:
        optimizer: The optimizer.
        scheduler_type: 'cosine' | 'step' | 'constant' | 'none'.
        warmup_epochs: Warmup duration.
        max_epochs: Total epochs.
        step_size: For StepScheduler.
        gamma: For StepScheduler.
        steps_per_epoch: Steps per epoch (for step-level scheduling).
    
    Returns:
        Scheduler object with .step() and .get_last_lr() methods.
        Returns None if scheduler_type == 'none'.
    """
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch
        )
        print(f"[Scheduler] Warmup Cosine: warmup={warmup_epochs}ep, max={max_epochs}ep")
        return scheduler
    
    elif scheduler_type == 'step':
        scheduler = WarmupStepScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            step_size=step_size,
            gamma=gamma,
            steps_per_epoch=steps_per_epoch
        )
        print(f"[Scheduler] Warmup Step: warmup={warmup_epochs}ep, step={step_size}ep, gamma={gamma}")
        return scheduler
    
    elif scheduler_type == 'constant':
        # Constant LR with warmup only
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs * 10,  # Very long cosine → effectively constant
            lr_min_factor=0.99,          # Barely decays
            steps_per_epoch=steps_per_epoch
        )
        print(f"[Scheduler] Constant with warmup: warmup={warmup_epochs}ep")
        return scheduler
    
    elif scheduler_type == 'none':
        print("[Scheduler] None (fixed LR)")
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ==============================================================
# 4. GRADIENT CLIPPING
# ==============================================================

class GradientClipper:
    """
    Gradient clipping with norm tracking and logging.
    
    Supports:
      - Global norm clipping (default).
      - Per-parameter-group clipping.
      - Gradient norm logging for monitoring.
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        clip_per_group: bool = False
    ):
        """
        Args:
            max_norm: Maximum gradient norm.
            norm_type: Type of norm (2.0 = L2, 1.0 = L1).
            clip_per_group: If True, clip each parameter group independently.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_per_group = clip_per_group
        
        # Tracking
        self._grad_norms: List[float] = []
        self._clip_counts: int = 0
        self._total_steps: int = 0
    
    def clip(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Optional[Iterator[nn.Parameter]] = None
    ) -> float:
        """
        Clip gradients and return the pre-clip gradient norm.
        
        Args:
            optimizer: The optimizer (for per-group clipping).
            parameters: Iterable of parameters to clip. If None, clips
                       all parameters in the optimizer.
        
        Returns:
            Pre-clip total gradient norm.
        """
        self._total_steps += 1
        
        if self.clip_per_group:
            total_norm = 0.0
            for group in optimizer.param_groups:
                group_params = [p for p in group['params'] if p.grad is not None]
                if group_params:
                    group_norm = torch.nn.utils.clip_grad_norm_(
                        group_params, self.max_norm, self.norm_type
                    )
                    total_norm += group_norm.item() ** self.norm_type
            total_norm = total_norm ** (1.0 / self.norm_type)
        else:
            if parameters is None:
                parameters = []
                for group in optimizer.param_groups:
                    parameters.extend(group['params'])
            
            params_with_grad = [p for p in parameters if p.grad is not None]
            if params_with_grad:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    params_with_grad, self.max_norm, self.norm_type
                ).item()
            else:
                total_norm = 0.0
        
        self._grad_norms.append(total_norm)
        if total_norm > self.max_norm:
            self._clip_counts += 1
        
        return total_norm
    
    def get_stats(self) -> Dict[str, float]:
        """Return gradient clipping statistics."""
        if not self._grad_norms:
            return {'mean_norm': 0.0, 'max_norm': 0.0, 'clip_rate': 0.0}
        
        return {
            'mean_norm': float(sum(self._grad_norms) / len(self._grad_norms)),
            'max_norm': float(max(self._grad_norms)),
            'min_norm': float(min(self._grad_norms)),
            'clip_rate': self._clip_counts / max(self._total_steps, 1),
            'total_steps': self._total_steps,
        }
    
    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        self._grad_norms.clear()
        self._clip_counts = 0
        self._total_steps = 0


# ==============================================================
# 5. EXPONENTIAL MOVING AVERAGE (EMA)
# ==============================================================

class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of the model weights that is updated as:
        shadow = decay * shadow + (1 - decay) * current
    
    The shadow weights are used for evaluation (more stable than
    the raw training weights).
    
    Usage:
        ema = EMAModel(model, decay=0.999)
        
        # During training:
        ema.update(model)
        
        # During evaluation:
        ema.apply_shadow(model)    # Copy shadow → model
        evaluate(model)
        ema.restore(model)         # Copy original → model
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 1000
    ):
        """
        Args:
            model: The model to track.
            decay: EMA decay factor. Higher = slower tracking.
            warmup_steps: Number of steps before EMA starts.
                         During warmup, decay ramps from 0 to target.
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step_count = 0
        
        # Create shadow parameters
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def _get_decay(self) -> float:
        """Get the current decay factor (with warmup ramp)."""
        if self._step_count < self.warmup_steps:
            return min(
                self.decay,
                (1 + self._step_count) / (10 + self._step_count)
            )
        return self.decay
    
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        self._step_count += 1
        decay = self._get_decay()
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    param.data, alpha=1.0 - decay
                )
    
    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Copy shadow parameters into the model (for evaluation)."""
        self.backup.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore original parameters after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
    
    def state_dict(self) -> Dict:
        """Save EMA state for checkpointing."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'step_count': self._step_count,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state: Dict, device: torch.device = torch.device('cpu')) -> None:
        """Load EMA state from checkpoint."""
        self.shadow = {k: v.to(device) for k, v in state['shadow'].items()}
        self._step_count = state['step_count']
        self.decay = state['decay']


# ==============================================================
# 6. LAYER FREEZING UTILITIES
# ==============================================================

def freeze_module(module: nn.Module) -> int:
    """
    Freeze all parameters in a module.
    
    Returns:
        Number of frozen parameters.
    """
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += param.numel()
    return count


def unfreeze_module(module: nn.Module) -> int:
    """
    Unfreeze all parameters in a module.
    
    Returns:
        Number of unfrozen parameters.
    """
    count = 0
    for param in module.parameters():
        param.requires_grad = True
        count += param.numel()
    return count


def freeze_backbone_partial(
    backbone: nn.Module,
    freeze_layers: List[str] = None
) -> Dict[str, int]:
    """
    Selectively freeze specific layers of the backbone.
    
    Default: freeze conv1, bn1, layer1, layer2 (early layers).
    Keep layer3, layer4 trainable (task-specific features).
    
    Args:
        backbone: The visual backbone.
        freeze_layers: List of layer name prefixes to freeze.
                      Default: ['conv1', 'bn1', 'layer1', 'layer2']
    
    Returns:
        Dict with frozen and trainable parameter counts.
    """
    if freeze_layers is None:
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in backbone.named_parameters():
        should_freeze = any(name.startswith(prefix) for prefix in freeze_layers)
        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()
    
    print(f"[Freeze] Backbone: {frozen_count:,} frozen, {trainable_count:,} trainable")
    return {'frozen': frozen_count, 'trainable': trainable_count}


class GradualUnfreezer:
    """
    Gradually unfreeze backbone layers during training.
    
    Schedule:
      - Epochs [0, phase1): freeze all backbone.
      - Epochs [phase1, phase2): unfreeze layer3 + layer4.
      - Epochs [phase2, ∞): unfreeze all.
    
    This prevents the pretrained features from being destroyed
    by large gradients from the randomly initialized controller
    in the early training phase.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        phase1_epoch: int = 5,
        phase2_epoch: int = 15
    ):
        self.backbone = backbone
        self.phase1_epoch = phase1_epoch
        self.phase2_epoch = phase2_epoch
        self._current_phase = -1
        
        # Start fully frozen
        freeze_module(backbone)
        print(f"[Unfreezer] Backbone fully frozen. "
              f"Phase 1 at epoch {phase1_epoch}, Phase 2 at epoch {phase2_epoch}")
    
    def step(self, epoch: int) -> None:
        """Check if we should unfreeze at this epoch."""
        if epoch >= self.phase2_epoch and self._current_phase < 2:
            unfreeze_module(self.backbone)
            self._current_phase = 2
            print(f"[Unfreezer] Epoch {epoch}: ALL backbone layers unfrozen")
        
        elif epoch >= self.phase1_epoch and self._current_phase < 1:
            # Unfreeze layer3 and layer4 only
            for name, param in self.backbone.named_parameters():
                if name.startswith('layer3') or name.startswith('layer4'):
                    param.requires_grad = True
            self._current_phase = 1
            trainable = sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            )
            print(f"[Unfreezer] Epoch {epoch}: layer3+layer4 unfrozen "
                  f"({trainable:,} trainable params)")


# ==============================================================
# 7. COMPLETE OPTIMIZER PACKAGE
# ==============================================================

class OptimizerPackage:
    """
    Bundles optimizer, scheduler, gradient clipper, EMA, and unfreezer
    into a single object for clean training loop integration.
    
    Usage:
        pkg = OptimizerPackage(backbone, controller, config)
        
        for epoch in range(num_epochs):
            pkg.unfreezer.step(epoch)
            
            for batch in dataloader:
                loss = train_step(batch)
                pkg.optimizer_step(loss)
            
            pkg.scheduler_step()
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        controller: nn.Module,
        backbone_lr: float = 1e-5,
        controller_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        steps_per_epoch: int = 100,
        max_grad_norm: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_gradual_unfreeze: bool = True,
        unfreeze_phase1: int = 5,
        unfreeze_phase2: int = 15,
        freeze_backbone_partial_layers: bool = False
    ):
        # Build optimizer
        self.optimizer = build_optimizer(
            backbone=backbone,
            controller=controller,
            backbone_lr=backbone_lr,
            controller_lr=controller_lr,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type
        )
        
        # Build scheduler
        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        # Gradient clipper
        self.grad_clipper = GradientClipper(max_norm=max_grad_norm)
        
        # EMA
        self.ema_backbone = None
        self.ema_controller = None
        if use_ema:
            self.ema_backbone = EMAModel(backbone, decay=ema_decay)
            self.ema_controller = EMAModel(controller, decay=ema_decay)
            print(f"[OptimizerPackage] EMA enabled (decay={ema_decay})")
        
        # Gradual unfreezer
        self.unfreezer = None
        if use_gradual_unfreeze:
            self.unfreezer = GradualUnfreezer(
                backbone=backbone,
                phase1_epoch=unfreeze_phase1,
                phase2_epoch=unfreeze_phase2
            )
        
        # Partial freeze (alternative to gradual unfreeze)
        if freeze_backbone_partial_layers and not use_gradual_unfreeze:
            freeze_backbone_partial(backbone)
        
        # Store references
        self._backbone = backbone
        self._controller = controller
        self._accumulation_step = 0
    
    def optimizer_step(
        self,
        loss: torch.Tensor,
        scaler: Optional[Any] = None,
        accumulation_steps: int = 1
    ) -> float:
        """
        Perform one optimizer step with gradient clipping.
        
        Handles mixed precision (scaler) and gradient accumulation.
        
        Args:
            loss: The computed loss tensor.
            scaler: GradScaler for mixed precision (optional).
            accumulation_steps: Number of steps to accumulate before updating.
        
        Returns:
            Pre-clip gradient norm.
        """
        self._accumulation_step += 1
        
        # Scale loss for accumulation
        scaled_loss = loss / accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Step (only after accumulation)
        if self._accumulation_step % accumulation_steps == 0:
            # Unscale for clipping
            if scaler is not None:
                scaler.unscale_(self.optimizer)
            
            # Clip gradients
            grad_norm = self.grad_clipper.clip(self.optimizer)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update EMA
            if self.ema_backbone is not None:
                self.ema_backbone.update(self._backbone)
            if self.ema_controller is not None:
                self.ema_controller.update(self._controller)
            
            return grad_norm
        
        return 0.0
    
    def scheduler_step(self) -> None:
        """Advance the LR scheduler by one step."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def apply_ema(self) -> None:
        """Apply EMA weights for evaluation."""
        if self.ema_backbone is not None:
            self.ema_backbone.apply_shadow(self._backbone)
        if self.ema_controller is not None:
            self.ema_controller.apply_shadow(self._controller)
    
    def restore_from_ema(self) -> None:
        """Restore training weights after evaluation."""
        if self.ema_backbone is not None:
            self.ema_backbone.restore(self._backbone)
        if self.ema_controller is not None:
            self.ema_controller.restore(self._controller)
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        """Save complete optimizer state."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'grad_clipper_stats': self.grad_clipper.get_stats(),
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        if self.ema_backbone is not None:
            state['ema_backbone'] = self.ema_backbone.state_dict()
        if self.ema_controller is not None:
            state['ema_controller'] = self.ema_controller.state_dict()
        return state
    
    def load_state_dict(self, state: Dict, device: torch.device = torch.device('cpu')) -> None:
        """Load complete optimizer state."""
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.ema_backbone is not None and 'ema_backbone' in state:
            self.ema_backbone.load_state_dict(state['ema_backbone'], device)
        if self.ema_controller is not None and 'ema_controller' in state:
            self.ema_controller.load_state_dict(state['ema_controller'], device)


if __name__ == "__main__":
    print("=" * 60)
    print("  train/optimizer.py — Unit Test")
    print("=" * 60)
    
    # Create dummy models
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
    )
    controller = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    
    # Test 1: Parameter groups
    print("\n[Test 1] Parameter groups")
    groups = get_parameter_groups(backbone, controller, backbone_lr=1e-5, controller_lr=1e-4)
    for g in groups:
        num = sum(p.numel() for p in g['params'])
        print(f"  {g['group_name']}: {num} params, lr={g['lr']}, wd={g['weight_decay']}")
    print("  ✓ Passed")
    
    # Test 2: Optimizer construction
    print("\n[Test 2] Optimizer construction")
    optimizer = build_optimizer(backbone, controller, optimizer_type='adamw')
    print(f"  Param groups: {len(optimizer.param_groups)}")
    print("  ✓ Passed")
    
    # Test 3: Warmup Cosine Scheduler
    print("\n[Test 3] Warmup Cosine Scheduler")
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=5, max_epochs=100, steps_per_epoch=10
    )
    
    lrs = []
    for step in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # Warmup: LR should increase
    assert lrs[10] > lrs[0], "LR should increase during warmup"
    # Peak: LR should be near peak around warmup end
    peak_idx = 50  # 5 epochs * 10 steps
    # Decay: LR should decrease after peak
    assert lrs[500] < lrs[peak_idx], "LR should decrease after warmup"
    print(f"  LR at step 0:   {lrs[0]:.2e}")
    print(f"  LR at step 50:  {lrs[peak_idx]:.2e}")
    print(f"  LR at step 500: {lrs[500]:.2e}")
    print(f"  LR at step 999: {lrs[999]:.2e}")
    print("  ✓ Passed")
    
    # Test 4: Gradient Clipper
    print("\n[Test 4] Gradient Clipper")
    clipper = GradientClipper(max_norm=1.0)
    
    # Create fake gradients
    x = torch.randn(10, requires_grad=True)
    loss = (x * 100).sum()  # Large gradients
    loss.backward()
    
    # Wrap in a fake optimizer
    fake_opt = torch.optim.SGD([x], lr=0.01)
    norm = clipper.clip(fake_opt, parameters=[x])
    print(f"  Pre-clip norm: {norm:.2f}")
    print(f"  Post-clip norm: {x.grad.norm():.2f}")
    assert x.grad.norm() <= 1.01, "Gradient should be clipped to max_norm"
    
    stats = clipper.get_stats()
    print(f"  Clip rate: {stats['clip_rate']:.2f}")
    print("  ✓ Passed")
    
    # Test 5: EMA
    print("\n[Test 5] Exponential Moving Average")
    model = nn.Linear(10, 5)
    ema = EMAModel(model, decay=0.99, warmup_steps=0)
    
    # Save original weights
    original_weight = model.weight.data.clone()
    
    # Simulate training steps
    for _ in range(10):
        model.weight.data += torch.randn_like(model.weight.data) * 0.1
        ema.update(model)
    
    # Shadow should be different from current
    assert not torch.equal(ema.shadow['weight'], model.weight.data)
    
    # Apply shadow
    ema.apply_shadow(model)
    shadow_weight = model.weight.data.clone()
    assert torch.equal(shadow_weight, ema.shadow['weight'])
    
    # Restore
    ema.restore(model)
    assert not torch.equal(model.weight.data, shadow_weight)
    print("  ✓ Passed")
    
    # Test 6: Layer freezing
    print("\n[Test 6] Layer freezing")
    frozen = freeze_module(backbone)
    assert all(not p.requires_grad for p in backbone.parameters())
    print(f"  Frozen: {frozen} params")
    
    unfrozen = unfreeze_module(backbone)
    assert all(p.requires_grad for p in backbone.parameters())
    print(f"  Unfrozen: {unfrozen} params")
    print("  ✓ Passed")
    
    # Test 7: OptimizerPackage
    print("\n[Test 7] OptimizerPackage")
    pkg = OptimizerPackage(
        backbone=backbone,
        controller=controller,
        backbone_lr=1e-5,
        controller_lr=1e-4,
        scheduler_type='cosine',
        warmup_epochs=2,
        max_epochs=10,
        steps_per_epoch=5,
        use_ema=True,
        use_gradual_unfreeze=False
    )
    
    print(f"  LR: {pkg.get_lr()}")
    
    # Simulate a training step
    dummy_input = torch.randn(1, 3, 8, 8)
    dummy_out = backbone(dummy_input)
    dummy_loss = dummy_out.sum()
    
    grad_norm = pkg.optimizer_step(dummy_loss, accumulation_steps=1)
    print(f"  Grad norm: {grad_norm:.4f}")
    
    pkg.scheduler_step()
    print(f"  LR after step: {pkg.get_lr()}")
    
    # Test EMA apply/restore
    pkg.apply_ema()
    pkg.restore_from_ema()
    print("  ✓ EMA apply/restore passed")
    
    # Test state dict
    state = pkg.state_dict()
    assert 'optimizer' in state
    print(f"  State dict keys: {list(state.keys())}")
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)