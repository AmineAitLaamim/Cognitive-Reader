"""
train/trainer.py
Training loop for the Cognitive Reader project.

Orchestrates:
  - Backbone forward pass (batched, GPU-efficient)
  - Heatmap detection loss (Focal Loss)
  - DualModeController teacher-forced training (per-sample)
  - Gradient accumulation, clipping, and optimizer stepping
  - Validation and OOD evaluation
  - Checkpointing (best + every 10 epochs)
  - Metrics saving to JSON

Training Flow per Batch:
  1. Backbone: image → node_embeddings + cls_token + heatmap_logits
  2. Heatmap Loss: focal_loss(heatmap_logits, heatmap_targets)
  3. For each sample in batch:
     a. Unpad graph, attach node_embeddings
     b. Controller.forward_train(graph, gt_sequence, cls_token)
     c. Accumulate controller losses
  4. Total loss = heatmap_loss + avg_controller_loss
  5. Backprop + gradient clip + optimizer step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from models.backbone.cnn import VisualBackbone, HeatmapHead
from models.controller.dual_mode import DualModeController, DualModeOutput
from data.collate import BatchedSample, unpad_graph
from data.dataset import (
    DatasetConfig, CognitiveReaderDataset, OODEvalDataset,
    create_dataloaders, create_ood_dataloaders
)
from utils.checkpoint import CheckpointManager
from utils.metrics_saver import MetricsSaver


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    # === Optimization ===
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # === Schedule ===
    num_epochs: int = 100
    warmup_epochs: int = 5
    scheduler: str = 'cosine'
    step_size: int = 30
    gamma: float = 0.1

    # === Data ===
    batch_size: int = 8
    num_workers: int = 4

    # === Loss weights ===
    heatmap_loss_weight: float = 1.0
    digit_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    jump_loss_weight: float = 1.0

    # === Evaluation ===
    val_every_n_epochs: int = 1
    ood_eval_every_n_epochs: int = 10
    ood_eval_lengths: List[int] = field(default_factory=lambda: [100, 200])
    ood_eval_samples: int = 20

    # === Checkpointing ===
    checkpoint_dir: str = './checkpoints'
    save_every_n_epochs: int = 10
    save_best: bool = True
    keep_last_n_checkpoints: int = 3

    # === Metrics ===
    metrics_dir: str = './metrics'

    # === Logging ===
    log_every_n_steps: int = 50
    log_dir: str = './logs'

    # === Mixed precision ===
    use_amp: bool = False

    # === Device ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === Reproducibility ===
    seed: int = 42


class Trainer:
    """
    Complete training orchestrator for the Cognitive Reader.

    Usage:
        trainer = Trainer(trainer_config, dataset_config)
        trainer.fit()
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        dataset_config: DatasetConfig
    ):
        self.tcfg = trainer_config
        self.dcfg = dataset_config
        self.device = torch.device(trainer_config.device)

        # Set global seed
        torch.manual_seed(trainer_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trainer_config.seed)

        # ============================================================
        # Initialize Models
        # ============================================================
        self.backbone = VisualBackbone(
            vis_dim=512,
            roi_output_size=7,
            pretrained=True,
            enable_heatmap=True,
            padding_factor=1.2
        ).to(self.device)

        self.controller = DualModeController(
            vis_dim=512,
            hidden_dim=256,
            edge_dim=256,
            key_dim=256,
            num_classes=10,
            radius=dataset_config.threshold_radius_r,
            T_intra=0.8 * dataset_config.threshold_radius_r + 4 * dataset_config.noise_sigma,
            T_inter=1.5 * dataset_config.threshold_radius_r - 4 * dataset_config.noise_sigma,
            num_frequencies=64,
            num_heads=4,
            dropout=0.1,
            loss_weights={
                'digit': trainer_config.digit_loss_weight,
                'action': trainer_config.action_loss_weight,
                'jump': trainer_config.jump_loss_weight
            }
        ).to(self.device)

        # ============================================================
        # Optimizer
        # ============================================================
        self.optimizer = torch.optim.AdamW([
            {
                'params': self.backbone.parameters(),
                'lr': trainer_config.backbone_lr,
                'weight_decay': trainer_config.weight_decay
            },
            {
                'params': self.controller.parameters(),
                'lr': trainer_config.learning_rate,
                'weight_decay': trainer_config.weight_decay
            }
        ])

        # ============================================================
        # Scheduler
        # ============================================================
        self.scheduler = self._build_scheduler()

        # ============================================================
        # Mixed Precision
        # ============================================================
        self.scaler = GradScaler(enabled=trainer_config.use_amp)

        # ============================================================
        # Data
        # ============================================================
        self.train_loader, self.val_loader = create_dataloaders(
            dataset_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers
        )

        # ============================================================
        # Checkpoint Manager
        # ============================================================
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=trainer_config.checkpoint_dir,
            keep_last_n=trainer_config.keep_last_n_checkpoints,
            save_best=trainer_config.save_best,
        )

        # ============================================================
        # Metrics Saver
        # ============================================================
        self.metrics_saver = MetricsSaver(
            metrics_dir=trainer_config.metrics_dir
        )

        # ============================================================
        # State
        # ============================================================
        self.current_epoch = 0
        self.global_step = 0
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

        # Create directories
        os.makedirs(trainer_config.checkpoint_dir, exist_ok=True)
        os.makedirs(trainer_config.metrics_dir, exist_ok=True)
        os.makedirs(trainer_config.log_dir, exist_ok=True)

        print(f"[Trainer] Initialized on {self.device}")
        print(f"[Trainer] Backbone params: {sum(p.numel() for p in self.backbone.parameters()):,}")
        print(f"[Trainer] Controller params: {sum(p.numel() for p in self.controller.parameters()):,}")
        print(f"[Trainer] Checkpoints: {trainer_config.checkpoint_dir}")
        print(f"[Trainer] Metrics: {trainer_config.metrics_dir}")

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.tcfg.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.tcfg.num_epochs - self.tcfg.warmup_epochs,
                eta_min=1e-7
            )
        elif self.tcfg.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.tcfg.step_size,
                gamma=self.tcfg.gamma
            )
        else:
            return None

    # ==============================================================
    # MAIN TRAINING LOOP
    # ==============================================================

    def fit(self) -> None:
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.tcfg.num_epochs} epochs")
        print(f"  Batch size: {self.tcfg.batch_size}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, self.tcfg.num_epochs):
            self.current_epoch = epoch

            # ----------------------------------------------------------
            # 1. Train one epoch
            # ----------------------------------------------------------
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            self.metrics_saver.append_train(train_metrics, epoch)

            # ----------------------------------------------------------
            # 2. Validate
            # ----------------------------------------------------------
            if (epoch + 1) % self.tcfg.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch)
                self.val_history.append(val_metrics)
                self.metrics_saver.append_val(val_metrics, epoch)

                # Save best checkpoint + best metrics
                is_best = self.ckpt_mgr.save_if_best(
                    val_loss=val_metrics['total'],
                    epoch=epoch,
                    global_step=self.global_step,
                    backbone=self.backbone,
                    controller=self.controller,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_history=self.train_history,
                    val_history=self.val_history,
                    trainer_config=self.tcfg,
                    dataset_config=self.dcfg,
                )
                if is_best:
                    self.metrics_saver.save_best(val_metrics, epoch)

            # ----------------------------------------------------------
            # 3. Periodic checkpoint (every N epochs)
            # ----------------------------------------------------------
            self.ckpt_mgr.save_periodic(
                epoch=epoch,
                save_every=self.tcfg.save_every_n_epochs,
                global_step=self.global_step,
                backbone=self.backbone,
                controller=self.controller,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_history=self.train_history,
                val_history=self.val_history,
                trainer_config=self.tcfg,
                dataset_config=self.dcfg,
            )

            # ----------------------------------------------------------
            # 4. OOD Evaluation
            # ----------------------------------------------------------
            if (epoch + 1) % self.tcfg.ood_eval_every_n_epochs == 0:
                ood_results = self.evaluate_ood(epoch)
                self.metrics_saver.save_ood(ood_results, epoch)

            # ----------------------------------------------------------
            # 5. Step scheduler
            # ----------------------------------------------------------
            if self.scheduler is not None and epoch >= self.tcfg.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[1]['lr']
            print(f"  LR: {current_lr:.2e}")

        # ----------------------------------------------------------
        # Final summary
        # ----------------------------------------------------------
        self.metrics_saver.save_summary({
            'num_epochs': self.tcfg.num_epochs,
            'best_val_loss': self.ckpt_mgr.best_val_loss,
            'best_epoch': self.ckpt_mgr.best_epoch,
            'final_train_loss': self.train_history[-1] if self.train_history else {},
            'config': {
                'learning_rate': self.tcfg.learning_rate,
                'backbone_lr': self.tcfg.backbone_lr,
                'batch_size': self.tcfg.batch_size,
                'num_epochs': self.tcfg.num_epochs,
                'radius': self.dcfg.threshold_radius_r,
                'max_digits': self.dcfg.max_digits,
                'noise_sigma': self.dcfg.noise_sigma,
                'max_chunk_size': self.dcfg.max_chunk_size,
            },
        })

        print(f"\n{'='*60}")
        print(f"  Training complete.")
        print(f"  Best val loss: {self.ckpt_mgr.best_val_loss:.4f} (epoch {self.ckpt_mgr.best_epoch + 1})")
        print(f"  Checkpoints: {self.tcfg.checkpoint_dir}/")
        print(f"  Metrics: {self.tcfg.metrics_dir}/")
        print(f"{'='*60}")

    # ==============================================================
    # TRAIN EPOCH
    # ==============================================================

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.backbone.train()
        self.controller.train()

        epoch_losses = {
            'total': 0.0,
            'heatmap': 0.0,
            'digit': 0.0,
            'action': 0.0,
            'jump': 0.0
        }
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += metrics.get(key, 0.0)
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.tcfg.log_every_n_steps == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"  [Epoch {epoch+1} | Step {self.global_step}] "
                    f"loss={metrics['total']:.4f} "
                    f"hm={metrics['heatmap']:.4f} "
                    f"dig={metrics['digit']:.4f} "
                    f"act={metrics['action']:.4f} "
                    f"jmp={metrics['jump']:.4f} "
                    f"({elapsed:.1f}s)"
                )

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        elapsed = time.time() - epoch_start
        print(
            f"\n  Epoch {epoch+1}/{self.tcfg.num_epochs} — "
            f"loss={epoch_losses['total']:.4f} "
            f"hm={epoch_losses['heatmap']:.4f} "
            f"dig={epoch_losses['digit']:.4f} "
            f"act={epoch_losses['action']:.4f} "
            f"jmp={epoch_losses['jump']:.4f} "
            f"({elapsed:.1f}s)\n"
        )

        return epoch_losses

    # ==============================================================
    # TRAIN STEP
    # ==============================================================

    def train_step(self, batch: BatchedSample) -> Dict[str, float]:
        """
        Single training step on one batch.

        1. Backbone forward (batched).
        2. Heatmap loss.
        3. Controller forward (per-sample, teacher forcing).
        4. Backprop + clip + step.
        """
        self.optimizer.zero_grad()

        images = batch.images.to(self.device)
        boxes = batch.boxes.to(self.device)
        box_batch_indices = batch.box_batch_indices.to(self.device)
        heatmap_targets = batch.heatmap_targets.to(self.device)

        with autocast(enabled=self.tcfg.use_amp):
            # 1. Backbone
            backbone_out = self.backbone(images, boxes, box_batch_indices)
            node_embeddings_all = backbone_out['node_embeddings']
            cls_tokens = backbone_out['cls_token']
            heatmap_logits = backbone_out.get('heatmap_logits')

            # 2. Heatmap loss
            heatmap_loss = torch.tensor(0.0, device=self.device)
            if heatmap_logits is not None:
                heatmap_loss = HeatmapHead.focal_loss(heatmap_logits, heatmap_targets)

            # 3. Controller (per-sample)
            total_digit_loss = torch.tensor(0.0, device=self.device)
            total_action_loss = torch.tensor(0.0, device=self.device)
            total_jump_loss = torch.tensor(0.0, device=self.device)

            B = batch.batch_size

            for i in range(B):
                sample_data = unpad_graph(batch, i, device=self.device)
                graph = sample_data['graph']
                gt_sequence = sample_data['gt_sequence']

                box_mask = (box_batch_indices == i)
                graph.node_embeddings = node_embeddings_all[box_mask]
                cls_token_i = cls_tokens[i]

                controller_out = self.controller.forward_train(
                    graph=graph,
                    gt_sequence=gt_sequence,
                    cls_token=cls_token_i,
                    device=self.device
                )

                if controller_out.digit_loss is not None:
                    total_digit_loss = total_digit_loss + controller_out.digit_loss
                if controller_out.action_loss is not None:
                    total_action_loss = total_action_loss + controller_out.action_loss
                if controller_out.jump_loss is not None:
                    total_jump_loss = total_jump_loss + controller_out.jump_loss

            avg_digit_loss = total_digit_loss / B
            avg_action_loss = total_action_loss / B
            avg_jump_loss = total_jump_loss / B

            # 4. Total loss
            total_loss = (
                self.tcfg.heatmap_loss_weight * heatmap_loss +
                self.tcfg.digit_loss_weight * avg_digit_loss +
                self.tcfg.action_loss_weight * avg_action_loss +
                self.tcfg.jump_loss_weight * avg_jump_loss
            )
            total_loss = total_loss / self.tcfg.gradient_accumulation_steps

        # 5. Backprop
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        all_params = list(self.backbone.parameters()) + list(self.controller.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.tcfg.max_grad_norm)

        if (self.global_step + 1) % self.tcfg.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {
            'total': total_loss.item() * self.tcfg.gradient_accumulation_steps,
            'heatmap': heatmap_loss.item(),
            'digit': avg_digit_loss.item(),
            'action': avg_action_loss.item(),
            'jump': avg_jump_loss.item()
        }

    # ==============================================================
    # VALIDATION
    # ==============================================================

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop."""
        self.backbone.eval()
        self.controller.eval()

        val_losses = {
            'total': 0.0,
            'heatmap': 0.0,
            'digit': 0.0,
            'action': 0.0,
            'jump': 0.0
        }
        num_batches = 0

        for batch in self.val_loader:
            images = batch.images.to(self.device)
            boxes = batch.boxes.to(self.device)
            box_batch_indices = batch.box_batch_indices.to(self.device)
            heatmap_targets = batch.heatmap_targets.to(self.device)

            backbone_out = self.backbone(images, boxes, box_batch_indices)
            node_embeddings_all = backbone_out['node_embeddings']
            cls_tokens = backbone_out['cls_token']
            heatmap_logits = backbone_out.get('heatmap_logits')

            heatmap_loss = torch.tensor(0.0, device=self.device)
            if heatmap_logits is not None:
                heatmap_loss = HeatmapHead.focal_loss(heatmap_logits, heatmap_targets)

            total_digit = torch.tensor(0.0, device=self.device)
            total_action = torch.tensor(0.0, device=self.device)
            total_jump = torch.tensor(0.0, device=self.device)
            B = batch.batch_size

            for i in range(B):
                sample_data = unpad_graph(batch, i, device=self.device)
                graph = sample_data['graph']
                box_mask = (box_batch_indices == i)
                graph.node_embeddings = node_embeddings_all[box_mask]
                cls_token_i = cls_tokens[i]

                out = self.controller.forward_train(
                    graph, sample_data['gt_sequence'], cls_token_i, self.device
                )

                if out.digit_loss is not None:
                    total_digit += out.digit_loss
                if out.action_loss is not None:
                    total_action += out.action_loss
                if out.jump_loss is not None:
                    total_jump += out.jump_loss

            batch_total = (
                self.tcfg.heatmap_loss_weight * heatmap_loss +
                self.tcfg.digit_loss_weight * (total_digit / B) +
                self.tcfg.action_loss_weight * (total_action / B) +
                self.tcfg.jump_loss_weight * (total_jump / B)
            )

            val_losses['total'] += batch_total.item()
            val_losses['heatmap'] += heatmap_loss.item()
            val_losses['digit'] += (total_digit / B).item()
            val_losses['action'] += (total_action / B).item()
            val_losses['jump'] += (total_jump / B).item()
            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        print(
            f"  [Val Epoch {epoch+1}] "
            f"loss={val_losses['total']:.4f} "
            f"hm={val_losses['heatmap']:.4f} "
            f"dig={val_losses['digit']:.4f} "
            f"act={val_losses['action']:.4f} "
            f"jmp={val_losses['jump']:.4f}"
        )

        return val_losses

    # ==============================================================
    # OOD EVALUATION
    # ==============================================================

    @torch.no_grad()
    def evaluate_ood(self, epoch: int) -> Dict[int, Dict]:
        """Evaluate on OOD datasets with fixed sequence lengths."""
        self.backbone.eval()
        self.controller.eval()

        ood_loaders = create_ood_dataloaders(
            self.dcfg,
            eval_lengths=self.tcfg.ood_eval_lengths,
            num_samples_per_length=self.tcfg.ood_eval_samples,
            batch_size=1
        )

        results = {}

        for length, loader in ood_loaders.items():
            correct_sequences = 0
            total_sequences = 0
            correct_digits = 0
            total_digits = 0
            avg_steps = 0

            for batch in loader:
                images = batch.images.to(self.device)
                boxes = batch.boxes.to(self.device)
                box_batch_indices = batch.box_batch_indices.to(self.device)

                backbone_out = self.backbone(images, boxes, box_batch_indices)
                node_embeddings_all = backbone_out['node_embeddings']
                cls_tokens = backbone_out['cls_token']

                sample_data = unpad_graph(batch, 0, device=self.device)
                graph = sample_data['graph']
                graph.node_embeddings = node_embeddings_all
                cls_token = cls_tokens[0]

                output = self.controller.forward_inference(
                    graph=graph,
                    cls_token=cls_token,
                    device=self.device,
                    max_steps=length * 3,
                    greedy=True
                )

                gt_tokens = [t['token'] for t in sample_data['gt_sequence']]
                pred_tokens = [t for t in output.predicted_sequence if t != '<END>']

                if pred_tokens == gt_tokens:
                    correct_sequences += 1
                total_sequences += 1

                gt_digits = [t for t in gt_tokens if t != '<CHUNK>']
                pred_digits = [t for t in pred_tokens if t != '<CHUNK>']
                for gt_d, pred_d in zip(gt_digits, pred_digits):
                    if gt_d == pred_d:
                        correct_digits += 1
                total_digits += len(gt_digits)

                avg_steps += output.num_steps

            seq_acc = correct_sequences / max(total_sequences, 1)
            digit_acc = correct_digits / max(total_digits, 1)
            avg_steps /= max(total_sequences, 1)

            results[length] = {
                'sequence_accuracy': seq_acc,
                'digit_accuracy': digit_acc,
                'avg_steps': avg_steps,
                'num_samples': total_sequences
            }

            print(
                f"  [OOD Length={length}] "
                f"seq_acc={seq_acc:.4f} "
                f"digit_acc={digit_acc:.4f} "
                f"avg_steps={avg_steps:.1f}"
            )

        return results

    # ==============================================================
    # RESUME
    # ==============================================================

    def resume_from_checkpoint(self, path: str) -> None:
        """Resume training from a saved checkpoint."""
        state = self.ckpt_mgr._load(path, self.device)
        self.ckpt_mgr.restore_model(
            state=state,
            backbone=self.backbone,
            controller=self.controller,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.current_epoch = state['epoch'] + 1
        self.global_step = state['global_step']
        self.train_history = state.get('train_history', [])
        self.val_history = state.get('val_history', [])

        # Reload metrics from disk
        self.metrics_saver._train_history = self.metrics_saver.load_train_history()
        self.metrics_saver._val_history = self.metrics_saver.load_val_history()
        self.metrics_saver._ood_history = self.metrics_saver.load_ood_results()

        print(f"  Resuming from epoch {self.current_epoch}")


# ==============================================================
# ENTRY POINT
# ==============================================================

def main():
    """Default training configuration."""

    dataset_config = DatasetConfig(
        min_digits=5,
        max_digits=50,
        img_width=640,
        img_height=640,
        threshold_radius_r=80.0,
        noise_sigma=3.0,
        max_chunk_size=4,
        min_chunk_size=1,
        samples_per_epoch=1000,
        seed=42
    )

    trainer_config = TrainerConfig(
        learning_rate=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        num_epochs=100,
        warmup_epochs=5,
        scheduler='cosine',
        batch_size=8,
        num_workers=4,
        heatmap_loss_weight=1.0,
        digit_loss_weight=1.0,
        action_loss_weight=1.0,
        jump_loss_weight=1.0,
        val_every_n_epochs=1,
        ood_eval_every_n_epochs=10,
        ood_eval_lengths=[100, 200],
        ood_eval_samples=20,
        checkpoint_dir='./checkpoints',
        save_every_n_epochs=10,
        save_best=True,
        keep_last_n_checkpoints=3,
        metrics_dir='./metrics',
        log_every_n_steps=50,
        log_dir='./logs',
        use_amp=torch.cuda.is_available(),
        seed=42
    )

    trainer = Trainer(trainer_config, dataset_config)
    trainer.fit()


if __name__ == "__main__":
    main()