"""
utils/logger.py
Unified logging for the Cognitive Reader project.

Supports multiple backends simultaneously:
  1. Console — formatted print output.
  2. File — JSON lines log for offline analysis.
  3. TensorBoard — scalar plots, image grids, histograms.
  4. Weights & Biases — cloud dashboard (optional).

Usage:
    logger = TrainingLogger(config)
    
    logger.log_scalars({'loss': 0.5, 'lr': 1e-4}, step=100)
    logger.log_image('sample_0', image_tensor, step=100)
    logger.log_histogram('gradients', grad_tensor, step=100)
    logger.log_text('config', json.dumps(config_dict))
    logger.log_metrics(metrics_dict, step=100, prefix='val')
    
    logger.close()
"""

import os
import json
import time
import math
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import numpy as np

# Optional backends
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==============================================================
# CONFIGURATION
# ==============================================================

@dataclass
class LoggerConfig:
    """Configuration for the training logger."""
    # Project
    project_name: str = 'cognitive_reader'
    experiment_name: str = ''
    run_id: str = ''
    
    # Backends
    use_console: bool = True
    use_file: bool = True
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Paths
    log_dir: str = './logs'
    log_file_name: str = 'train_log.jsonl'
    
    # Console
    console_log_every: int = 50       # Print every N steps
    console_format: str = 'compact'   # 'compact' | 'detailed'
    
    # File
    file_flush_every: int = 10        # Flush to disk every N writes
    
    # Wandb (if enabled)
    wandb_project: str = 'cognitive_reader'
    wandb_entity: str = ''
    wandb_tags: List[str] = field(default_factory=list)
    
    # Image logging
    log_images_every: int = 500       # Log sample images every N steps
    max_image_samples: int = 4        # Max images per logging event
    
    # Metrics
    log_histograms: bool = False      # Log gradient/weight histograms


# ==============================================================
# CONSOLE LOGGER
# ==============================================================

class ConsoleLogger:
    """Formatted console output with timestamps and color."""
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'gray': '\033[90m',
    }
    
    def __init__(self, config: LoggerConfig):
        self.cfg = config
        self._step_count = 0
        self._epoch_start_time: Optional[float] = None
        self._train_start_time = time.time()
    
    def _timestamp(self) -> str:
        elapsed = time.time() - self._train_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _color(self, text: str, color: str) -> str:
        if color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text
    
    def log_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        lr: Optional[float] = None
    ) -> None:
        """Log a training step."""
        self._step_count = step
        
        if self._epoch_start_time is None:
            self._epoch_start_time = time.time()
        
        ts = self._timestamp()
        
        if self.cfg.console_format == 'compact':
            parts = [f"[{ts}] Ep {epoch+1} | Step {step}"]
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}")
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            print(" | ".join(parts))
        else:
            print(f"\n[{ts}] Epoch {epoch+1} | Step {step}")
            for k, v in metrics.items():
                print(f"  {k:20s}: {v:.6f}")
            if lr is not None:
                print(f"  {'learning_rate':20s}: {lr:.2e}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: Optional[float] = None
    ) -> None:
        """Log end of epoch summary."""
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1} Complete" + (f" ({epoch_time:.1f}s)" if epoch_time else ""))
        print(f"{'='*60}")
        
        print(f"  Train:")
        for k, v in train_metrics.items():
            print(f"    {k:25s}: {v:.6f}")
        
        if val_metrics:
            print(f"  Val:")
            for k, v in val_metrics.items():
                print(f"    {k:25s}: {v:.6f}")
        
        print(f"{'='*60}\n")
        
        self._epoch_start_time = None
    
    def log_info(self, message: str) -> None:
        """Log an informational message."""
        ts = self._timestamp()
        print(f"[{ts}] {self._color('INFO', 'cyan')} | {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        ts = self._timestamp()
        print(f"[{ts}] {self._color('WARN', 'yellow')} | {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        ts = self._timestamp()
        print(f"[{ts}] {self._color('ERROR', 'red')} | {message}")
    
    def log_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Log a detected anomaly."""
        ts = self._timestamp()
        atype = anomaly.get('type', 'unknown')
        details = anomaly.get('details', '')
        print(f"[{ts}] {self._color('ANOMALY', 'red')} | [{atype}] {details}")
    
    def log_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: str = ""
    ) -> None:
        """Log a formatted table."""
        if title:
            print(f"\n  {title}")
        
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(f"  {header_str}")
        print(f"  {'-' * len(header_str)}")
        
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(f"  {row_str}")
        print()


# ==============================================================
# FILE LOGGER
# ==============================================================

class FileLogger:
    """JSON Lines file logger for offline analysis."""
    
    def __init__(self, config: LoggerConfig):
        self.cfg = config
        self._buffer: List[Dict] = []
        self._write_count = 0
        
        os.makedirs(config.log_dir, exist_ok=True)
        self._path = os.path.join(config.log_dir, config.log_file_name)
        
        # Write session header
        self._write_entry({
            'type': 'session_start',
            'timestamp': datetime.now().isoformat(),
            'project': config.project_name,
            'experiment': config.experiment_name,
            'run_id': config.run_id,
        })
    
    def _write_entry(self, entry: Dict) -> None:
        """Write a single JSON line."""
        self._buffer.append(entry)
        self._write_count += 1
        
        if self._write_count % self.cfg.file_flush_every == 0:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffered entries to disk."""
        if not self._buffer:
            return
        
        with open(self._path, 'a') as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, default=str) + '\n')
        self._buffer.clear()
    
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ) -> None:
        """Log scalar metrics."""
        self._write_entry({
            'type': 'scalars',
            'step': step,
            'prefix': prefix,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self._write_entry({
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            **config
        })
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event."""
        self._write_entry({
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            **data
        })
    
    def close(self) -> None:
        """Flush and close."""
        self._write_entry({
            'type': 'session_end',
            'timestamp': datetime.now().isoformat(),
        })
        self.flush()


# ==============================================================
# TENSORBOARD LOGGER
# ==============================================================

class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, config: LoggerConfig):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        
        log_path = os.path.join(config.log_dir, 'tensorboard')
        os.makedirs(log_path, exist_ok=True)
        
        run_name = config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_path, run_name)
        )
    
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ) -> None:
        """Log scalar metrics."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}"
            self.writer.add_scalar(tag, value, step)
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int
    ) -> None:
        """
        Log an image.
        
        Args:
            tag: Image tag.
            image: [3, H, W] or [H, W, 3] tensor in [0, 1].
            step: Global step.
        """
        if image.dim() == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        self.writer.add_image(tag, image.clamp(0, 1), step)
    
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        nrow: int = 4
    ) -> None:
        """
        Log a grid of images.
        
        Args:
            tag: Grid tag.
            images: [N, 3, H, W] tensor.
            step: Global step.
            nrow: Number of images per row.
        """
        from torchvision.utils import make_grid
        grid = make_grid(images.clamp(0, 1), nrow=nrow)
        self.writer.add_image(tag, grid, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ) -> None:
        """Log a histogram of values."""
        self.writer.add_histogram(tag, values.detach().cpu(), step)
    
    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        self.writer.add_figure(tag, figure, step)
    
    def flush(self) -> None:
        self.writer.flush()
    
    def close(self) -> None:
        self.writer.close()


# ==============================================================
# WANDB LOGGER
# ==============================================================

class WandbLogger:
    """Weights & Biases logging wrapper."""
    
    def __init__(self, config: LoggerConfig):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb not available. Install with: pip install wandb"
            )
        
        run_name = config.experiment_name or config.run_id or 'unnamed'
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            name=run_name,
            tags=config.wandb_tags,
            dir=config.log_dir,
            resume='allow' if config.run_id else None,
            id=config.run_id or None,
        )
    
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ) -> None:
        """Log scalar metrics."""
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_dict['global_step'] = step
        wandb.log(log_dict, step=step)
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int
    ) -> None:
        """Log an image."""
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        
        img_np = image.detach().cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        wandb.log({tag: wandb.Image(img_np)}, step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ) -> None:
        """Log a histogram."""
        wandb.log({tag: wandb.Histogram(values.detach().cpu().numpy())}, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        wandb.config.update(config, allow_val_change=True)
    
    def log_table(
        self,
        tag: str,
        columns: List[str],
        data: List[List[Any]]
    ) -> None:
        """Log a table."""
        table = wandb.Table(columns=columns, data=data)
        wandb.log({tag: table})
    
    def close(self) -> None:
        wandb.finish()


# ==============================================================
# UNIFIED TRAINING LOGGER
# ==============================================================

class TrainingLogger:
    """
    Unified logger that dispatches to all enabled backends.
    
    Usage:
        logger = TrainingLogger(LoggerConfig(
            use_tensorboard=True,
            use_wandb=False,
            log_dir='./logs'
        ))
        
        # Log training step
        logger.log_scalars({'loss': 0.5, 'digit_loss': 0.3}, step=100)
        
        # Log with prefix
        logger.log_scalars({'accuracy': 0.95}, step=100, prefix='val')
        
        # Log image
        logger.log_image('sample', image_tensor, step=100)
        
        # Log config
        logger.log_config({'lr': 1e-4, 'batch_size': 8})
        
        # End
        logger.close()
    """
    
    def __init__(self, config: LoggerConfig):
        self.cfg = config
        
        # Generate run ID if not provided
        if not config.run_id:
            config.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize backends
        self._console: Optional[ConsoleLogger] = None
        self._file: Optional[FileLogger] = None
        self._tensorboard: Optional[TensorBoardLogger] = None
        self._wandb: Optional[WandbLogger] = None
        
        if config.use_console:
            self._console = ConsoleLogger(config)
        
        if config.use_file:
            self._file = FileLogger(config)
        
        if config.use_tensorboard:
            if TENSORBOARD_AVAILABLE:
                self._tensorboard = TensorBoardLogger(config)
            else:
                print("[Logger] TensorBoard not available. Skipping.")
        
        if config.use_wandb:
            if WANDB_AVAILABLE:
                self._wandb = WandbLogger(config)
            else:
                print("[Logger] wandb not available. Skipping.")
        
        # State
        self._global_step = 0
        self._current_epoch = 0
        
        self.info(f"Logger initialized. Run ID: {config.run_id}")
        self.info(f"Backends: console={config.use_console}, file={config.use_file}, "
                  f"tensorboard={self._tensorboard is not None}, wandb={self._wandb is not None}")
    
    # ----------------------------------------------------------
    # Scalar Logging
    # ----------------------------------------------------------
    
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = 'train'
    ) -> None:
        """
        Log scalar metrics to all backends.
        
        Args:
            metrics: Dict of metric_name → value.
            step: Global step. If None, uses internal counter.
            prefix: Prefix for metric tags (e.g., 'train', 'val', 'ood').
        """
        if step is not None:
            self._global_step = step
        
        # Filter NaN/Inf
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                clean_metrics[k] = v
        
        if self._file:
            self._file.log_scalars(clean_metrics, self._global_step, prefix)
        
        if self._tensorboard:
            self._tensorboard.log_scalars(clean_metrics, self._global_step, prefix)
        
        if self._wandb:
            self._wandb.log_scalars(clean_metrics, self._global_step, prefix)
    
    def log_train_step(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: int,
        lr: Optional[float] = None
    ) -> None:
        """
        Log a training step (console + backends).
        
        Args:
            metrics: Loss metrics.
            step: Global step.
            epoch: Current epoch.
            lr: Current learning rate.
        """
        self._global_step = step
        self._current_epoch = epoch
        
        if lr is not None:
            metrics = {**metrics, 'learning_rate': lr}
        
        # Console (respecting log frequency)
        if self._console and step % self.cfg.console_log_every == 0:
            self._console.log_step(step, epoch, metrics, lr)
        
        # Backends (every step)
        self.log_scalars(metrics, step, prefix='train')
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: Optional[float] = None
    ) -> None:
        """Log end-of-epoch summary."""
        if self._console:
            self._console.log_epoch(epoch, train_metrics, val_metrics, epoch_time)
        
        self.log_scalars(train_metrics, self._global_step, prefix='train_epoch')
        
        if val_metrics:
            self.log_scalars(val_metrics, self._global_step, prefix='val')
    
    # ----------------------------------------------------------
    # Image Logging
    # ----------------------------------------------------------
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: Optional[int] = None
    ) -> None:
        """
        Log a single image.
        
        Args:
            tag: Image tag.
            image: [3, H, W] or [H, W, 3] tensor.
            step: Global step.
        """
        s = step or self._global_step
        
        if self._tensorboard:
            self._tensorboard.log_image(tag, image, s)
        
        if self._wandb:
            self._wandb.log_image(tag, image, s)
    
    def log_image_grid(
        self,
        tag: str,
        images: torch.Tensor,
        step: Optional[int] = None,
        nrow: int = 4
    ) -> None:
        """Log a grid of images (TensorBoard only)."""
        s = step or self._global_step
        if self._tensorboard:
            self._tensorboard.log_images(tag, images, s, nrow)
    
    def should_log_images(self, step: int) -> bool:
        """Check if images should be logged at this step."""
        return step % self.cfg.log_images_every == 0
    
    # ----------------------------------------------------------
    # Histogram Logging
    # ----------------------------------------------------------
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None
    ) -> None:
        """Log a histogram of values (gradients, weights, activations)."""
        if not self.cfg.log_histograms:
            return
        
        s = step or self._global_step
        
        if self._tensorboard:
            self._tensorboard.log_histogram(tag, values, s)
        
        if self._wandb:
            self._wandb.log_histogram(tag, values, s)
    
    def log_gradient_histograms(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
        max_layers: int = 10
    ) -> None:
        """Log gradient histograms for model parameters."""
        if not self.cfg.log_histograms:
            return
        
        count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and count < max_layers:
                self.log_histogram(f"grad/{name}", param.grad, step)
                count += 1
    
    # ----------------------------------------------------------
    # Text and Config Logging
    # ----------------------------------------------------------
    
    def log_config(self, config: Union[Dict, Any]) -> None:
        """Log hyperparameters / configuration."""
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {'config': str(config)}
        
        if self._file:
            self._file.log_config(config_dict)
        
        if self._tensorboard:
            text = json.dumps(config_dict, indent=2, default=str)
            self._tensorboard.log_text('config', text)
        
        if self._wandb:
            self._wandb.log_config(config_dict)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text (TensorBoard only)."""
        s = step or self._global_step
        if self._tensorboard:
            self._tensorboard.log_text(tag, text, s)
    
    def log_table(
        self,
        tag: str,
        columns: List[str],
        data: List[List[Any]]
    ) -> None:
        """Log a table (wandb) or formatted text (console)."""
        if self._wandb:
            self._wandb.log_table(tag, columns, data)
        
        if self._console:
            self._console.log_table(columns, data, title=tag)
    
    # ----------------------------------------------------------
    # Messages
    # ----------------------------------------------------------
    
    def info(self, message: str) -> None:
        """Log info message."""
        if self._console:
            self._console.log_info(message)
        if self._file:
            self._file.log_event('info', {'message': message})
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if self._console:
            self._console.log_warning(message)
        if self._file:
            self._file.log_event('warning', {'message': message})
    
    def error(self, message: str) -> None:
        """Log error message."""
        if self._console:
            self._console.log_error(message)
        if self._file:
            self._file.log_event('error', {'message': message})
    
    def log_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Log a detected loss anomaly."""
        if self._console:
            self._console.log_anomaly(anomaly)
        if self._file:
            self._file.log_event('anomaly', anomaly)
    
    # ----------------------------------------------------------
    # OOD Results
    # ----------------------------------------------------------
    
    def log_ood_results(
        self,
        results_by_length: Dict[int, Dict[str, float]],
        step: Optional[int] = None
    ) -> None:
        """
        Log OOD evaluation results.
        
        Args:
            results_by_length: Dict mapping length → metrics.
            step: Global step.
        """
        s = step or self._global_step
        
        for length, metrics in results_by_length.items():
            prefixed = {f"len_{length}/{k}": v for k, v in metrics.items()}
            self.log_scalars(prefixed, s, prefix='ood')
        
        # Log table
        columns = ['Length'] + list(next(iter(results_by_length.values())).keys())
        data = []
        for length in sorted(results_by_length.keys()):
            row = [length] + [
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in results_by_length[length].values()
            ]
            data.append(row)
        
        self.log_table('OOD Results', columns, data)
    
    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------
    
    def flush(self) -> None:
        """Flush all backends."""
        if self._file:
            self._file.flush()
        if self._tensorboard:
            self._tensorboard.flush()
    
    def close(self) -> None:
        """Close all backends."""
        self.info("Logger closing.")
        
        if self._file:
            self._file.close()
        if self._tensorboard:
            self._tensorboard.close()
        if self._wandb:
            self._wandb.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  utils/logger.py — Unit Test")
    print("=" * 60)
    
    # Test 1: Console Logger
    print("\n[Test 1] Console Logger")
    config = LoggerConfig(
        use_console=True,
        use_file=True,
        use_tensorboard=False,
        use_wandb=False,
        log_dir='./test_logs',
        console_log_every=1
    )
    
    console = ConsoleLogger(config)
    console.log_info("Training started")
    console.log_step(100, epoch=5, metrics={'loss': 0.5432, 'digit': 0.3211}, lr=1e-4)
    console.log_warning("Loss spike detected")
    console.log_epoch(
        epoch=5,
        train_metrics={'loss': 0.5, 'digit': 0.3, 'action': 0.15},
        val_metrics={'loss': 0.6, 'digit': 0.35},
        epoch_time=42.5
    )
    console.log_table(
        headers=['Length', 'Seq Acc', 'Digit Acc'],
        rows=[[100, '0.85', '0.95'], [200, '0.60', '0.88'], [500, '0.20', '0.75']],
        title='OOD Results'
    )
    print("  ✓ Console logger passed")
    
    # Test 2: File Logger
    print("\n[Test 2] File Logger")
    file_logger = FileLogger(config)
    file_logger.log_scalars({'loss': 0.5, 'lr': 1e-4}, step=100)
    file_logger.log_config({'batch_size': 8, 'lr': 1e-4})
    file_logger.log_event('checkpoint', {'path': './ckpt.pt', 'epoch': 10})
    file_logger.close()
    
    log_path = os.path.join(config.log_dir, config.log_file_name)
    assert os.path.exists(log_path), "Log file not created"
    with open(log_path, 'r') as f:
        lines = f.readlines()
    print(f"  Log file: {log_path} ({len(lines)} entries)")
    assert len(lines) >= 4  # header + 3 entries + close
    print("  ✓ File logger passed")
    
    # Test 3: TrainingLogger (console + file only)
    print("\n[Test 3] TrainingLogger")
    logger = TrainingLogger(LoggerConfig(
        use_console=True,
        use_file=True,
        use_tensorboard=False,
        use_wandb=False,
        log_dir='./test_logs',
        console_log_every=1,
        log_histograms=False
    ))
    
    logger.log_config({'lr': 1e-4, 'batch_size': 8, 'epochs': 100})
    logger.info("Training started")
    
    # Simulate training steps
    for step in range(5):
        logger.log_train_step(
            metrics={'loss': 1.0 - step * 0.1, 'digit': 0.5 - step * 0.05},
            step=step,
            epoch=0,
            lr=1e-4
        )
    
    logger.log_epoch(
        epoch=0,
        train_metrics={'loss': 0.6, 'digit': 0.3},
        val_metrics={'loss': 0.7, 'digit': 0.35},
        epoch_time=30.0
    )
    
    # Log OOD results
    logger.log_ood_results({
        100: {'exact_match': 0.85, 'digit_accuracy': 0.95},
        200: {'exact_match': 0.60, 'digit_accuracy': 0.88},
    })
    
    logger.log_anomaly({
        'type': 'spike',
        'component': 'digit',
        'details': 'digit loss spiked to 5.0 (3.2x running mean)'
    })
    
    logger.close()
    print("  ✓ TrainingLogger passed")
    
    # Test 4: TensorBoard (if available)
    if TENSORBOARD_AVAILABLE:
        print("\n[Test 4] TensorBoard Logger")
        tb_config = LoggerConfig(
            use_console=False,
            use_file=False,
            use_tensorboard=True,
            use_wandb=False,
            log_dir='./test_logs'
        )
        tb_logger = TensorBoardLogger(tb_config)
        
        tb_logger.log_scalars({'loss': 0.5, 'acc': 0.9}, step=100)
        
        fake_image = torch.rand(3, 64, 64)
        tb_logger.log_image('test_image', fake_image, step=100)
        
        fake_hist = torch.randn(1000)
        tb_logger.log_histogram('test_hist', fake_hist, step=100)
        
        tb_logger.log_text('test_text', 'Hello World', step=100)
        
        tb_logger.close()
        print("  ✓ TensorBoard logger passed")
    else:
        print("\n[Test 4] TensorBoard not available. Skipping.")
    
    # Cleanup
    import shutil
    if os.path.exists('./test_logs'):
        shutil.rmtree('./test_logs')
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)