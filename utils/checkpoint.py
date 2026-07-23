"""
utils/checkpoint.py
Checkpoint management with disk space control.

Saves:
  - checkpoint_best.pt    — best validation loss (overwritten)
  - checkpoint_epoch_N.pt — periodic saves every 10 epochs (pruned to keep last K)

Each checkpoint contains:
  - backbone weights
  - controller weights
  - optimizer state
  - scheduler state
  - EMA weights (if enabled)
  - training history
  - config
"""

import os
import glob
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import asdict


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and pruning.
    
    Usage:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir='./checkpoints',
            keep_last_n=3,
        )
        
        # During training:
        ckpt_mgr.save_if_best(val_loss, epoch, ...)
        ckpt_mgr.save_periodic(epoch, save_every=10, ...)
        
        # Resume:
        state = ckpt_mgr.load_best()
        ckpt_mgr.restore_model(state, backbone, controller, optimizer)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        save_optimizer: bool = True,
        save_history: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints.
            keep_last_n: Keep only the last N epoch checkpoints.
                        0 = keep all. -1 = save only best.
            save_best: Save checkpoint_best.pt when val loss improves.
            save_optimizer: Include optimizer state (needed for resuming).
            save_history: Include training history (for plotting after resume).
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.save_optimizer = save_optimizer
        self.save_history = save_history
        
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ==============================================================
    # BUILD STATE
    # ==============================================================
    
    def build_state(
        self,
        epoch: int,
        global_step: int,
        backbone: nn.Module,
        controller: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_backbone: Optional[Any] = None,
        ema_controller: Optional[Any] = None,
        train_history: Optional[List] = None,
        val_history: Optional[List] = None,
        trainer_config: Optional[Any] = None,
        dataset_config: Optional[Any] = None,
        extra: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete checkpoint state dict.
        
        Args:
            epoch: Current epoch.
            global_step: Current global step.
            backbone: Backbone module.
            controller: Controller module.
            optimizer: Optimizer (optional).
            scheduler: LR scheduler (optional).
            ema_backbone: EMA of backbone (optional).
            ema_controller: EMA of controller (optional).
            train_history: Training loss history.
            val_history: Validation loss history.
            trainer_config: Trainer config dataclass.
            dataset_config: Dataset config dataclass.
            extra: Any additional data to save.
        
        Returns:
            Complete state dict ready for torch.save().
        """
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'backbone_state_dict': backbone.state_dict(),
            'controller_state_dict': controller.state_dict(),
        }
        
        if self.save_optimizer and optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None and hasattr(scheduler, 'state_dict'):
            state['scheduler_state_dict'] = scheduler.state_dict()
        
        if ema_backbone is not None and hasattr(ema_backbone, 'state_dict'):
            state['ema_backbone_state_dict'] = ema_backbone.state_dict()
        
        if ema_controller is not None and hasattr(ema_controller, 'state_dict'):
            state['ema_controller_state_dict'] = ema_controller.state_dict()
        
        if self.save_history:
            if train_history is not None:
                state['train_history'] = train_history
            if val_history is not None:
                state['val_history'] = val_history
        
        if trainer_config is not None:
            state['trainer_config'] = (
                asdict(trainer_config)
                if hasattr(trainer_config, '__dataclass_fields__')
                else trainer_config
            )
        
        if dataset_config is not None:
            state['dataset_config'] = (
                asdict(dataset_config)
                if hasattr(dataset_config, '__dataclass_fields__')
                else dataset_config
            )
        
        if extra is not None:
            state.update(extra)
        
        return state
    
    # ==============================================================
    # SAVE
    # ==============================================================
    
    def save(self, state: Dict[str, Any], tag: str) -> str:
        """
        Save a checkpoint with the given tag.
        
        Args:
            state: Complete state dict.
            tag: Filename tag ('best', 'epoch_10', 'epoch_20', etc.).
        
        Returns:
            Path to saved file.
        """
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{tag}.pt')
        torch.save(state, path)
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [Checkpoint] Saved: checkpoint_{tag}.pt ({size_mb:.1f} MB)")
        
        return path
    
    def save_if_best(self, val_loss: float, epoch: int, **kwargs) -> bool:
        """
        Save checkpoint if val_loss is the best so far.
        
        Args:
            val_loss: Current validation loss.
            epoch: Current epoch.
            **kwargs: All arguments for build_state().
        
        Returns:
            True if a new best was saved, False otherwise.
        """
        if not self.save_best:
            return False
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
            state = self.build_state(epoch=epoch, **kwargs)
            self.save(state, 'best')
            
            print(f"  ★ New best val loss: {val_loss:.4f} (epoch {epoch + 1})")
            return True
        
        return False
    
    def save_periodic(
        self,
        epoch: int,
        save_every: int = 10,
        **kwargs
    ) -> Optional[str]:
        """
        Save a periodic checkpoint every N epochs.
        Automatically prunes old checkpoints beyond keep_last_n.
        
        Args:
            epoch: Current epoch (0-indexed).
            save_every: Save every N epochs.
            **kwargs: All arguments for build_state().
        
        Returns:
            Path if saved, None otherwise.
        """
        if (epoch + 1) % save_every != 0:
            return None
        
        state = self.build_state(epoch=epoch, **kwargs)
        path = self.save(state, f'epoch_{epoch + 1}')
        
        self._prune_old_checkpoints()
        
        return path
    
    def _prune_old_checkpoints(self) -> None:
        """
        Remove old epoch checkpoints, keeping only the last keep_last_n.
        Never removes 'best'.
        """
        if self.keep_last_n <= 0:
            return
        
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')
        epoch_files = glob.glob(pattern)
        
        if len(epoch_files) <= self.keep_last_n:
            return
        
        def _extract_epoch(path):
            basename = os.path.basename(path)
            try:
                return int(
                    basename
                    .replace('checkpoint_epoch_', '')
                    .replace('.pt', '')
                )
            except ValueError:
                return -1
        
        epoch_files.sort(key=_extract_epoch)
        
        num_to_remove = len(epoch_files) - self.keep_last_n
        for path in epoch_files[:num_to_remove]:
            os.remove(path)
            print(f"  [Checkpoint] Pruned: {os.path.basename(path)}")
    
    # ==============================================================
    # LOAD
    # ==============================================================
    
    def load_best(
        self, device: torch.device = torch.device('cpu')
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pt')
        if not os.path.exists(path):
            print(f"  [Checkpoint] No best checkpoint found")
            return None
        return self._load(path, device)
    
    def load_epoch(
        self, epoch: int, device: torch.device = torch.device('cpu')
    ) -> Optional[Dict[str, Any]]:
        """Load a specific epoch checkpoint."""
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        if not os.path.exists(path):
            print(f"  [Checkpoint] No checkpoint found for epoch {epoch}")
            return None
        return self._load(path, device)
    
    def _load(
        self, path: str, device: torch.device
    ) -> Dict[str, Any]:
        """Load a checkpoint from path."""
        state = torch.load(path, map_location=device)
        
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.best_epoch = state.get('best_epoch', -1)
        
        epoch = state.get('epoch', '?')
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [Checkpoint] Loaded: {os.path.basename(path)} "
              f"(epoch {epoch}, {size_mb:.1f} MB)")
        
        return state
    
    def restore_model(
        self,
        state: Dict[str, Any],
        backbone: nn.Module,
        controller: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Restore model weights from a checkpoint state.
        
        Args:
            state: Loaded checkpoint state dict.
            backbone: Backbone module to restore.
            controller: Controller module to restore.
            optimizer: Optimizer to restore (optional).
            scheduler: Scheduler to restore (optional).
        
        Returns:
            The epoch to resume from.
        """
        backbone.load_state_dict(state['backbone_state_dict'])
        controller.load_state_dict(state['controller_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in state:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(state['scheduler_state_dict'])
        
        epoch = state.get('epoch', 0)
        print(f"  [Checkpoint] Model restored (resume from epoch {epoch})")
        
        return epoch
    
    # ==============================================================
    # UTILITIES
    # ==============================================================
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.pt')
        files = glob.glob(pattern)
        
        results = []
        for path in sorted(files):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            basename = os.path.basename(path)
            
            try:
                state = torch.load(path, map_location='cpu')
                epoch = state.get('epoch', '?')
                val_loss = state.get('best_val_loss', '?')
            except Exception:
                epoch = '?'
                val_loss = '?'
            
            results.append({
                'file': basename,
                'path': path,
                'size_mb': size_mb,
                'epoch': epoch,
                'best_val_loss': val_loss,
            })
        
        return results
    
    def print_checkpoints(self) -> None:
        """Print a table of all checkpoints."""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            print("  No checkpoints found.")
            return
        
        print(f"\n  {'File':<30s} {'Size':>8s} {'Epoch':>8s} {'Best Val':>10s}")
        print(f"  {'─' * 60}")
        for ckpt in checkpoints:
            print(
                f"  {ckpt['file']:<30s} "
                f"{ckpt['size_mb']:>7.1f}M "
                f"{str(ckpt['epoch']):>8s} "
                f"{str(ckpt['best_val_loss']):>10s}"
            )
        
        total_mb = sum(c['size_mb'] for c in checkpoints)
        print(f"  {'─' * 60}")
        print(f"  Total: {total_mb:.1f} MB ({len(checkpoints)} files)")
    
    def get_total_size_mb(self) -> float:
        """Get total size of all checkpoints in MB."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.pt')
        files = glob.glob(pattern)
        return sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    
    def clean_all(self) -> None:
        """Delete all checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.pt')
        files = glob.glob(pattern)
        for f in files:
            os.remove(f)
        print(f"  [Checkpoint] Deleted {len(files)} files")


if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("=" * 60)
    print("  utils/checkpoint.py — Unit Test")
    print("=" * 60)
    
    tmp_dir = tempfile.mkdtemp()
    
    # Test 1: Creation
    print("\n[Test 1] Creation")
    mgr = CheckpointManager(checkpoint_dir=tmp_dir, keep_last_n=3)
    print(f"  Dir: {tmp_dir}")
    print("  ✓ Created")
    
    # Test 2: Build and save
    print("\n[Test 2] Build and save")
    backbone = torch.nn.Linear(10, 5)
    controller = torch.nn.Linear(5, 3)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(controller.parameters())
    )
    
    state = mgr.build_state(
        epoch=10, global_step=500,
        backbone=backbone, controller=controller, optimizer=optimizer,
    )
    path = mgr.save(state, 'test')
    assert os.path.exists(path)
    print("  ✓ Passed")
    
    # Test 3: Save best
    print("\n[Test 3] Save best")
    saved = mgr.save_if_best(
        val_loss=0.5, epoch=10,
        global_step=500, backbone=backbone, controller=controller,
    )
    assert saved is True
    assert mgr.best_val_loss == 0.5
    
    saved2 = mgr.save_if_best(
        val_loss=0.6, epoch=11,
        global_step=550, backbone=backbone, controller=controller,
    )
    assert saved2 is False
    print("  ✓ Best tracking works")
    
    # Test 4: Periodic save + pruning
    print("\n[Test 4] Periodic save + pruning")
    for epoch in range(50):
        mgr.save_periodic(
            epoch=epoch, save_every=10,
            global_step=epoch * 50,
            backbone=backbone, controller=controller,
        )
    
    ckpts = mgr.list_checkpoints()
    epoch_ckpts = [c for c in ckpts if 'epoch_' in c['file']]
    print(f"  Epoch checkpoints: {len(epoch_ckpts)} (keep_last_n=3)")
    for c in epoch_ckpts:
        print(f"    {c['file']}")
    assert len(epoch_ckpts) <= 3
    print("  ✓ Pruning works")
    
    # Test 5: Load best
    print("\n[Test 5] Load best")
    loaded = mgr.load_best()
    assert loaded is not None
    assert loaded['epoch'] == 10
    assert 'backbone_state_dict' in loaded
    assert 'controller_state_dict' in loaded
    print("  ✓ Load works")
    
    # Test 6: Restore model
    print("\n[Test 6] Restore model")
    new_backbone = torch.nn.Linear(10, 5)
    new_controller = torch.nn.Linear(5, 3)
    resume_epoch = mgr.restore_model(loaded, new_backbone, new_controller)
    assert resume_epoch == 10
    for p1, p2 in zip(backbone.parameters(), new_backbone.parameters()):
        assert torch.equal(p1, p2)
    print("  ✓ Model restore works")
    
    # Test 7: Print checkpoints
    print("\n[Test 7] Print checkpoints")
    mgr.print_checkpoints()
    total = mgr.get_total_size_mb()
    print(f"  Total size: {total:.1f} MB")
    print("  ✓ Passed")
    
    # Cleanup
    shutil.rmtree(tmp_dir)
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)