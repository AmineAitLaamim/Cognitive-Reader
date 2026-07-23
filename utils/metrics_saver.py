"""
utils/metrics_saver.py
Saves training metrics to JSON files in a dedicated metrics/ folder.

Saves:
  metrics/
  ├── train_history.json      — per-epoch training losses
  ├── val_history.json        — per-epoch validation losses
  ├── ood_results.json        — OOD evaluation results
  ├── best.json               — best validation metrics
  └── summary.json            — final summary
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class MetricsSaver:
    """
    Saves metrics to JSON files for offline analysis and plotting.
    
    Usage:
        saver = MetricsSaver('./metrics')
        
        # After each epoch:
        saver.append_train(train_metrics, epoch)
        saver.append_val(val_metrics, epoch)
        
        # After OOD eval:
        saver.save_ood(ood_results, epoch)
        
        # When best improves:
        saver.save_best(val_metrics, epoch)
        
        # At end of training:
        saver.save_summary(final_summary)
    """
    
    def __init__(self, metrics_dir: str = './metrics'):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        self._train_history: List[Dict] = []
        self._val_history: List[Dict] = []
        self._ood_history: List[Dict] = []
    
    # ==============================================================
    # TRAINING METRICS
    # ==============================================================
    
    def append_train(self, metrics: Dict[str, float], epoch: int) -> None:
        """Append one epoch of training metrics."""
        entry = {'epoch': epoch, **self._clean(metrics)}
        self._train_history.append(entry)
        self._write_json('train_history.json', self._train_history)
    
    def append_val(self, metrics: Dict[str, float], epoch: int) -> None:
        """Append one epoch of validation metrics."""
        entry = {'epoch': epoch, **self._clean(metrics)}
        self._val_history.append(entry)
        self._write_json('val_history.json', self._val_history)
    
    # ==============================================================
    # OOD RESULTS
    # ==============================================================
    
    def save_ood(self, results: Dict[int, Dict], epoch: int) -> None:
        """Save OOD evaluation results."""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'results': {str(k): self._clean(v) for k, v in results.items()}
        }
        self._ood_history.append(entry)
        self._write_json('ood_results.json', self._ood_history)
    
    # ==============================================================
    # BEST METRICS
    # ==============================================================
    
    def save_best(self, metrics: Dict[str, float], epoch: int) -> None:
        """Save the best validation metrics (overwritten each time best improves)."""
        data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **self._clean(metrics)
        }
        self._write_json('best.json', data)
    
    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    
    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save final training summary."""
        summary['timestamp'] = datetime.now().isoformat()
        self._write_json('summary.json', self._clean_deep(summary))
    
    # ==============================================================
    # LOAD (for resuming or analysis)
    # ==============================================================
    
    def load_train_history(self) -> List[Dict]:
        """Load training history from disk."""
        return self._read_json('train_history.json') or []
    
    def load_val_history(self) -> List[Dict]:
        """Load validation history from disk."""
        return self._read_json('val_history.json') or []
    
    def load_ood_results(self) -> List[Dict]:
        """Load OOD results from disk."""
        return self._read_json('ood_results.json') or []
    
    def load_best(self) -> Optional[Dict]:
        """Load best metrics from disk."""
        return self._read_json('best.json')
    
    # ==============================================================
    # INTERNAL
    # ==============================================================
    
    def _write_json(self, filename: str, data: Any) -> None:
        """Write data to a JSON file in the metrics directory."""
        path = os.path.join(self.metrics_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
    
    def _read_json(self, filename: str) -> Optional[Any]:
        """Read a JSON file from the metrics directory."""
        path = os.path.join(self.metrics_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def _clean(self, d: Dict) -> Dict:
        """Convert numpy/torch types to Python native types."""
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating,)):
                cleaned[k] = float(v)
            elif isinstance(v, (np.ndarray,)):
                cleaned[k] = v.tolist()
            elif isinstance(v, float) and (v != v):  # NaN check
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned
    
    def _clean_deep(self, obj: Any) -> Any:
        """Recursively clean nested structures."""
        if isinstance(obj, dict):
            return {k: self._clean_deep(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_deep(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj
    
    @staticmethod
    def _json_default(obj):
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)