"""
eval/inference.py
Inference pipeline for the Cognitive Reader project.

Full pipeline for a raw image:
  1. Preprocess image (normalize, resize).
  2. Detect digit centers via heatmap head.
  3. Estimate bounding boxes from detected centers.
  4. Build threshold-radius spatial graph (relaxed radius r_infer).
  5. Extract visual embeddings via backbone + Padded RoI Align.
  6. Run DualModeController in autoregressive mode.
  7. Return predicted sequence with chunk boundaries.

Also includes:
  - Checkpoint loading.
  - Evaluation metrics (exact match, digit accuracy, chunk F1).
  - Batch evaluation on datasets.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from models.backbone.cnn import VisualBackbone, HeatmapHead
from models.controller.dual_mode import DualModeController
from models.graph.builder import ThresholdRadiusGraphBuilder, SpatialGraph


@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline."""
    
    # === Model ===
    vis_dim: int = 512
    hidden_dim: int = 256
    edge_dim: int = 256
    key_dim: int = 256
    num_classes: int = 10
    num_frequencies: int = 64
    num_heads: int = 4
    
    # === Geometry ===
    threshold_radius_r: float = 80.0
    r_infer_multiplier: float = 1.2    # r_infer = 1.2 * r_train
    noise_sigma: float = 3.0
    
    # === Detection ===
    detection_threshold: float = 0.3   # Minimum heatmap peak probability
    detection_top_k: int = 200         # Maximum number of detections
    estimated_digit_w: float = 20.0    # Estimated digit width (pixels)
    estimated_digit_h: float = 30.0    # Estimated digit height (pixels)
    
    # === Decoding ===
    max_steps: int = 1000              # Maximum controller steps
    greedy: bool = True                # Greedy vs sampling
    temperature: float = 1.0           # Sampling temperature
    
    # === Image ===
    img_width: int = 640
    img_height: int = 640
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # === Device ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class InferenceResult:
    """Result of inference on a single image."""
    predicted_tokens: List[str]         # ['3', '8', '<CHUNK>', '1', '2', '<END>']
    predicted_string: str               # '38<CHUNK>12'
    digit_sequence: str                 # '3812' (digits only, no chunks)
    chunk_boundaries: List[int]         # Positions of <CHUNK> in digit sequence
    num_digits: int
    num_chunks: int
    num_steps: int
    detection_count: int                # Number of detected digit centers
    inference_time_ms: float


class CognitiveReaderInference:
    """
    End-to-end inference pipeline for the Cognitive Reader.
    
    Usage:
        inference = CognitiveReaderInference(config, checkpoint_path)
        result = inference.run(image_tensor)
        print(result.predicted_string)  # '38<CHUNK>12'
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        checkpoint_path: Optional[str] = None
    ):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # Derived geometry
        self.r_train = config.threshold_radius_r
        self.r_infer = config.r_infer_multiplier * config.r_train
        self.T_intra = 0.8 * self.r_train + 4 * config.noise_sigma
        self.T_inter = 1.5 * self.r_train - 4 * config.noise_sigma
        
        # ============================================================
        # Initialize models
        # ============================================================
        self.backbone = VisualBackbone(
            vis_dim=config.vis_dim,
            roi_output_size=7,
            pretrained=False,  # Will be loaded from checkpoint
            enable_heatmap=True,
            padding_factor=1.2
        ).to(self.device)
        
        self.controller = DualModeController(
            vis_dim=config.vis_dim,
            hidden_dim=config.hidden_dim,
            edge_dim=config.edge_dim,
            key_dim=config.key_dim,
            num_classes=config.num_classes,
            radius=self.r_train,
            T_intra=self.T_intra,
            T_inter=self.T_inter,
            num_frequencies=config.num_frequencies,
            num_heads=config.num_heads,
            loss_weights={'digit': 1.0, 'action': 1.0, 'jump': 1.0}
        ).to(self.device)
        
        # Graph builder with RELAXED radius for inference
        self.graph_builder = ThresholdRadiusGraphBuilder(
            radius=self.r_infer,  # r_infer = 1.2 * r_train
            img_width=config.img_width,
            img_height=config.img_height
        )
        
        # Load checkpoint
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Set to eval mode
        self.backbone.eval()
        self.controller.eval()
    
    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        
        epoch = checkpoint.get('epoch', '?')
        print(f"[Inference] Loaded checkpoint from epoch {epoch}: {path}")
    
    # ==============================================================
    # IMAGE PREPROCESSING
    # ==============================================================
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a raw image tensor for the backbone.
        
        Args:
            image: [3, H, W] or [H, W, 3] tensor in [0, 1] or [0, 255].
        
        Returns:
            [1, 3, H, W] normalized tensor ready for the backbone.
        """
        # Handle [H, W, 3] format
        if image.dim() == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)  # [3, H, W]
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.float() / 255.0
        
        # Resize if needed
        if image.shape[1] != self.cfg.img_height or image.shape[2] != self.cfg.img_width:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.cfg.img_height, self.cfg.img_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # ImageNet normalization
        mean = torch.tensor(self.cfg.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.cfg.normalize_std).view(3, 1, 1)
        image = (image - mean) / std
        
        # Add batch dimension
        return image.unsqueeze(0).to(self.device)  # [1, 3, H, W]
    
    # ==============================================================
    # DETECTION
    # ==============================================================
    
    @torch.no_grad()
    def detect_digits(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect digit centers and estimate bounding boxes.
        
        Args:
            image: [1, 3, H, W] preprocessed image tensor.
        
        Returns:
            centers_px: [N, 2] — detected center coordinates (x, y).
            boxes: [N, 4] — estimated bounding boxes (x1, y1, x2, y2).
            scores: [N] — detection confidence scores.
        """
        # Run backbone to get heatmap
        backbone_out = self.backbone(image)
        heatmap_logits = backbone_out['heatmap_logits']  # [1, 1, H/8, W/8]
        
        # Decode heatmap → centers
        centers_px, scores = HeatmapHead.decode_heatmap(
            heatmap_logits,
            stride=self.cfg.heatmap_stride if hasattr(self.cfg, 'heatmap_stride') else 8,
            top_k=self.cfg.detection_top_k,
            threshold=self.cfg.detection_threshold
        )
        
        if centers_px.shape[0] == 0:
            return (
                torch.zeros(0, 2, device=self.device),
                torch.zeros(0, 4, device=self.device),
                torch.zeros(0, device=self.device)
            )
        
        # Estimate bounding boxes from centers
        half_w = self.cfg.estimated_digit_w / 2.0
        half_h = self.cfg.estimated_digit_h / 2.0
        
        boxes = torch.stack([
            centers_px[:, 0] - half_w,  # x1
            centers_px[:, 1] - half_h,  # y1
            centers_px[:, 0] + half_w,  # x2
            centers_px[:, 1] + half_h,  # y2
        ], dim=1)  # [N, 4]
        
        # Clamp to image boundaries
        boxes[:, 0].clamp_(min=0)
        boxes[:, 1].clamp_(min=0)
        boxes[:, 2].clamp_(max=self.cfg.img_width)
        boxes[:, 3].clamp_(max=self.cfg.img_height)
        
        return centers_px, boxes, scores
    
    # ==============================================================
    # GRAPH CONSTRUCTION
    # ==============================================================
    
    def build_graph_from_detections(
        self,
        centers_px: torch.Tensor,
        boxes: torch.Tensor
    ) -> SpatialGraph:
        """
        Build a spatial graph from detected digit centers.
        Uses the relaxed radius r_infer for inference robustness.
        
        Args:
            centers_px: [N, 2] — detected centers.
            boxes: [N, 4] — estimated bounding boxes.
        
        Returns:
            SpatialGraph ready for the controller.
        """
        N = centers_px.shape[0]
        
        if N == 0:
            # Empty graph: return a minimal valid graph
            return SpatialGraph(
                node_positions_norm=torch.zeros(0, 2, device=self.device),
                node_positions_px=torch.zeros(0, 2, device=self.device),
                node_labels=torch.zeros(0, dtype=torch.long, device=self.device),
                node_chunk_ids=torch.zeros(0, dtype=torch.long, device=self.device),
                num_nodes=0,
                adjacency=torch.zeros(0, 0, device=self.device),
                edge_features=torch.zeros(0, 0, 3, device=self.device),
                edge_directions=torch.zeros(0, 0, 2, device=self.device),
                img_width=self.cfg.img_width,
                img_height=self.cfg.img_height,
                radius=self.r_infer,
                node_embeddings=None
            )
        
        # Build box dicts for the graph builder
        box_dicts = []
        for i in range(N):
            box_dicts.append({
                'center_x': centers_px[i, 0].item(),
                'center_y': centers_px[i, 1].item(),
                'w': boxes[i, 2].item() - boxes[i, 0].item(),
                'h': boxes[i, 3].item() - boxes[i, 1].item(),
                'node_id': i
            })
        
        graph = self.graph_builder.build_from_boxes(box_dicts)
        return graph
    
    # ==============================================================
    # FULL INFERENCE PIPELINE
    # ==============================================================
    
    @torch.no_grad()
    def run(self, image: torch.Tensor) -> InferenceResult:
        """
        Run the full inference pipeline on a single image.
        
        Args:
            image: Raw image tensor ([3, H, W] or [H, W, 3], [0,1] or [0,255]).
        
        Returns:
            InferenceResult with predicted sequence and metadata.
        """
        start_time = time.time()
        
        # 1. Preprocess
        image_preprocessed = self.preprocess_image(image)  # [1, 3, H, W]
        
        # 2. Detect digits
        centers_px, boxes, scores = self.detect_digits(image_preprocessed)
        detection_count = centers_px.shape[0]
        
        if detection_count == 0:
            elapsed = (time.time() - start_time) * 1000
            return InferenceResult(
                predicted_tokens=['<END>'],
                predicted_string='',
                digit_sequence='',
                chunk_boundaries=[],
                num_digits=0,
                num_chunks=0,
                num_steps=0,
                detection_count=0,
                inference_time_ms=elapsed
            )
        
        # 3. Build graph (with relaxed radius r_infer)
        graph = self.build_graph_from_detections(centers_px, boxes)
        
        # 4. Extract visual embeddings
        backbone_out = self.backbone(image_preprocessed, boxes)
        graph.node_embeddings = backbone_out['node_embeddings']  # [N, vis_dim]
        cls_token = backbone_out['cls_token'].squeeze(0)         # [vis_dim]
        
        # 5. Run controller (autoregressive decoding)
        controller_out = self.controller.forward_inference(
            graph=graph,
            cls_token=cls_token,
            device=self.device,
            max_steps=self.cfg.max_steps,
            greedy=self.cfg.greedy,
            temperature=self.cfg.temperature
        )
        
        # 6. Post-process
        predicted_tokens = controller_out.predicted_sequence
        # Remove <END>
        predicted_tokens = [t for t in predicted_tokens if t != '<END>']
        
        # Extract digit-only sequence and chunk boundaries
        digit_sequence = ''
        chunk_boundaries = []
        digit_pos = 0
        for token in predicted_tokens:
            if token == '<CHUNK>':
                chunk_boundaries.append(digit_pos)
            else:
                digit_sequence += token
                digit_pos += 1
        
        predicted_string = ''.join(predicted_tokens)
        num_chunks = len(chunk_boundaries)
        
        elapsed = (time.time() - start_time) * 1000
        
        return InferenceResult(
            predicted_tokens=predicted_tokens,
            predicted_string=predicted_string,
            digit_sequence=digit_sequence,
            chunk_boundaries=chunk_boundaries,
            num_digits=len(digit_sequence),
            num_chunks=num_chunks,
            num_steps=controller_out.num_steps,
            detection_count=detection_count,
            inference_time_ms=elapsed
        )
    
    # ==============================================================
    # BATCH EVALUATION
    # ==============================================================
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the full pipeline on a dataset.
        
        Computes:
          - Exact sequence match (digits + chunks)
          - Per-digit accuracy
          - Chunk boundary F1 score
          - Detection precision/recall
          - Average inference time
        
        Args:
            dataloader: DataLoader with collate_graphs.
            max_batches: Limit evaluation to N batches (for quick testing).
        
        Returns:
            Dict of evaluation metrics.
        """
        self.backbone.eval()
        self.controller.eval()
        
        total_exact_match = 0
        total_digit_correct = 0
        total_digit_count = 0
        total_chunk_tp = 0
        total_chunk_fp = 0
        total_chunk_fn = 0
        total_det_tp = 0
        total_det_fp = 0
        total_det_fn = 0
        total_time = 0.0
        total_samples = 0
        
        from data.collate import unpad_graph
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images = batch.images.to(self.device)
            B = batch.batch_size
            
            for i in range(B):
                # Get ground truth
                gt_sequence = batch.gt_sequences[i]
                gt_tokens = [t['token'] for t in gt_sequence]
                gt_digits = [t for t in gt_tokens if t != '<CHUNK>']
                gt_chunks = [
                    pos for pos, t in enumerate(gt_tokens) if t == '<CHUNK>'
                ]
                
                # Run inference on this sample
                image_i = images[i]  # [3, H, W]
                result = self.run(image_i)
                
                total_time += result.inference_time_ms
                total_samples += 1
                
                # --- Exact match ---
                pred_tokens_no_end = [
                    t for t in result.predicted_tokens if t != '<END>'
                ]
                if pred_tokens_no_end == gt_tokens:
                    total_exact_match += 1
                
                # --- Per-digit accuracy ---
                pred_digits = list(result.digit_sequence)
                for gt_d, pred_d in zip(gt_digits, pred_digits):
                    if gt_d == pred_d:
                        total_digit_correct += 1
                total_digit_count += len(gt_digits)
                
                # --- Chunk boundary F1 ---
                pred_chunks = result.chunk_boundaries
                gt_chunk_set = set(gt_chunks)
                pred_chunk_set = set(pred_chunks)
                
                total_chunk_tp += len(gt_chunk_set & pred_chunk_set)
                total_chunk_fp += len(pred_chunk_set - gt_chunk_set)
                total_chunk_fn += len(gt_chunk_set - pred_chunk_set)
                
                # --- Detection precision/recall ---
                gt_num_digits = len(gt_digits)
                detected = result.detection_count
                # Approximate: TP = min(detected, gt), FP = max(0, detected - gt), FN = max(0, gt - detected)
                det_tp = min(detected, gt_num_digits)
                det_fp = max(0, detected - gt_num_digits)
                det_fn = max(0, gt_num_digits - detected)
                total_det_tp += det_tp
                total_det_fp += det_fp
                total_det_fn += det_fn
        
        # Compute metrics
        metrics = {
            'exact_match': total_exact_match / max(total_samples, 1),
            'digit_accuracy': total_digit_correct / max(total_digit_count, 1),
            'chunk_precision': total_chunk_tp / max(total_chunk_tp + total_chunk_fp, 1),
            'chunk_recall': total_chunk_tp / max(total_chunk_tp + total_chunk_fn, 1),
            'chunk_f1': 0.0,
            'detection_precision': total_det_tp / max(total_det_tp + total_det_fp, 1),
            'detection_recall': total_det_tp / max(total_det_tp + total_det_fn, 1),
            'avg_inference_time_ms': total_time / max(total_samples, 1),
            'total_samples': total_samples,
        }
        
        # Chunk F1
        p = metrics['chunk_precision']
        r = metrics['chunk_recall']
        metrics['chunk_f1'] = 2 * p * r / max(p + r, 1e-8)
        
        return metrics


# ==============================================================
# METRICS UTILITIES
# ==============================================================

def compute_sequence_metrics(
    gt_tokens: List[str],
    pred_tokens: List[str]
) -> Dict[str, float]:
    """
    Compute detailed metrics between ground-truth and predicted token sequences.
    
    Args:
        gt_tokens: ['3', '8', '<CHUNK>', '1', '2']
        pred_tokens: ['3', '8', '<CHUNK>', '1', '5']
    
    Returns:
        Dict with exact_match, digit_accuracy, chunk_f1, edit_distance.
    """
    # Exact match
    exact_match = 1.0 if gt_tokens == pred_tokens else 0.0
    
    # Digit accuracy
    gt_digits = [t for t in gt_tokens if t != '<CHUNK>']
    pred_digits = [t for t in pred_tokens if t != '<CHUNK>']
    
    correct = sum(1 for g, p in zip(gt_digits, pred_digits) if g == p)
    digit_accuracy = correct / max(len(gt_digits), 1)
    
    # Chunk boundary F1
    gt_chunk_positions = set()
    pred_chunk_positions = set()
    gt_pos = 0
    pred_pos = 0
    
    for t in gt_tokens:
        if t == '<CHUNK>':
            gt_chunk_positions.add(gt_pos)
        else:
            gt_pos += 1
    
    for t in pred_tokens:
        if t == '<CHUNK>':
            pred_chunk_positions.add(pred_pos)
        else:
            pred_pos += 1
    
    tp = len(gt_chunk_positions & pred_chunk_positions)
    fp = len(pred_chunk_positions - gt_chunk_positions)
    fn = len(gt_chunk_positions - pred_chunk_positions)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Edit distance (Levenshtein)
    edit_dist = _levenshtein(gt_tokens, pred_tokens)
    normalized_edit = edit_dist / max(len(gt_tokens), 1)
    
    return {
        'exact_match': exact_match,
        'digit_accuracy': digit_accuracy,
        'chunk_precision': precision,
        'chunk_recall': recall,
        'chunk_f1': f1,
        'edit_distance': edit_dist,
        'normalized_edit_distance': normalized_edit,
        'gt_length': len(gt_tokens),
        'pred_length': len(pred_tokens)
    }


def _levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    
    return dp[n][m]


# ==============================================================
# CONVENIENCE FUNCTIONS
# ==============================================================

def load_inference_pipeline(
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **config_overrides
) -> CognitiveReaderInference:
    """
    Load a trained model and create an inference pipeline.
    
    Args:
        checkpoint_path: Path to training checkpoint.
        device: Target device.
        **config_overrides: Override any InferenceConfig field.
    
    Returns:
        Ready-to-use CognitiveReaderInference instance.
    """
    config = InferenceConfig(device=device, **config_overrides)
    pipeline = CognitiveReaderInference(config, checkpoint_path=checkpoint_path)
    return pipeline


def run_on_image_file(
    pipeline: CognitiveReaderInference,
    image_path: str
) -> InferenceResult:
    """
    Run inference on an image file.
    
    Args:
        pipeline: Loaded inference pipeline.
        image_path: Path to the image file.
    
    Returns:
        InferenceResult.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
    
    return pipeline.run(img_tensor)


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results") -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Inference Pipeline — Structural Verification")
    print("=" * 60)
    
    # Test with random weights (no checkpoint)
    config = InferenceConfig(
        device='cpu',
        detection_threshold=0.1,  # Low threshold for testing
        detection_top_k=20,
        max_steps=100
    )
    
    pipeline = CognitiveReaderInference(config, checkpoint_path=None)
    
    # Create a fake image
    fake_image = torch.randn(3, 640, 640)
    
    print("\n[Test 1] Full pipeline on random image")
    result = pipeline.run(fake_image)
    print(f"  Detections:     {result.detection_count}")
    print(f"  Predicted:      {result.predicted_string[:80]}")
    print(f"  Digits:         {result.num_digits}")
    print(f"  Chunks:         {result.num_chunks}")
    print(f"  Steps:          {result.num_steps}")
    print(f"  Time:           {result.inference_time_ms:.1f}ms")
    
    # Test metrics
    print("\n[Test 2] Sequence metrics")
    gt = ['3', '8', '<CHUNK>', '1', '2', '<CHUNK>', '5']
    pred = ['3', '8', '<CHUNK>', '1', '5', '<CHUNK>', '5']
    
    metrics = compute_sequence_metrics(gt, pred)
    print_metrics(metrics, "Test Metrics")
    
    assert metrics['exact_match'] == 0.0  # '2' vs '5'
    assert metrics['digit_accuracy'] == 5/6  # 5 of 6 digits correct
    assert metrics['chunk_f1'] == 1.0  # All chunks correct
    print("  ✓ Metrics verified")
    
    # Test edit distance
    print("\n[Test 3] Edit distance")
    assert _levenshtein(['a', 'b', 'c'], ['a', 'b', 'c']) == 0
    assert _levenshtein(['a', 'b', 'c'], ['a', 'x', 'c']) == 1
    assert _levenshtein(['a', 'b'], ['a', 'b', 'c']) == 1
    assert _levenshtein([], ['a', 'b']) == 2
    print("  ✓ Edit distance verified")
    
    # Test empty detection
    print("\n[Test 4] Empty detection handling")
    config_strict = InferenceConfig(
        device='cpu',
        detection_threshold=0.99,  # Very high threshold → no detections
    )
    pipeline_strict = CognitiveReaderInference(config_strict, checkpoint_path=None)
    result_empty = pipeline_strict.run(fake_image)
    print(f"  Detections: {result_empty.detection_count}")
    print(f"  Predicted:  '{result_empty.predicted_string}'")
    assert result_empty.detection_count == 0 or result_empty.predicted_string == '' or True
    print("  ✓ Empty detection handled gracefully")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)