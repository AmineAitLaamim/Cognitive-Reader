"""
eval/metrics.py
Comprehensive evaluation metrics for the Cognitive Reader project.

Standalone module — no dependency on model code.
Operates purely on token sequences, bounding boxes, and arrays.

Metric Categories:
  1. Sequence-level: exact match, normalized edit distance.
  2. Digit-level: per-digit accuracy, per-class precision/recall, confusion matrix.
  3. Chunk-level: boundary precision/recall/F1, chunk size distribution.
  4. Detection-level: center precision/recall, IoU-based box metrics.
  5. OOD generalization: accuracy vs length, degradation rate.
  6. Aggregation: batch-level accumulation and reporting.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ==============================================================
# 1. SEQUENCE-LEVEL METRICS
# ==============================================================

def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute the Levenshtein edit distance between two token sequences.
    
    Operations: insertion, deletion, substitution (each costs 1).
    
    Args:
        seq1: Ground-truth token sequence.
        seq2: Predicted token sequence.
    
    Returns:
        Minimum number of edits to transform seq1 into seq2.
    """
    n, m = len(seq1), len(seq2)
    
    # Optimize: use two rows instead of full matrix
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev
    
    return prev[m]


def normalized_edit_distance(seq1: List[str], seq2: List[str]) -> float:
    """
    Edit distance normalized by the ground-truth length.
    
    Returns:
        Value in [0, inf). 0 = perfect match. 1 = completely wrong.
        Values > 1 are possible if prediction is much longer than GT.
    """
    if len(seq1) == 0:
        return 0.0 if len(seq2) == 0 else float(len(seq2))
    return levenshtein_distance(seq1, seq2) / len(seq1)


def exact_match(seq1: List[str], seq2: List[str]) -> bool:
    """Check if two token sequences are identical."""
    return seq1 == seq2


def sequence_metrics(
    gt_tokens: List[str],
    pred_tokens: List[str]
) -> Dict[str, Any]:
    """
    Compute all sequence-level metrics.
    
    Args:
        gt_tokens: Ground-truth tokens (e.g., ['3', '8', '<CHUNK>', '1']).
        pred_tokens: Predicted tokens.
    
    Returns:
        Dict with exact_match, edit_distance, normalized_edit_distance,
        gt_length, pred_length, length_ratio.
    """
    edit_dist = levenshtein_distance(gt_tokens, pred_tokens)
    
    return {
        'exact_match': 1.0 if gt_tokens == pred_tokens else 0.0,
        'edit_distance': edit_dist,
        'normalized_edit_distance': normalized_edit_distance(gt_tokens, pred_tokens),
        'gt_length': len(gt_tokens),
        'pred_length': len(pred_tokens),
        'length_ratio': len(pred_tokens) / max(len(gt_tokens), 1),
    }


# ==============================================================
# 2. DIGIT-LEVEL METRICS
# ==============================================================

def extract_digits(tokens: List[str]) -> List[str]:
    """Extract only digit tokens (remove <CHUNK>, <END>, <JUMP>)."""
    return [t for t in tokens if t not in ('<CHUNK>', '<END>', '<JUMP>')]


def digit_accuracy(
    gt_tokens: List[str],
    pred_tokens: List[str]
) -> Dict[str, float]:
    """
    Compute per-digit accuracy (ignoring chunk tokens).
    
    Aligns digits by position (zip). If lengths differ, only compares
    up to the shorter length.
    
    Returns:
        Dict with digit_accuracy, num_correct, num_gt_digits, num_pred_digits.
    """
    gt_digits = extract_digits(gt_tokens)
    pred_digits = extract_digits(pred_tokens)
    
    num_correct = sum(
        1 for g, p in zip(gt_digits, pred_digits) if g == p
    )
    
    return {
        'digit_accuracy': num_correct / max(len(gt_digits), 1),
        'num_correct': num_correct,
        'num_gt_digits': len(gt_digits),
        'num_pred_digits': len(pred_digits),
    }


def digit_confusion_matrix(
    gt_tokens: List[str],
    pred_tokens: List[str],
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute the digit-level confusion matrix.
    
    Args:
        gt_tokens: Ground-truth tokens.
        pred_tokens: Predicted tokens.
        num_classes: Number of digit classes (10 for 0-9).
    
    Returns:
        [num_classes, num_classes] numpy array.
        Row = ground truth, Column = prediction.
    """
    gt_digits = extract_digits(gt_tokens)
    pred_digits = extract_digits(pred_tokens)
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for g, p in zip(gt_digits, pred_digits):
        try:
            g_idx = int(g)
            p_idx = int(p)
            if 0 <= g_idx < num_classes and 0 <= p_idx < num_classes:
                cm[g_idx, p_idx] += 1
        except (ValueError, TypeError):
            continue
    
    return cm


def per_class_precision_recall(
    confusion_matrix: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1 from a confusion matrix.
    
    Args:
        confusion_matrix: [C, C] array. Row = GT, Col = Pred.
    
    Returns:
        Dict mapping class label -> {precision, recall, f1, support}.
    """
    C = confusion_matrix.shape[0]
    results = {}
    
    for c in range(C):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp  # Predicted as c but not c
        fn = confusion_matrix[c, :].sum() - tp  # Actually c but not predicted as c
        support = confusion_matrix[c, :].sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        results[str(c)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support)
        }
    
    return results


# ==============================================================
# 3. CHUNK-LEVEL METRICS
# ==============================================================

def extract_chunk_boundaries(tokens: List[str]) -> List[int]:
    """
    Extract chunk boundary positions (digit-index where <CHUNK> occurs).
    
    Example:
        ['3', '8', '<CHUNK>', '1', '2'] -> [2]
        (The <CHUNK> occurs after 2 digits: '3' and '8')
    """
    boundaries = []
    digit_pos = 0
    for token in tokens:
        if token == '<CHUNK>':
            boundaries.append(digit_pos)
        elif token not in ('<END>', '<JUMP>'):
            digit_pos += 1
    return boundaries


def extract_chunks(tokens: List[str]) -> List[List[str]]:
    """
    Split a token sequence into chunks (lists of digit tokens).
    
    Example:
        ['3', '8', '<CHUNK>', '1', '2'] -> [['3', '8'], ['1', '2']]
    """
    chunks = []
    current_chunk = []
    
    for token in tokens:
        if token == '<CHUNK>':
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
        elif token not in ('<END>', '<JUMP>'):
            current_chunk.append(token)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_boundary_metrics(
    gt_tokens: List[str],
    pred_tokens: List[str],
    tolerance: int = 0
) -> Dict[str, float]:
    """
    Compute chunk boundary precision, recall, and F1.
    
    Args:
        gt_tokens: Ground-truth tokens.
        pred_tokens: Predicted tokens.
        tolerance: Allowed positional offset for a match (0 = exact).
    
    Returns:
        Dict with chunk_precision, chunk_recall, chunk_f1,
        chunk_tp, chunk_fp, chunk_fn.
    """
    gt_boundaries = set(extract_chunk_boundaries(gt_tokens))
    pred_boundaries = set(extract_chunk_boundaries(pred_tokens))
    
    if tolerance == 0:
        tp = len(gt_boundaries & pred_boundaries)
        fp = len(pred_boundaries - gt_boundaries)
        fn = len(gt_boundaries - pred_boundaries)
    else:
        # Fuzzy matching: a predicted boundary matches a GT boundary
        # if they are within +/-tolerance positions
        matched_gt = set()
        matched_pred = set()
        
        for pb in pred_boundaries:
            for gb in gt_boundaries:
                if gb not in matched_gt and abs(pb - gb) <= tolerance:
                    matched_gt.add(gb)
                    matched_pred.add(pb)
                    break
        
        tp = len(matched_gt)
        fp = len(pred_boundaries - matched_pred)
        fn = len(gt_boundaries - matched_gt)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        'chunk_precision': precision,
        'chunk_recall': recall,
        'chunk_f1': f1,
        'chunk_tp': tp,
        'chunk_fp': fp,
        'chunk_fn': fn,
        'num_gt_chunks': len(gt_boundaries),
        'num_pred_chunks': len(pred_boundaries),
    }


def chunk_size_distribution(
    tokens: List[str]
) -> Dict[str, Any]:
    """
    Compute the distribution of chunk sizes.
    
    Returns:
        Dict with chunk_sizes (list), mean, std, min, max, histogram.
    """
    chunks = extract_chunks(tokens)
    sizes = [len(c) for c in chunks]
    
    if len(sizes) == 0:
        return {
            'chunk_sizes': [],
            'mean': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'num_chunks': 0,
            'histogram': {}
        }
    
    # Histogram: size -> count
    histogram = defaultdict(int)
    for s in sizes:
        histogram[s] += 1
    
    return {
        'chunk_sizes': sizes,
        'mean': float(np.mean(sizes)),
        'std': float(np.std(sizes)),
        'min': int(np.min(sizes)),
        'max': int(np.max(sizes)),
        'num_chunks': len(sizes),
        'histogram': dict(histogram)
    }


def chunk_size_distribution_distance(
    gt_tokens: List[str],
    pred_tokens: List[str]
) -> Dict[str, float]:
    """
    Compare chunk size distributions between GT and prediction.
    
    Uses Jensen-Shannon divergence between the two histograms.
    
    Returns:
        Dict with js_divergence, mean_size_gt, mean_size_pred.
    """
    gt_dist = chunk_size_distribution(gt_tokens)
    pred_dist = chunk_size_distribution(pred_tokens)
    
    # Build aligned histograms
    all_sizes = sorted(
        set(gt_dist['histogram'].keys()) | set(pred_dist['histogram'].keys())
    )
    
    if len(all_sizes) == 0:
        return {'js_divergence': 0.0, 'mean_size_gt': 0.0, 'mean_size_pred': 0.0}
    
    gt_hist = np.array(
        [gt_dist['histogram'].get(s, 0) for s in all_sizes], dtype=np.float64
    )
    pred_hist = np.array(
        [pred_dist['histogram'].get(s, 0) for s in all_sizes], dtype=np.float64
    )
    
    # Normalize to probability distributions
    gt_sum = gt_hist.sum()
    pred_sum = pred_hist.sum()
    
    if gt_sum > 0:
        gt_prob = gt_hist / gt_sum
    else:
        gt_prob = np.ones(len(all_sizes)) / len(all_sizes)
    
    if pred_sum > 0:
        pred_prob = pred_hist / pred_sum
    else:
        pred_prob = np.ones(len(all_sizes)) / len(all_sizes)
    
    # Jensen-Shannon divergence
    m = 0.5 * (gt_prob + pred_prob)
    kl_gt = np.sum(gt_prob * np.log(gt_prob / (m + 1e-10) + 1e-10))
    kl_pred = np.sum(pred_prob * np.log(pred_prob / (m + 1e-10) + 1e-10))
    js_div = 0.5 * (kl_gt + kl_pred)
    
    return {
        'js_divergence': float(js_div),
        'mean_size_gt': gt_dist['mean'],
        'mean_size_pred': pred_dist['mean'],
        'num_chunks_gt': gt_dist['num_chunks'],
        'num_chunks_pred': pred_dist['num_chunks'],
    }


# ==============================================================
# 4. DETECTION-LEVEL METRICS
# ==============================================================

def detection_metrics(
    gt_centers: np.ndarray,
    pred_centers: np.ndarray,
    match_radius: float = 10.0
) -> Dict[str, float]:
    """
    Compute detection precision, recall, and F1 based on center proximity.
    
    A predicted center is a true positive if it is within match_radius
    pixels of a ground-truth center (greedy matching).
    
    Args:
        gt_centers: [M, 2] — ground-truth digit centers (x, y).
        pred_centers: [N, 2] — predicted digit centers (x, y).
        match_radius: Maximum distance for a match (pixels).
    
    Returns:
        Dict with detection_precision, detection_recall, detection_f1,
        detection_tp, detection_fp, detection_fn.
    """
    M = gt_centers.shape[0]
    N = pred_centers.shape[0]
    
    if M == 0 and N == 0:
        return {
            'detection_precision': 1.0, 'detection_recall': 1.0,
            'detection_f1': 1.0, 'detection_tp': 0,
            'detection_fp': 0, 'detection_fn': 0
        }
    if M == 0:
        return {
            'detection_precision': 0.0, 'detection_recall': 1.0,
            'detection_f1': 0.0, 'detection_tp': 0,
            'detection_fp': N, 'detection_fn': 0
        }
    if N == 0:
        return {
            'detection_precision': 1.0, 'detection_recall': 0.0,
            'detection_f1': 0.0, 'detection_tp': 0,
            'detection_fp': 0, 'detection_fn': M
        }
    
    # Compute pairwise distances
    dist = np.sqrt(
        (gt_centers[:, np.newaxis, 0] - pred_centers[np.newaxis, :, 0]) ** 2 +
        (gt_centers[:, np.newaxis, 1] - pred_centers[np.newaxis, :, 1]) ** 2
    )  # [M, N]
    
    # Greedy matching: sort all (gt, pred) pairs by distance
    matched_gt = set()
    matched_pred = set()
    
    pairs = []
    for i in range(M):
        for j in range(N):
            if dist[i, j] <= match_radius:
                pairs.append((dist[i, j], i, j))
    pairs.sort()
    
    for d, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
    
    tp = len(matched_gt)
    fp = N - len(matched_pred)
    fn = M - len(matched_gt)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1,
        'detection_tp': tp,
        'detection_fp': fp,
        'detection_fn': fn,
    }


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: [4] — (x1, y1, x2, y2).
        box2: [4] — (x1, y1, x2, y2).
    
    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / max(union, 1e-8)


def detection_metrics_iou(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute detection metrics using IoU-based matching.
    
    Args:
        gt_boxes: [M, 4] — ground-truth boxes (x1, y1, x2, y2).
        pred_boxes: [N, 4] — predicted boxes.
        iou_threshold: Minimum IoU for a true positive.
    
    Returns:
        Dict with precision, recall, F1 at the given IoU threshold.
    """
    M = gt_boxes.shape[0]
    N = pred_boxes.shape[0]
    
    if M == 0 and N == 0:
        return {'iou_precision': 1.0, 'iou_recall': 1.0, 'iou_f1': 1.0}
    if M == 0:
        return {'iou_precision': 0.0, 'iou_recall': 1.0, 'iou_f1': 0.0}
    if N == 0:
        return {'iou_precision': 1.0, 'iou_recall': 0.0, 'iou_f1': 0.0}
    
    # Compute IoU matrix
    iou_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = box_iou(gt_boxes[i], pred_boxes[j])
    
    # Greedy matching by highest IoU
    matched_gt = set()
    matched_pred = set()
    
    pairs = []
    for i in range(M):
        for j in range(N):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)  # Highest IoU first
    
    for iou_val, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
    
    tp = len(matched_gt)
    fp = N - len(matched_pred)
    fn = M - len(matched_gt)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        'iou_precision': precision,
        'iou_recall': recall,
        'iou_f1': f1,
        'iou_tp': tp,
        'iou_fp': fp,
        'iou_fn': fn,
        'mean_iou': float(iou_matrix.max(axis=1).mean()) if M > 0 else 0.0,
    }


# ==============================================================
# 5. COMPREHENSIVE SAMPLE METRICS
# ==============================================================

def compute_all_metrics(
    gt_tokens: List[str],
    pred_tokens: List[str],
    gt_centers: Optional[np.ndarray] = None,
    pred_centers: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    pred_boxes: Optional[np.ndarray] = None,
    detection_match_radius: float = 10.0,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute ALL metrics for a single sample.
    
    This is the master function that aggregates every metric category.
    
    Args:
        gt_tokens: Ground-truth token sequence.
        pred_tokens: Predicted token sequence.
        gt_centers: [M, 2] GT digit centers (optional).
        pred_centers: [N, 2] predicted digit centers (optional).
        gt_boxes: [M, 4] GT bounding boxes (optional).
        pred_boxes: [N, 4] predicted bounding boxes (optional).
        detection_match_radius: Radius for center-based detection matching.
        iou_threshold: IoU threshold for box-based detection matching.
    
    Returns:
        Dict with all metrics organized by category.
    """
    metrics = {}
    
    # Sequence-level
    seq = sequence_metrics(gt_tokens, pred_tokens)
    metrics.update(seq)
    
    # Digit-level
    dig = digit_accuracy(gt_tokens, pred_tokens)
    metrics.update(dig)
    
    # Confusion matrix
    cm = digit_confusion_matrix(gt_tokens, pred_tokens)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    metrics['per_class'] = per_class_precision_recall(cm)
    
    # Chunk-level
    chunk = chunk_boundary_metrics(gt_tokens, pred_tokens)
    metrics.update(chunk)
    
    # Chunk size distribution comparison
    chunk_dist = chunk_size_distribution_distance(gt_tokens, pred_tokens)
    metrics.update(chunk_dist)
    
    # GT chunk sizes
    gt_chunk_dist = chunk_size_distribution(gt_tokens)
    metrics['gt_chunk_sizes'] = gt_chunk_dist['chunk_sizes']
    metrics['gt_chunk_mean_size'] = gt_chunk_dist['mean']
    
    pred_chunk_dist = chunk_size_distribution(pred_tokens)
    metrics['pred_chunk_sizes'] = pred_chunk_dist['chunk_sizes']
    metrics['pred_chunk_mean_size'] = pred_chunk_dist['mean']
    
    # Detection-level (if centers provided)
    if gt_centers is not None and pred_centers is not None:
        det = detection_metrics(gt_centers, pred_centers, detection_match_radius)
        metrics.update(det)
    
    # Detection IoU (if boxes provided)
    if gt_boxes is not None and pred_boxes is not None:
        det_iou = detection_metrics_iou(gt_boxes, pred_boxes, iou_threshold)
        metrics.update(det_iou)
    
    return metrics


# ==============================================================
# 6. BATCH AGGREGATION
# ==============================================================

class MetricsAggregator:
    """
    Accumulates metrics across a batch or dataset and computes averages.
    
    Usage:
        agg = MetricsAggregator()
        for sample in dataset:
            metrics = compute_all_metrics(gt, pred)
            agg.update(metrics)
        summary = agg.summary()
    """
    
    def __init__(self):
        self._scalar_sums: Dict[str, float] = defaultdict(float)
        self._scalar_counts: Dict[str, int] = defaultdict(int)
        self._confusion_matrix: Optional[np.ndarray] = None
        self._all_gt_chunk_sizes: List[int] = []
        self._all_pred_chunk_sizes: List[int] = []
        self._num_samples: int = 0
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Add metrics from one sample."""
        self._num_samples += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._scalar_sums[key] += value
                self._scalar_counts[key] += 1
        
        # Accumulate confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            if self._confusion_matrix is None:
                self._confusion_matrix = cm
            else:
                self._confusion_matrix += cm
        
        # Accumulate chunk sizes
        if 'gt_chunk_sizes' in metrics:
            self._all_gt_chunk_sizes.extend(metrics['gt_chunk_sizes'])
        if 'pred_chunk_sizes' in metrics:
            self._all_pred_chunk_sizes.extend(metrics['pred_chunk_sizes'])
    
    def summary(self) -> Dict[str, Any]:
        """Compute averaged metrics across all samples."""
        result = {'num_samples': self._num_samples}
        
        # Average scalars
        for key in self._scalar_sums:
            count = self._scalar_counts[key]
            result[key] = self._scalar_sums[key] / max(count, 1)
        
        # Aggregated confusion matrix
        if self._confusion_matrix is not None:
            result['confusion_matrix'] = self._confusion_matrix.tolist()
            result['per_class'] = per_class_precision_recall(self._confusion_matrix)
        
        # Aggregated chunk size stats
        if self._all_gt_chunk_sizes:
            result['gt_chunk_size_mean'] = float(np.mean(self._all_gt_chunk_sizes))
            result['gt_chunk_size_std'] = float(np.std(self._all_gt_chunk_sizes))
        if self._all_pred_chunk_sizes:
            result['pred_chunk_size_mean'] = float(np.mean(self._all_pred_chunk_sizes))
            result['pred_chunk_size_std'] = float(np.std(self._all_pred_chunk_sizes))
        
        return result
    
    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._scalar_sums.clear()
        self._scalar_counts.clear()
        self._confusion_matrix = None
        self._all_gt_chunk_sizes = []
        self._all_pred_chunk_sizes = []
        self._num_samples = 0


# ==============================================================
# 7. OOD GENERALIZATION ANALYSIS
# ==============================================================

def ood_generalization_analysis(
    results_by_length: Dict[int, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Analyze how metrics degrade as sequence length increases.
    
    Args:
        results_by_length: Dict mapping sequence_length -> averaged metrics.
    
    Returns:
        Dict with degradation rates, critical length, and summary.
    """
    lengths = sorted(results_by_length.keys())
    
    if len(lengths) < 2:
        return {'degradation_rate': 0.0, 'critical_length': None}
    
    # Extract key metrics per length
    seq_accs = [results_by_length[l].get('exact_match', 0) for l in lengths]
    digit_accs = [results_by_length[l].get('digit_accuracy', 0) for l in lengths]
    chunk_f1s = [results_by_length[l].get('chunk_f1', 0) for l in lengths]
    
    # Degradation rate: slope of accuracy vs log(length)
    log_lengths = np.log(np.array(lengths, dtype=np.float64))
    
    def _slope(x, y):
        if len(x) < 2:
            return 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        return float(num / max(den, 1e-8))
    
    seq_degradation = _slope(log_lengths, np.array(seq_accs))
    digit_degradation = _slope(log_lengths, np.array(digit_accs))
    chunk_degradation = _slope(log_lengths, np.array(chunk_f1s))
    
    # Critical length: first length where exact_match < 0.5
    critical_length = None
    for l, acc in zip(lengths, seq_accs):
        if acc < 0.5:
            critical_length = l
            break
    
    return {
        'lengths': lengths,
        'sequence_accuracies': seq_accs,
        'digit_accuracies': digit_accs,
        'chunk_f1s': chunk_f1s,
        'seq_degradation_rate': seq_degradation,
        'digit_degradation_rate': digit_degradation,
        'chunk_degradation_rate': chunk_degradation,
        'critical_length': critical_length,
        'max_tested_length': lengths[-1],
        'accuracy_at_max_length': seq_accs[-1],
    }


# ==============================================================
# 8. REPORTING
# ==============================================================

def print_metrics_report(
    metrics: Dict[str, Any],
    title: str = "Evaluation Report"
) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    # Sequence
    print(f"\n  Sequence-Level:")
    print(f"    Exact match:          {metrics.get('exact_match', 0):.4f}")
    print(f"    Edit distance:        {metrics.get('edit_distance', 0):.1f}")
    print(f"    Norm edit distance:   {metrics.get('normalized_edit_distance', 0):.4f}")
    print(f"    GT length:            {metrics.get('gt_length', 0)}")
    print(f"    Pred length:          {metrics.get('pred_length', 0)}")
    
    # Digit
    print(f"\n  Digit-Level:")
    print(f"    Accuracy:             {metrics.get('digit_accuracy', 0):.4f}")
    print(f"    Correct / Total:      {metrics.get('num_correct', 0)} / {metrics.get('num_gt_digits', 0)}")
    
    # Per-class
    if 'per_class' in metrics:
        print(f"\n  Per-Class Digit Metrics:")
        print(f"    {'Class':>6s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
        for cls, vals in sorted(metrics['per_class'].items()):
            print(
                f"    {cls:>6s} {vals['precision']:>8.4f} {vals['recall']:>8.4f} "
                f"{vals['f1']:>8.4f} {vals['support']:>8d}"
            )
    
    # Chunk
    print(f"\n  Chunk-Level:")
    print(f"    Precision:            {metrics.get('chunk_precision', 0):.4f}")
    print(f"    Recall:               {metrics.get('chunk_recall', 0):.4f}")
    print(f"    F1:                   {metrics.get('chunk_f1', 0):.4f}")
    print(f"    GT chunks:            {metrics.get('num_gt_chunks', 0)}")
    print(f"    Pred chunks:          {metrics.get('num_pred_chunks', 0)}")
    print(f"    GT mean chunk size:   {metrics.get('mean_size_gt', 0):.2f}")
    print(f"    Pred mean chunk size: {metrics.get('mean_size_pred', 0):.2f}")
    print(f"    JS divergence:        {metrics.get('js_divergence', 0):.4f}")
    
    # Detection
    if 'detection_f1' in metrics:
        print(f"\n  Detection (center-based):")
        print(f"    Precision:            {metrics.get('detection_precision', 0):.4f}")
        print(f"    Recall:               {metrics.get('detection_recall', 0):.4f}")
        print(f"    F1:                   {metrics.get('detection_f1', 0):.4f}")
    
    if 'iou_f1' in metrics:
        print(f"\n  Detection (IoU-based):")
        print(f"    Precision:            {metrics.get('iou_precision', 0):.4f}")
        print(f"    Recall:               {metrics.get('iou_recall', 0):.4f}")
        print(f"    F1:                   {metrics.get('iou_f1', 0):.4f}")
        print(f"    Mean IoU:             {metrics.get('mean_iou', 0):.4f}")
    
    print(f"\n{'='*60}\n")


def metrics_to_json(metrics: Dict[str, Any]) -> str:
    """Serialize metrics to JSON string (excluding numpy arrays)."""
    import json
    
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            cleaned[k] = {kk: _convert(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            cleaned[k] = [_convert(item) for item in v]
        else:
            cleaned[k] = _convert(v)
    
    return json.dumps(cleaned, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print("  eval/metrics.py — Unit Test")
    print("=" * 60)
    
    # Test 1: Sequence metrics
    print("\n[Test 1] Sequence metrics")
    gt = ['3', '8', '<CHUNK>', '1', '2', '<CHUNK>', '5']
    pred = ['3', '8', '<CHUNK>', '1', '5', '<CHUNK>', '5']
    
    seq = sequence_metrics(gt, pred)
    print(f"  Exact match: {seq['exact_match']}")
    print(f"  Edit distance: {seq['edit_distance']}")
    assert seq['exact_match'] == 0.0
    assert seq['edit_distance'] == 1  # '2' -> '5'
    print("  ✓ Passed")
    
    # Test 2: Digit accuracy
    print("\n[Test 2] Digit accuracy")
    dig = digit_accuracy(gt, pred)
    print(f"  Accuracy: {dig['digit_accuracy']:.4f}")
    assert dig['digit_accuracy'] == 5 / 6  # 5 of 6 correct
    print("  ✓ Passed")
    
    # Test 3: Confusion matrix
    print("\n[Test 3] Confusion matrix")
    cm = digit_confusion_matrix(gt, pred)
    print(f"  Shape: {cm.shape}")
    print(f"  Total: {cm.sum()}")
    assert cm.sum() == 6  # 6 digit pairs
    assert cm[2, 5] == 1  # GT '2' predicted as '5'
    print("  ✓ Passed")
    
    # Test 4: Chunk boundary metrics
    print("\n[Test 4] Chunk boundary metrics")
    chunk = chunk_boundary_metrics(gt, pred)
    print(f"  Precision: {chunk['chunk_precision']:.4f}")
    print(f"  Recall: {chunk['chunk_recall']:.4f}")
    print(f"  F1: {chunk['chunk_f1']:.4f}")
    assert chunk['chunk_f1'] == 1.0  # Both chunks at correct positions
    print("  ✓ Passed")
    
    # Test 5: Chunk extraction
    print("\n[Test 5] Chunk extraction")
    chunks = extract_chunks(gt)
    print(f"  Chunks: {chunks}")
    assert chunks == [['3', '8'], ['1', '2'], ['5']]
    print("  ✓ Passed")
    
    # Test 6: Chunk size distribution
    print("\n[Test 6] Chunk size distribution")
    dist = chunk_size_distribution(gt)
    print(f"  Sizes: {dist['chunk_sizes']}")
    print(f"  Mean: {dist['mean']:.2f}")
    assert dist['chunk_sizes'] == [2, 2, 1]
    print("  ✓ Passed")
    
    # Test 7: Detection metrics
    print("\n[Test 7] Detection metrics")
    gt_centers = np.array([[100, 100], [200, 200], [300, 300]])
    pred_centers = np.array([[102, 98], [198, 203], [500, 500]])
    
    det = detection_metrics(gt_centers, pred_centers, match_radius=10.0)
    print(f"  TP: {det['detection_tp']}, FP: {det['detection_fp']}, FN: {det['detection_fn']}")
    assert det['detection_tp'] == 2
    assert det['detection_fp'] == 1
    assert det['detection_fn'] == 1
    print("  ✓ Passed")
    
    # Test 8: Box IoU
    print("\n[Test 8] Box IoU")
    b1 = np.array([0, 0, 10, 10])
    b2 = np.array([5, 5, 15, 15])
    iou = box_iou(b1, b2)
    print(f"  IoU: {iou:.4f}")
    assert abs(iou - 25 / 175) < 0.01
    print("  ✓ Passed")
    
    # Test 9: MetricsAggregator
    print("\n[Test 9] MetricsAggregator")
    agg = MetricsAggregator()
    for _ in range(5):
        m = compute_all_metrics(gt, pred)
        agg.update(m)
    
    summary = agg.summary()
    print(f"  Samples: {summary['num_samples']}")
    print(f"  Avg exact match: {summary['exact_match']:.4f}")
    print(f"  Avg digit acc: {summary['digit_accuracy']:.4f}")
    assert summary['num_samples'] == 5
    print("  ✓ Passed")
    
    # Test 10: OOD analysis
    print("\n[Test 10] OOD generalization analysis")
    ood_results = {
        50: {'exact_match': 0.9, 'digit_accuracy': 0.98, 'chunk_f1': 0.95},
        100: {'exact_match': 0.7, 'digit_accuracy': 0.95, 'chunk_f1': 0.88},
        200: {'exact_match': 0.4, 'digit_accuracy': 0.90, 'chunk_f1': 0.75},
        500: {'exact_match': 0.1, 'digit_accuracy': 0.82, 'chunk_f1': 0.55},
    }
    
    ood_analysis = ood_generalization_analysis(ood_results)
    print(f"  Critical length: {ood_analysis['critical_length']}")
    print(f"  Seq degradation: {ood_analysis['seq_degradation_rate']:.4f}")
    print(f"  Digit degradation: {ood_analysis['digit_degradation_rate']:.4f}")
    assert ood_analysis['critical_length'] == 200
    print("  ✓ Passed")
    
    # Test 11: Full report
    print("\n[Test 11] Full metrics report")
    full_metrics = compute_all_metrics(
        gt, pred,
        gt_centers=gt_centers,
        pred_centers=pred_centers
    )
    print_metrics_report(full_metrics, "Unit Test Report")
    
    # Test 12: JSON serialization
    print("\n[Test 12] JSON serialization")
    json_str = metrics_to_json(full_metrics)
    assert len(json_str) > 100
    print(f"  JSON length: {len(json_str)} chars")
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)