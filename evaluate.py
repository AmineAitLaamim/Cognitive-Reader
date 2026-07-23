"""
evaluate.py
Standalone entry point for OOD evaluation of the Cognitive Reader.

Runs the full inference pipeline on synthetic datasets with sequence
lengths that exceed the training distribution, computes comprehensive
metrics, and generates visualizations.

Usage:
    # Basic evaluation
    python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pt
    
    # Custom lengths and sample count
    python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pt \
                       --lengths 50 100 200 500 1000 \
                       --num_samples 100
    
    # With visualization
    python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pt \
                       --lengths 100 200 \
                       --num_samples 50 \
                       --visualize \
                       --viz_dir ./eval_viz
    
    # Save results to JSON
    python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pt \
                       --output ./eval_results.json
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

# Project imports
from models.backbone.cnn import VisualBackbone
from models.controller.dual_mode import DualModeController
from models.detector.heatmap import HeatmapHead
from models.detector.postprocess import (
    DigitDetector, PostProcessConfig, DetectionResult
)
from models.graph.builder import ThresholdRadiusGraphBuilder, SpatialGraph
from data.generator import ConstrainedPolarGenerator, GeneratorConfig
from data.renderer import DigitRenderer, RendererConfig
from data.collate import unpad_graph
from eval.metrics import (
    compute_all_metrics,
    MetricsAggregator,
    ood_generalization_analysis,
    print_metrics_report,
    metrics_to_json
)
from utils.logger import TrainingLogger, LoggerConfig


# ==============================================================
# CONFIGURATION
# ==============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='OOD Evaluation for Cognitive Reader'
    )
    
    # Model
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for evaluation'
    )
    
    # Evaluation
    parser.add_argument(
        '--lengths', type=int, nargs='+', default=[50, 100, 200, 500],
        help='Sequence lengths to evaluate'
    )
    parser.add_argument(
        '--num_samples', type=int, default=50,
        help='Number of samples per length'
    )
    parser.add_argument(
        '--max_steps_multiplier', type=int, default=3,
        help='Max controller steps = length * multiplier'
    )
    
    # Data generation
    parser.add_argument(
        '--img_size', type=int, default=640,
        help='Image size (square)'
    )
    parser.add_argument(
        '--radius', type=float, default=80.0,
        help='Threshold radius r'
    )
    parser.add_argument(
        '--noise_sigma', type=float, default=3.0,
        help='Detector noise sigma'
    )
    parser.add_argument(
        '--max_chunk_size', type=int, default=4,
        help='Maximum digits per chunk'
    )
    parser.add_argument(
        '--seed', type=int, default=9999,
        help='Random seed for reproducibility'
    )
    
    # Detection
    parser.add_argument(
        '--detection_threshold', type=float, default=0.3,
        help='Heatmap peak confidence threshold'
    )
    parser.add_argument(
        '--r_infer_multiplier', type=float, default=1.2,
        help='Inference radius multiplier (r_infer = r * multiplier)'
    )
    
    # Output
    parser.add_argument(
        '--output', type=str, default='./eval_results.json',
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualizations for sample predictions'
    )
    parser.add_argument(
        '--viz_dir', type=str, default='./eval_viz',
        help='Directory for visualization output'
    )
    parser.add_argument(
        '--viz_per_length', type=int, default=5,
        help='Number of visualizations per length'
    )
    
    # Logging
    parser.add_argument(
        '--log_dir', type=str, default='./eval_logs',
        help='Directory for evaluation logs'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-sample output'
    )
    
    return parser.parse_args()


# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
    radius: float,
    noise_sigma: float,
    r_infer_multiplier: float
) -> Dict[str, Any]:
    """
    Load all model components from a training checkpoint.
    
    Returns:
        Dict with 'backbone', 'controller', 'graph_builder', 'config'.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config if available
    trainer_config = checkpoint.get('trainer_config', {})
    dataset_config = checkpoint.get('dataset_config', {})
    
    # Derived geometry
    r_train = radius
    r_infer = r_infer_multiplier * r_train
    T_intra = 0.8 * r_train + 4 * noise_sigma
    T_inter = 1.5 * r_train - 4 * noise_sigma
    
    # Build backbone
    backbone = VisualBackbone(
        vis_dim=512,
        roi_output_size=7,
        pretrained=False,
        enable_heatmap=True,
        padding_factor=1.2
    ).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.eval()
    
    # Build controller
    controller = DualModeController(
        vis_dim=512,
        hidden_dim=256,
        edge_dim=256,
        key_dim=256,
        num_classes=10,
        radius=r_train,
        T_intra=T_intra,
        T_inter=T_inter,
        num_frequencies=64,
        num_heads=4,
        dropout=0.0  # No dropout at inference
    ).to(device)
    controller.load_state_dict(checkpoint['controller_state_dict'])
    controller.eval()
    
    # Graph builder with relaxed radius
    graph_builder = ThresholdRadiusGraphBuilder(
        radius=r_infer,
        img_width=dataset_config.get('img_width', 640),
        img_height=dataset_config.get('img_height', 640)
    )
    
    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}")
    print(f"  r_train={r_train}, r_infer={r_infer}")
    print(f"  T_intra={T_intra:.1f}, T_inter={T_inter:.1f}")
    
    return {
        'backbone': backbone,
        'controller': controller,
        'graph_builder': graph_builder,
        'r_train': r_train,
        'r_infer': r_infer,
        'epoch': epoch,
        'checkpoint_path': checkpoint_path,
    }


# ==============================================================
# SAMPLE GENERATION
# ==============================================================

def generate_eval_sample(
    total_digits: int,
    img_size: int,
    radius: float,
    noise_sigma: float,
    max_chunk_size: int,
    seed: int
) -> Dict[str, Any]:
    """
    Generate a single evaluation sample with ground truth.
    
    Returns:
        Dict with image, boxes, graph, gt_sequence, centers, etc.
    """
    # Generator
    gen_config = GeneratorConfig(
        img_width=img_size,
        img_height=img_size,
        threshold_radius_r=radius,
        noise_sigma=noise_sigma,
        max_chunk_size=max_chunk_size,
        min_chunk_size=1,
    )
    generator = ConstrainedPolarGenerator(gen_config)
    layout = generator.generate_sample(total_digits)
    
    # Renderer (no augmentation for eval)
    render_config = RendererConfig(
        img_width=img_size,
        img_height=img_size,
        rotation_max_deg=0.0,
        blur_probability=0.0,
        seed=seed,
    )
    renderer = DigitRenderer(render_config)
    render_output = renderer.render(layout)
    
    # Graph (with noisy boxes, matching inference conditions)
    graph_builder = ThresholdRadiusGraphBuilder(
        radius=radius,
        img_width=img_size,
        img_height=img_size
    )
    
    boxes_list = []
    labels_list = []
    chunk_ids_list = []
    for node in layout.nodes:
        boxes_list.append({
            'center_x': node.noisy_center_x,
            'center_y': node.noisy_center_y,
            'w': node.width,
            'h': node.height,
            'node_id': node.node_id,
        })
        labels_list.append(node.label)
        chunk_ids_list.append(node.chunk_id)
    
    graph = graph_builder.build_from_boxes(boxes_list, labels_list, chunk_ids_list)
    
    # Ground truth centers
    gt_centers = np.array([
        [node.center_x, node.center_y] for node in layout.nodes
    ])
    
    return {
        'image': render_output['image'],
        'boxes': render_output['boxes'],
        'heatmap_target': render_output['heatmap_target'],
        'graph': graph,
        'gt_sequence': layout.gt_sequence,
        'gt_centers': gt_centers,
        'gt_tokens': [t['token'] for t in layout.gt_sequence],
        'total_digits': len(layout.nodes),
        'num_chunks': layout.num_chunks,
    }


# ==============================================================
# SINGLE SAMPLE EVALUATION
# ==============================================================

@torch.no_grad()
def evaluate_single_sample(
    sample: Dict[str, Any],
    model: Dict[str, Any],
    device: torch.device,
    max_steps: int,
    greedy: bool = True
) -> Dict[str, Any]:
    """
    Run inference on a single sample and compute metrics.
    
    Returns:
        Dict with predicted tokens, metrics, and timing.
    """
    backbone = model['backbone']
    controller = model['controller']
    graph_builder = model['graph_builder']
    
    start_time = time.time()
    
    # Prepare image
    image = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
    boxes = sample['boxes'].to(device)                # [N, 4]
    graph = sample['graph']
    
    # Backbone: extract embeddings + CLS token
    backbone_out = backbone(image, boxes)
    graph.node_embeddings = backbone_out['node_embeddings']  # [N, vis_dim]
    cls_token = backbone_out['cls_token'].squeeze(0)         # [vis_dim]
    
    # Move graph to device
    graph = graph.to(device)
    
    # Run controller (autoregressive)
    controller_out = controller.forward_inference(
        graph=graph,
        cls_token=cls_token,
        device=device,
        max_steps=max_steps,
        greedy=greedy,
    )
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Extract predictions
    pred_tokens = [t for t in controller_out.predicted_sequence if t != '<END>']
    
    # Compute metrics
    gt_tokens = sample['gt_tokens']
    
    metrics = compute_all_metrics(
        gt_tokens=gt_tokens,
        pred_tokens=pred_tokens,
        gt_centers=sample['gt_centers'],
        pred_centers=None,  # No separate detection eval here
    )
    
    metrics['inference_time_ms'] = inference_time
    metrics['num_steps'] = controller_out.num_steps
    metrics['num_detected'] = graph.num_nodes
    metrics['num_gt'] = sample['total_digits']
    
    return {
        'pred_tokens': pred_tokens,
        'pred_string': ''.join(pred_tokens),
        'gt_string': ''.join(gt_tokens),
        'metrics': metrics,
        'output_tokens': controller_out.state.output_tokens,
    }


# ==============================================================
# BATCH EVALUATION PER LENGTH
# ==============================================================

def evaluate_length(
    length: int,
    num_samples: int,
    model: Dict[str, Any],
    device: torch.device,
    img_size: int,
    radius: float,
    noise_sigma: float,
    max_chunk_size: int,
    base_seed: int,
    max_steps_multiplier: int,
    quiet: bool = False,
    visualize: bool = False,
    viz_dir: str = './eval_viz',
    viz_count: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate the model on a fixed sequence length.
    
    Returns:
        Dict with aggregated metrics and per-sample results.
    """
    aggregator = MetricsAggregator()
    per_sample_results = []
    total_time = 0.0
    
    max_steps = length * max_steps_multiplier
    
    for i in range(num_samples):
        seed = base_seed + length * 10000 + i
        
        # Generate sample
        sample = generate_eval_sample(
            total_digits=length,
            img_size=img_size,
            radius=radius,
            noise_sigma=noise_sigma,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )
        
        # Evaluate
        result = evaluate_single_sample(
            sample=sample,
            model=model,
            device=device,
            max_steps=max_steps,
            greedy=True,
        )
        
        aggregator.update(result['metrics'])
        total_time += result['metrics']['inference_time_ms']
        
        per_sample_results.append({
            'sample_idx': i,
            'seed': seed,
            'gt_string': result['gt_string'],
            'pred_string': result['pred_string'],
            'exact_match': result['metrics']['exact_match'],
            'digit_accuracy': result['metrics']['digit_accuracy'],
            'chunk_f1': result['metrics']['chunk_f1'],
            'inference_time_ms': result['metrics']['inference_time_ms'],
            'num_steps': result['metrics']['num_steps'],
        })
        
        if not quiet and (i + 1) % 10 == 0:
            summary = aggregator.summary()
            print(
                f"    [{i+1}/{num_samples}] "
                f"exact={summary.get('exact_match', 0):.3f} "
                f"digit={summary.get('digit_accuracy', 0):.3f} "
                f"chunk_f1={summary.get('chunk_f1', 0):.3f} "
                f"avg_time={total_time/(i+1):.0f}ms"
            )
        
        # Visualization
        if visualize and i < viz_count:
            _save_visualization(
                sample=sample,
                result=result,
                model=model,
                length=length,
                sample_idx=i,
                viz_dir=viz_dir,
            )
    
    summary = aggregator.summary()
    summary['avg_inference_time_ms'] = total_time / max(num_samples, 1)
    summary['sequence_length'] = length
    summary['num_samples'] = num_samples
    
    return {
        'summary': summary,
        'per_sample': per_sample_results,
    }


def _save_visualization(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    model: Dict[str, Any],
    length: int,
    sample_idx: int,
    viz_dir: str
) -> None:
    """Save visualization for a single evaluation sample."""
    try:
        from utils.viz import VisualizationSuite, denormalize_image
        
        os.makedirs(viz_dir, exist_ok=True)
        
        viz = VisualizationSuite(output_dir=viz_dir)
        tag = f"len{length}_sample{sample_idx}"
        
        viz.visualize_sample(
            image=sample['image'],
            graph=sample['graph'],
            output_tokens=result['output_tokens'],
            heatmap_target=sample['heatmap_target'],
            tag=tag,
        )
        
        # Save prediction text
        pred_path = os.path.join(viz_dir, f"{tag}_prediction.txt")
        with open(pred_path, 'w') as f:
            f.write(f"GT:   {result['gt_string']}\n")
            f.write(f"PRED: {result['pred_string']}\n")
            f.write(f"Exact match: {result['metrics']['exact_match']}\n")
            f.write(f"Digit acc:   {result['metrics']['digit_accuracy']:.4f}\n")
            f.write(f"Chunk F1:    {result['metrics']['chunk_f1']:.4f}\n")
            f.write(f"Steps:       {result['metrics']['num_steps']}\n")
            f.write(f"Time:        {result['metrics']['inference_time_ms']:.1f}ms\n")
    
    except Exception as e:
        print(f"    [Viz] Failed for sample {sample_idx}: {e}")


# ==============================================================
# MAIN EVALUATION
# ==============================================================

def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"  Cognitive Reader — OOD Evaluation")
    print(f"{'='*70}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {args.device}")
    print(f"  Lengths:     {args.lengths}")
    print(f"  Samples:     {args.num_samples} per length")
    print(f"  Seed:        {args.seed}")
    print(f"  Output:      {args.output}")
    print(f"  Visualize:   {args.visualize}")
    print(f"{'='*70}\n")
    
    device = torch.device(args.device)
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        radius=args.radius,
        noise_sigma=args.noise_sigma,
        r_infer_multiplier=args.r_infer_multiplier,
    )
    
    # Initialize logger
    logger = TrainingLogger(LoggerConfig(
        project_name='cognitive_reader',
        experiment_name=f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        use_console=not args.quiet,
        use_file=True,
        use_tensorboard=False,
        use_wandb=False,
        log_dir=args.log_dir,
    ))
    
    logger.log_config(vars(args))
    
    # Evaluate each length
    results_by_length: Dict[int, Dict] = {}
    all_per_sample: Dict[int, List] = {}
    total_eval_start = time.time()
    
    for length in sorted(args.lengths):
        print(f"\n{'─'*50}")
        print(f"  Evaluating length = {length}")
        print(f"{'─'*50}")
        
        length_result = evaluate_length(
            length=length,
            num_samples=args.num_samples,
            model=model,
            device=device,
            img_size=args.img_size,
            radius=args.radius,
            noise_sigma=args.noise_sigma,
            max_chunk_size=args.max_chunk_size,
            base_seed=args.seed,
            max_steps_multiplier=args.max_steps_multiplier,
            quiet=args.quiet,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            viz_count=args.viz_per_length,
        )
        
        results_by_length[length] = length_result['summary']
        all_per_sample[length] = length_result['per_sample']
        
        # Print summary for this length
        s = length_result['summary']
        print(f"\n  Length {length} Results:")
        print(f"    Exact match:     {s.get('exact_match', 0):.4f}")
        print(f"    Digit accuracy:  {s.get('digit_accuracy', 0):.4f}")
        print(f"    Chunk F1:        {s.get('chunk_f1', 0):.4f}")
        print(f"    Avg time:        {s.get('avg_inference_time_ms', 0):.1f}ms")
        print(f"    Avg steps:       {s.get('num_steps', 0):.1f}")
        
        # Log to logger
        logger.log_scalars(
            {f"len_{length}/{k}": v for k, v in s.items() if isinstance(v, (int, float))},
            step=length,
            prefix='ood'
        )
    
    total_eval_time = time.time() - total_eval_start
    
    # OOD generalization analysis
    print(f"\n{'='*70}")
    print(f"  OOD Generalization Analysis")
    print(f"{'='*70}")
    
    ood_analysis = ood_generalization_analysis(results_by_length)
    
    print(f"  Lengths tested:        {ood_analysis['lengths']}")
    print(f"  Sequence accuracies:   {[f'{a:.3f}' for a in ood_analysis['sequence_accuracies']]}")
    print(f"  Digit accuracies:      {[f'{a:.3f}' for a in ood_analysis['digit_accuracies']]}")
    print(f"  Chunk F1s:             {[f'{f:.3f}' for f in ood_analysis['chunk_f1s']]}")
    print(f"  Seq degradation rate:  {ood_analysis['seq_degradation_rate']:.4f}")
    print(f"  Digit degradation rate:{ood_analysis['digit_degradation_rate']:.4f}")
    print(f"  Critical length:       {ood_analysis['critical_length']}")
    print(f"  Accuracy at max len:   {ood_analysis['accuracy_at_max_length']:.4f}")
    print(f"  Total eval time:       {total_eval_time:.1f}s")
    
    # Log OOD results
    logger.log_ood_results(results_by_length)
    
    # Save results
    output_data = {
        'metadata': {
            'checkpoint': args.checkpoint,
            'model_epoch': model['epoch'],
            'device': args.device,
            'timestamp': datetime.now().isoformat(),
            'total_eval_time_s': total_eval_time,
            'args': vars(args),
        },
        'results_by_length': {
            str(k): v for k, v in results_by_length.items()
        },
        'ood_analysis': ood_analysis,
        'per_sample': {
            str(k): v for k, v in all_per_sample.items()
        },
    }
    
    # Remove non-serializable items from per-sample
    for length_key in output_data['per_sample']:
        for sample in output_data['per_sample'][length_key]:
            for key in list(sample.keys()):
                if isinstance(sample[key], (np.ndarray, torch.Tensor)):
                    del sample[key]
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {args.output}")
    
    if args.visualize:
        print(f"  Visualizations saved to: {args.viz_dir}")
    
    # Print final summary table
    print(f"\n{'='*70}")
    print(f"  Final Summary")
    print(f"{'='*70}")
    print(f"  {'Length':>8s} {'Exact':>8s} {'Digit':>8s} {'Chunk F1':>10s} {'Time(ms)':>10s}")
    print(f"  {'─'*48}")
    for length in sorted(results_by_length.keys()):
        s = results_by_length[length]
        print(
            f"  {length:>8d} "
            f"{s.get('exact_match', 0):>8.4f} "
            f"{s.get('digit_accuracy', 0):>8.4f} "
            f"{s.get('chunk_f1', 0):>10.4f} "
            f"{s.get('avg_inference_time_ms', 0):>10.1f}"
        )
    print(f"{'='*70}\n")
    
    logger.close()


if __name__ == "__main__":
    main()