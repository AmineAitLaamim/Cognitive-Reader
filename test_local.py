"""
test_local.py
Local testing script for the Cognitive Reader.

Generates synthetic samples, runs inference, computes metrics,
and optionally saves visualizations.

Usage:
    # Basic test (10 samples, 20 digits each)
    python test_local.py --checkpoint ./checkpoints/checkpoint_best.pt

    # Custom settings
    python test_local.py --checkpoint ./checkpoints/checkpoint_best.pt \
                         --num_samples 50 \
                         --digits 10 20 50 100 \
                         --visualize \
                         --output_dir ./test_results

    # Single sample with verbose output
    python test_local.py --checkpoint ./checkpoints/checkpoint_best.pt \
                         --digits 20 \
                         --num_samples 1 \
                         --verbose
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

# Project imports
from data.generator import ConstrainedPolarGenerator, GeneratorConfig
from data.renderer import DigitRenderer, RendererConfig
from models.backbone.cnn import VisualBackbone
from models.controller.dual_mode import DualModeController
from models.graph.builder import ThresholdRadiusGraphBuilder, SpatialGraph
from eval.metrics import compute_all_metrics


# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model(checkpoint_path: str, device: torch.device, radius: float = 80.0,
               noise_sigma: float = 3.0, r_infer_multiplier: float = 1.2) -> Dict[str, Any]:
    """Load all model components from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    r_train = radius
    r_infer = r_infer_multiplier * r_train
    T_intra = 0.8 * r_train + 4 * noise_sigma
    T_inter = 1.5 * r_train - 4 * noise_sigma

    # Build backbone
    backbone = VisualBackbone(
        vis_dim=512, roi_output_size=7,
        pretrained=False, enable_heatmap=True, padding_factor=1.2
    ).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.eval()

    # Build controller
    controller = DualModeController(
        vis_dim=512, hidden_dim=256, edge_dim=256, key_dim=256,
        num_classes=10, radius=r_train,
        T_intra=T_intra, T_inter=T_inter,
        num_frequencies=64, num_heads=4, dropout=0.0
    ).to(device)
    controller.load_state_dict(checkpoint['controller_state_dict'])
    controller.eval()

    # Graph builder with relaxed radius for inference
    graph_builder = ThresholdRadiusGraphBuilder(
        radius=r_infer, img_width=640, img_height=640
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
    }


# ==============================================================
# SAMPLE GENERATION
# ==============================================================

def generate_sample(total_digits: int, img_size: int = 640, radius: float = 80.0,
                    noise_sigma: float = 3.0, max_chunk_size: int = 4,
                    seed: int = 42) -> Dict[str, Any]:
    """Generate a single test sample with ground truth."""
    gen_config = GeneratorConfig(
        img_width=img_size, img_height=img_size,
        threshold_radius_r=radius, noise_sigma=noise_sigma,
        max_chunk_size=max_chunk_size, min_chunk_size=1,
    )
    generator = ConstrainedPolarGenerator(gen_config)
    layout = generator.generate_sample(total_digits)

    render_config = RendererConfig(
        img_width=img_size, img_height=img_size,
        rotation_max_deg=0.0, blur_probability=0.0, seed=seed,
    )
    renderer = DigitRenderer(render_config)
    render_output = renderer.render(layout)

    # Build graph with noisy boxes (matching inference conditions)
    graph_builder = ThresholdRadiusGraphBuilder(
        radius=radius, img_width=img_size, img_height=img_size
    )
    boxes_list = []
    labels_list = []
    chunk_ids_list = []
    for node in layout.nodes:
        boxes_list.append({
            'center_x': node.noisy_center_x,
            'center_y': node.noisy_center_y,
            'w': node.width, 'h': node.height,
            'node_id': node.node_id,
        })
        labels_list.append(node.label)
        chunk_ids_list.append(node.chunk_id)

    graph = graph_builder.build_from_boxes(boxes_list, labels_list, chunk_ids_list)

    gt_centers = np.array([[node.center_x, node.center_y] for node in layout.nodes])

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
# INFERENCE
# ==============================================================

@torch.no_grad()
def run_inference(sample: Dict[str, Any], model: Dict[str, Any],
                  device: torch.device, max_steps: int = 300,
                  greedy: bool = True) -> Dict[str, Any]:
    """Run inference on a single sample."""
    backbone = model['backbone']
    controller = model['controller']
    graph_builder = model['graph_builder']

    start_time = time.time()

    # Prepare image
    image = sample['image'].unsqueeze(0).to(device)
    boxes = sample['boxes'].to(device)

    # Backbone forward
    backbone_out = backbone(image, boxes)
    graph = sample['graph']
    graph.node_embeddings = backbone_out['node_embeddings']
    cls_token = backbone_out['cls_token'].squeeze(0)
    graph = graph.to(device)

    # Controller inference
    controller_out = controller.forward_inference(
        graph=graph, cls_token=cls_token, device=device,
        max_steps=max_steps, greedy=greedy,
    )

    inference_time = (time.time() - start_time) * 1000

    # Extract predictions
    pred_tokens = [t for t in controller_out.predicted_sequence if t != '<END>']

    # Compute metrics
    gt_tokens = sample['gt_tokens']
    metrics = compute_all_metrics(
        gt_tokens=gt_tokens, pred_tokens=pred_tokens,
        gt_centers=sample['gt_centers'], pred_centers=None,
    )
    metrics['inference_time_ms'] = inference_time
    metrics['num_steps'] = controller_out.num_steps

    return {
        'pred_tokens': pred_tokens,
        'pred_string': ''.join(pred_tokens),
        'gt_string': ''.join(gt_tokens),
        'metrics': metrics,
        'output_tokens': controller_out.state.output_tokens,
    }


# ==============================================================
# VISUALIZATION
# ==============================================================

def save_visualization(sample: Dict[str, Any], result: Dict[str, Any],
                       output_dir: str, tag: str) -> None:
    """Save visualization for a single sample."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        # Denormalize image
        img = sample['image']
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Image with bounding boxes
        axes[0].imshow(img_display)
        axes[0].set_title(f'Input ({sample["total_digits"]} digits)')
        axes[0].axis('off')

        for node in sample['graph'].node_positions_px:
            pass  # Boxes drawn below

        # Draw reading path
        output_tokens = result['output_tokens']
        positions = sample['graph'].node_positions_px.numpy()

        # Draw nodes
        for i, pos in enumerate(positions):
            axes[1].plot(pos[0], pos[1], 'ko', markersize=8)
            axes[1].annotate(str(i), (pos[0], pos[1]),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Draw reading path
        read_nodes = [t['node_id'] for t in output_tokens
                      if t.get('mode') == 'READ' and t.get('node_id') is not None]
        for j in range(len(read_nodes) - 1):
            n1, n2 = read_nodes[j], read_nodes[j + 1]
            if n1 < len(positions) and n2 < len(positions):
                axes[1].annotate('', xy=positions[n2], xytext=positions[n1],
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        axes[1].imshow(img_display, alpha=0.3)
        axes[1].set_title(f'Reading Path (steps={result["metrics"]["num_steps"]})')
        axes[1].axis('off')
        axes[1].set_xlim(0, 640)
        axes[1].set_ylim(640, 0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Save text result
        with open(os.path.join(output_dir, f'{tag}.txt'), 'w') as f:
            f.write(f"GT:   {result['gt_string']}\n")
            f.write(f"PRED: {result['pred_string']}\n")
            f.write(f"Exact match: {result['metrics']['exact_match']}\n")
            f.write(f"Digit acc:   {result['metrics']['digit_accuracy']:.4f}\n")
            f.write(f"Chunk F1:    {result['metrics']['chunk_f1']:.4f}\n")
            f.write(f"Steps:       {result['metrics']['num_steps']}\n")
            f.write(f"Time:        {result['metrics']['inference_time_ms']:.1f}ms\n")

    except Exception as e:
        print(f"  [Viz] Failed for {tag}: {e}")


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description='Local testing for Cognitive Reader')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--digits', type=int, nargs='+', default=[10, 20, 50],
                       help='Sequence lengths to test')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples per length')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--radius', type=float, default=80.0)
    parser.add_argument('--noise_sigma', type=float, default=3.0)
    parser.add_argument('--max_chunk_size', type=int, default=4)
    parser.add_argument('--r_infer_multiplier', type=float, default=1.2)
    parser.add_argument('--max_steps_multiplier', type=int, default=3)
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualizations')
    parser.add_argument('--output_dir', type=str, default='./test_results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print token-by-token output')
    parser.add_argument('--save_json', action='store_true',
                       help='Save results to JSON')
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Cognitive Reader — Local Testing")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Lengths:    {args.digits}")
    print(f"  Samples:    {args.num_samples} per length")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        radius=args.radius,
        noise_sigma=args.noise_sigma,
        r_infer_multiplier=args.r_infer_multiplier,
    )

    # Test each length
    all_results = {}

    for num_digits in args.digits:
        print(f"\n{'─'*50}")
        print(f"  Testing {num_digits} digits ({args.num_samples} samples)")
        print(f"{'─'*50}")

        length_results = []
        total_time = 0.0

        for i in range(args.num_samples):
            seed = args.seed + num_digits * 10000 + i

            # Generate sample
            sample = generate_sample(
                total_digits=num_digits,
                img_size=args.img_size,
                radius=args.radius,
                noise_sigma=args.noise_sigma,
                max_chunk_size=args.max_chunk_size,
                seed=seed,
            )

            # Run inference
            max_steps = num_digits * args.max_steps_multiplier
            result = run_inference(sample, model, device, max_steps=max_steps)
            total_time += result['metrics']['inference_time_ms']
            length_results.append(result)

            # Verbose output
            if args.verbose:
                print(f"\n  Sample {i+1}/{args.num_samples} (seed={seed})")
                print(f"    GT:   {result['gt_string']}")
                print(f"    PRED: {result['pred_string']}")
                print(f"    Exact: {result['metrics']['exact_match']}  "
                      f"Digit: {result['metrics']['digit_accuracy']:.3f}  "
                      f"Chunk F1: {result['metrics']['chunk_f1']:.3f}  "
                      f"Steps: {result['metrics']['num_steps']}  "
                      f"Time: {result['metrics']['inference_time_ms']:.0f}ms")

                if args.verbose and num_digits <= 30:
                    print(f"    Tokens:")
                    for t in result['output_tokens']:
                        print(f"      step={t.get('step','?'):>3}  "
                              f"mode={t.get('mode','?'):>6}  "
                              f"token={t.get('token','?')}")

            # Visualization
            if args.visualize:
                tag = f"len{num_digits}_sample{i}"
                save_visualization(sample, result, args.output_dir, tag)

            # Progress
            if not args.verbose and (i + 1) % 5 == 0:
                avg_digit = np.mean([r['metrics']['digit_accuracy'] for r in length_results])
                avg_chunk = np.mean([r['metrics']['chunk_f1'] for r in length_results])
                print(f"    [{i+1}/{args.num_samples}] "
                      f"digit={avg_digit:.3f} chunk_f1={avg_chunk:.3f}")

        # Summary for this length
        avg_metrics = {
            'exact_match': np.mean([r['metrics']['exact_match'] for r in length_results]),
            'digit_accuracy': np.mean([r['metrics']['digit_accuracy'] for r in length_results]),
            'chunk_f1': np.mean([r['metrics']['chunk_f1'] for r in length_results]),
            'avg_steps': np.mean([r['metrics']['num_steps'] for r in length_results]),
            'avg_time_ms': total_time / args.num_samples,
            'num_samples': args.num_samples,
        }
        all_results[num_digits] = avg_metrics

        print(f"\n  Length {num_digits} Summary:")
        print(f"    Exact match:  {avg_metrics['exact_match']:.4f}")
        print(f"    Digit acc:    {avg_metrics['digit_accuracy']:.4f}")
        print(f"    Chunk F1:     {avg_metrics['chunk_f1']:.4f}")
        print(f"    Avg steps:    {avg_metrics['avg_steps']:.1f}")
        print(f"    Avg time:     {avg_metrics['avg_time_ms']:.1f}ms")

    # Final summary table
    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}")
    print(f"  {'Length':>8s} {'Exact':>8s} {'Digit':>8s} {'Chunk F1':>10s} {'Steps':>8s} {'Time':>8s}")
    print(f"  {'─'*54}")
    for length in sorted(all_results.keys()):
        m = all_results[length]
        print(f"  {length:>8d} {m['exact_match']:>8.4f} {m['digit_accuracy']:>8.4f} "
              f"{m['chunk_f1']:>10.4f} {m['avg_steps']:>8.1f} {m['avg_time_ms']:>7.1f}ms")
    print(f"{'='*60}")

    # Save JSON
    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'test_results.json')
        output_data = {
            'metadata': {
                'checkpoint': args.checkpoint,
                'model_epoch': model['epoch'],
                'device': str(device),
                'timestamp': datetime.now().isoformat(),
                'args': vars(args),
            },
            'results': {str(k): v for k, v in all_results.items()},
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    if args.visualize:
        print(f"  Visualizations saved to: {args.output_dir}/")

    print()


if __name__ == "__main__":
    main()