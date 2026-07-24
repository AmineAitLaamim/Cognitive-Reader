"""
visualize.py
Generate one sample, run inference, show 3 visualizations.

Usage:
    uv run visualize.py --checkpoint ./checkpoints/checkpoint_best.pt --digits 20
    uv run visualize.py --checkpoint ./checkpoints/checkpoint_best.pt --digits 50 --seed 123
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from data.generator import ConstrainedPolarGenerator, GeneratorConfig
from data.renderer import DigitRenderer, RendererConfig
from models.backbone.cnn import VisualBackbone
from models.controller.dual_mode import DualModeController
from models.graph.builder import ThresholdRadiusGraphBuilder


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    backbone = VisualBackbone(
        vis_dim=512, roi_output_size=7,
        pretrained=False, enable_heatmap=True, padding_factor=1.2
    ).to(device)
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    backbone.eval()

    controller = DualModeController(
        vis_dim=512, hidden_dim=256, edge_dim=256, key_dim=256,
        num_classes=10, radius=80.0,
        T_intra=0.8 * 80.0 + 4 * 3.0,
        T_inter=1.5 * 80.0 - 4 * 3.0,
        num_frequencies=64, num_heads=4, dropout=0.0
    ).to(device)
    controller.load_state_dict(ckpt['controller_state_dict'])
    controller.eval()

    graph_builder = ThresholdRadiusGraphBuilder(
        radius=96.0, img_width=640, img_height=640
    )

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    return backbone, controller, graph_builder


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--digits', type=int, default=20, help='Number of digits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Digits: {args.digits}")
    print(f"Seed:   {args.seed}\n")

    # ── Load model ──
    backbone, controller, graph_builder = load_model(args.checkpoint, device)

    # ── Generate sample ──
    gen_config = GeneratorConfig(
        img_width=640, img_height=640,
        threshold_radius_r=80.0, noise_sigma=3.0,
        max_chunk_size=4, min_chunk_size=1,
    )
    generator = ConstrainedPolarGenerator(gen_config)
    layout = generator.generate_sample(args.digits)

    render_config = RendererConfig(img_width=640, img_height=640, seed=args.seed)
    renderer = DigitRenderer(render_config)
    render_out = renderer.render(layout)

    # ── Build graph ──
    boxes_list, labels_list, chunk_ids_list = [], [], []
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

    # ── Run inference ──
    image = render_out['image'].unsqueeze(0).to(device)
    boxes = render_out['boxes'].to(device)

    t0 = time.time()
    with torch.no_grad():
        bb_out = backbone(image, boxes)
        graph.node_embeddings = bb_out['node_embeddings']
        cls_token = bb_out['cls_token'].squeeze(0)
        heatmap_logits = bb_out.get('heatmap_logits')
        graph = graph.to(device)

        ctrl_out = controller.forward_inference(
            graph=graph, cls_token=cls_token, device=device,
            max_steps=args.digits * 3, greedy=True,
        )
    elapsed = (time.time() - t0) * 1000

    pred_tokens = [t for t in ctrl_out.predicted_sequence if t != '<END>']
    gt_tokens = [t['token'] for t in layout.gt_sequence]
    gt_str = ''.join(gt_tokens)
    pred_str = ''.join(pred_tokens)

    # ── Prepare display data ──
    img_display = denormalize(render_out['image'])
    positions = graph.node_positions_px.cpu().numpy()
    adjacency = graph.adjacency.cpu().numpy()
    chunk_ids = graph.node_chunk_ids.cpu().numpy()
    output_tokens = ctrl_out.state.output_tokens

    # Heatmap
    if heatmap_logits is not None:
        hm = torch.sigmoid(heatmap_logits[0, 0]).cpu().numpy()
    else:
        hm = render_out['heatmap_target'].squeeze().numpy()

    # GT heatmap target
    hm_target = render_out['heatmap_target'].squeeze().numpy()

    # Reading path nodes
    read_nodes = [t['node_id'] for t in output_tokens
                  if t.get('mode') == 'READ' and t.get('node_id') is not None]

    # Chunk colors
    unique_chunks = np.unique(chunk_ids)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(unique_chunks), 1)))
    chunk_color = {c: cmap[i % len(cmap)] for i, c in enumerate(unique_chunks)}

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(
        f'Cognitive Reader — {args.digits} digits (seed={args.seed})\n'
        f'GT: {gt_str[:80]}{"..." if len(gt_str) > 80 else ""}\n'
        f'PRED: {pred_str[:80]}{"..." if len(pred_str) > 80 else ""}',
        fontsize=11, fontfamily='monospace', y=0.98
    )

    # ── Row 1, Col 1: Generated image with bounding boxes ──
    ax = axes[0][0]
    ax.imshow(img_display)
    ax.set_title('Generated Image + Bounding Boxes', fontsize=12, fontweight='bold')
    ax.axis('off')
    for node in layout.nodes:
        x, y = node.center_x, node.center_y
        w, h = node.width, node.height
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - h/2 - 4, node.label, color='lime',
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6))

    # ── Row 1, Col 2: Heatmap target (GT) ──
    ax = axes[0][1]
    ax.imshow(img_display, alpha=0.3)
    ax.imshow(hm_target, cmap='hot', alpha=0.7)
    ax.set_title('Heatmap Target (GT)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # ── Row 1, Col 3: Predicted heatmap ──
    ax = axes[0][2]
    ax.imshow(img_display, alpha=0.3)
    ax.imshow(hm, cmap='hot', alpha=0.7)
    ax.set_title('Predicted Heatmap', fontsize=12, fontweight='bold')
    ax.axis('off')

    # ── Row 2, Col 1: Spatial graph ──
    ax = axes[1][0]
    ax.imshow(img_display, alpha=0.3)
    ax.set_title('Spatial Graph (edges + chunks)', fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(0, 640)
    ax.set_ylim(640, 0)

    # Edges
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if adjacency[i][j] > 0:
                ax.plot([positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        'b-', alpha=0.3, linewidth=1)

    # Nodes colored by chunk
    for i, pos in enumerate(positions):
        color = chunk_color.get(chunk_ids[i], 'gray')
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=16,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(str(i), (pos[0], pos[1]),
                    ha='center', va='center', fontsize=7, fontweight='bold')

    # ── Row 2, Col 2: Reading path ──
    ax = axes[1][1]
    ax.imshow(img_display, alpha=0.25)
    ax.set_title(f'Reading Path ({len(read_nodes)} nodes visited)', fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(0, 640)
    ax.set_ylim(640, 0)

    # Path arrows
    for j in range(len(read_nodes) - 1):
        n1, n2 = read_nodes[j], read_nodes[j + 1]
        if n1 < len(positions) and n2 < len(positions):
            ax.annotate('',
                xy=positions[n2], xytext=positions[n1],
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                connectionstyle='arc3,rad=0.15'))

    # Nodes with reading order
    for order, node_id in enumerate(read_nodes):
        if node_id < len(positions):
            ax.plot(positions[node_id][0], positions[node_id][1],
                    'o', color='red', markersize=18,
                    markeredgecolor='darkred', markeredgewidth=2)
            ax.annotate(str(order + 1),
                (positions[node_id][0], positions[node_id][1]),
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Unvisited nodes
    visited_set = set(read_nodes)
    for i, pos in enumerate(positions):
        if i not in visited_set:
            ax.plot(pos[0], pos[1], 'o', color='gray', markersize=10,
                    alpha=0.4, markeredgecolor='black', markeredgewidth=1)

    # ── Row 2, Col 3: Token-by-token trace ──
    ax = axes[1][2]
    ax.axis('off')
    ax.set_title('Token-by-Token Trace', fontsize=12, fontweight='bold')

    trace_lines = []
    trace_lines.append(f"{'Step':>4}  {'Mode':>6}  {'Token':>8}  {'Node':>5}")
    trace_lines.append("─" * 35)
    for t in output_tokens:
        step = t.get('step', '?')
        mode = t.get('mode', '?')
        token = t.get('token', '?')
        node = t.get('node_id', '-')
        node_str = str(node) if node is not None else '-'
        trace_lines.append(f"{step:>4}  {mode:>6}  {token:>8}  {node_str:>5}")
    trace_lines.append("─" * 35)
    trace_lines.append(f"Total tokens: {len(output_tokens)}")
    trace_lines.append(f"Digits read:  {len(read_nodes)}")
    trace_lines.append(f"Time:         {elapsed:.1f}ms")
    trace_lines.append("")
    trace_lines.append(f"GT:   {gt_str[:50]}")
    trace_lines.append(f"PRED: {pred_str[:50]}")

    # Digit accuracy
    gt_digits = [t for t in gt_tokens if t != '<CHUNK>']
    pred_digits = [t for t in pred_tokens if t != '<CHUNK>']
    correct = sum(1 for a, b in zip(gt_digits, pred_digits) if a == b)
    total = len(gt_digits)
    trace_lines.append(f"Digit acc: {correct}/{total} = {correct/max(total,1):.3f}")

    trace_text = '\n'.join(trace_lines)
    ax.text(0.05, 0.95, trace_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'viz_digits{args.digits}_seed{args.seed}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: viz_digits{args.digits}_seed{args.seed}.png")
    plt.show()

    # ── Console summary ──
    print(f"\n{'='*50}")
    print(f"  GT:   {gt_str}")
    print(f"  PRED: {pred_str}")
    print(f"  Digit acc: {correct}/{total} = {correct/max(total,1):.3f}")
    print(f"  Nodes visited: {len(read_nodes)}/{len(positions)}")
    print(f"  Time: {elapsed:.1f}ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()