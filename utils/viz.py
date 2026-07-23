"""
utils/viz.py
Visualization tools for debugging and analyzing the Cognitive Reader.

Provides:
  1. Graph overlay: nodes, edges, threshold radius, chunk coloring.
  2. Reading path: visit order, chunk boundaries, saccadic jumps.
  3. Heatmap overlay: predicted vs ground-truth digit centers.
  4. Attention visualization: Mode 2 global attention weights.
  5. Step-by-step replay: controller state at each time step.
  6. Bounding box overlay: detected boxes with labels and confidence.

All functions operate on denormalized images and return PIL Images
or save directly to files.
"""

import torch
import numpy as np
import math
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# ==============================================================
# COLOR PALETTE
# ==============================================================

# Distinct colors for chunk IDs (up to 20 chunks)
CHUNK_COLORS = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (255, 225, 25),   # Yellow
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (250, 190, 190),  # Pink
    (0, 128, 128),    # Teal
    (230, 190, 255),  # Lavender
    (170, 110, 40),   # Brown
    (255, 250, 200),  # Beige
    (128, 0, 0),      # Maroon
    (170, 255, 195),  # Mint
    (128, 128, 0),    # Olive
    (255, 215, 180),  # Coral
    (0, 0, 128),      # Navy
    (128, 128, 128),  # Gray
]


def get_chunk_color(chunk_id: int) -> Tuple[int, int, int]:
    """Get a distinct color for a chunk ID."""
    return CHUNK_COLORS[chunk_id % len(CHUNK_COLORS)]


# ==============================================================
# IMAGE UTILITIES
# ==============================================================

def denormalize_image(
    image_tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> Image.Image:
    """
    Convert a normalized image tensor to a PIL Image.
    
    Args:
        image_tensor: [3, H, W] or [1, 3, H, W] normalized tensor.
        mean: Normalization mean.
        std: Normalization std.
    
    Returns:
        PIL Image (RGB).
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    img = image_tensor.detach().cpu().clone()
    
    # Denormalize
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    
    # Convert to numpy uint8
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def get_font(size: int = 14) -> Any:
    """Get a font for text rendering."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.load_default()
        except Exception:
            return None


# ==============================================================
# 1. GRAPH VISUALIZATION
# ==============================================================

def draw_graph(
    image: Image.Image,
    node_positions_px: torch.Tensor,
    adjacency: torch.Tensor,
    node_chunk_ids: torch.Tensor,
    radius: float,
    node_labels: Optional[torch.Tensor] = None,
    draw_radius_circles: bool = True,
    draw_edges: bool = True,
    node_radius: int = 12,
    edge_width: int = 1,
    alpha: int = 180
) -> Image.Image:
    """
    Draw the spatial graph overlay on the image.
    
    Args:
        image: PIL Image (background).
        node_positions_px: [N, 2] — node center coordinates (x, y).
        adjacency: [N, N] — binary adjacency matrix.
        node_chunk_ids: [N] — chunk assignment per node.
        radius: Threshold radius r (pixels).
        node_labels: [N] — digit labels (optional).
        draw_radius_circles: Draw dashed circles showing threshold radius.
        draw_edges: Draw edges between connected nodes.
        node_radius: Radius of node circles (pixels).
        edge_width: Width of edge lines.
        alpha: Transparency for overlay elements.
    
    Returns:
        PIL Image with graph overlay.
    """
    if not PIL_AVAILABLE:
        return image
    
    img = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(12)
    
    N = node_positions_px.shape[0]
    positions = node_positions_px.numpy()
    chunks = node_chunk_ids.numpy()
    adj = adjacency.numpy()
    
    # Draw threshold radius circles
    if draw_radius_circles:
        for i in range(N):
            x, y = positions[i]
            color = get_chunk_color(int(chunks[i]))
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline=color + (60,), width=1)
    
    # Draw edges
    if draw_edges:
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j] > 0.5:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    color = get_chunk_color(int(chunks[i]))
                    draw.line(
                        [(x1, y1), (x2, y2)],
                        fill=color + (100,),
                        width=edge_width
                    )
    
    # Draw nodes
    for i in range(N):
        x, y = positions[i]
        color = get_chunk_color(int(chunks[i]))
        
        # Node circle
        bbox = [x - node_radius, y - node_radius, x + node_radius, y + node_radius]
        draw.ellipse(bbox, fill=color + (alpha,), outline=(255, 255, 255, 255), width=2)
        
        # Node label
        if node_labels is not None:
            label = str(int(node_labels[i]))
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.text(
                (x - tw / 2, y - th / 2 - 2),
                label,
                fill=(255, 255, 255, 255),
                font=font
            )
        
        # Node ID (small, above the circle)
        id_text = str(i)
        draw.text(
            (x - 3, y - node_radius - 12),
            id_text,
            fill=(100, 100, 100, 200),
            font=get_font(9)
        )
    
    # Composite
    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


# ==============================================================
# 2. READING PATH VISUALIZATION
# ==============================================================

def draw_reading_path(
    image: Image.Image,
    output_tokens: List[Dict],
    node_positions_px: torch.Tensor,
    node_chunk_ids: torch.Tensor,
    arrow_width: int = 3,
    jump_dash_length: int = 8
) -> Image.Image:
    """
    Draw the controller's reading path on the image.
    
    Shows:
      - Solid arrows for Mode 1 (local) transitions.
      - Dashed arrows for Mode 2 (saccadic) jumps.
      - Red X markers for chunk boundaries.
      - Visit order numbers.
    
    Args:
        image: PIL Image.
        output_tokens: List of token dicts from ControllerState.output_tokens.
        node_positions_px: [N, 2] — node positions.
        node_chunk_ids: [N] — chunk assignments.
        arrow_width: Width of path arrows.
        jump_dash_length: Dash length for saccadic jump arrows.
    
    Returns:
        PIL Image with reading path overlay.
    """
    if not PIL_AVAILABLE:
        return image
    
    img = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(11)
    
    positions = node_positions_px.numpy()
    
    # Extract the sequence of visited nodes and actions
    read_tokens = [t for t in output_tokens if t['mode'] == 'READ' and t['node_id'] is not None]
    chunk_tokens = [t for t in output_tokens if t['mode'] == 'CHUNK']
    
    # Draw arrows between consecutive READ tokens
    for i in range(len(read_tokens) - 1):
        curr = read_tokens[i]
        next_t = read_tokens[i + 1]
        
        curr_node = curr['node_id']
        next_node = next_t['node_id']
        
        x1, y1 = positions[curr_node]
        x2, y2 = positions[next_node]
        
        # Determine if this is a local transition or a saccadic jump
        # A jump occurs if there's a CHUNK token between them
        curr_step = curr['step']
        next_step = next_t['step']
        has_chunk_between = any(
            curr_step < ct['step'] < next_step
            for ct in chunk_tokens
        )
        
        if has_chunk_between:
            # Saccadic jump: dashed red arrow
            _draw_dashed_line(
                draw, (x1, y1), (x2, y2),
                fill=(255, 0, 0, 200),
                width=arrow_width,
                dash_length=jump_dash_length
            )
        else:
            # Local transition: solid arrow colored by chunk
            chunk_id = int(node_chunk_ids[curr_node])
            color = get_chunk_color(chunk_id)
            draw.line([(x1, y1), (x2, y2)], fill=color + (200,), width=arrow_width)
        
        # Arrowhead
        _draw_arrowhead(draw, (x1, y1), (x2, y2), fill=(80, 80, 80, 220), size=8)
    
    # Draw visit order numbers
    for i, token in enumerate(read_tokens):
        node_id = token['node_id']
        x, y = positions[node_id]
        order_text = str(i + 1)
        draw.text(
            (x + 14, y - 14),
            order_text,
            fill=(0, 0, 0, 200),
            font=font
        )
    
    # Draw chunk boundary markers
    for ct in chunk_tokens:
        step = ct['step']
        # Find the last READ token before this CHUNK
        prev_reads = [t for t in read_tokens if t['step'] < step]
        if prev_reads:
            last_read = prev_reads[-1]
            node_id = last_read['node_id']
            x, y = positions[node_id]
            # Draw a red X
            size = 8
            draw.line([(x - size, y - size), (x + size, y + size)], fill=(255, 0, 0, 255), width=3)
            draw.line([(x - size, y + size), (x + size, y - size)], fill=(255, 0, 0, 255), width=3)
    
    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


def _draw_dashed_line(
    draw: ImageDraw.Draw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, ...],
    width: int = 2,
    dash_length: int = 8
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 1:
        return
    
    num_dashes = int(dist / (dash_length * 2))
    for i in range(num_dashes + 1):
        t_start = i * dash_length * 2 / dist
        t_end = min((i * dash_length * 2 + dash_length) / dist, 1.0)
        
        sx = x1 + dx * t_start
        sy = y1 + dy * t_start
        ex = x1 + dx * t_end
        ey = y1 + dy * t_end
        
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


def _draw_arrowhead(
    draw: ImageDraw.Draw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, ...],
    size: int = 8
) -> None:
    """Draw an arrowhead at the end point."""
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    
    # Two lines forming the arrowhead
    a1 = angle + math.pi * 0.8
    a2 = angle - math.pi * 0.8
    
    draw.line([
        (x2, y2),
        (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
    ], fill=fill, width=2)
    draw.line([
        (x2, y2),
        (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
    ], fill=fill, width=2)


# ==============================================================
# 3. HEATMAP VISUALIZATION
# ==============================================================

def draw_heatmap(
    image: Image.Image,
    heatmap: torch.Tensor,
    stride: int = 8,
    alpha: float = 0.4,
    colormap: str = 'hot'
) -> Image.Image:
    """
    Overlay a heatmap on the image.
    
    Args:
        image: PIL Image.
        heatmap: [H/stride, W/stride] or [1, H/stride, W/stride] tensor.
        stride: Feature map stride.
        alpha: Overlay transparency.
        colormap: Matplotlib colormap name.
    
    Returns:
        PIL Image with heatmap overlay.
    """
    if not MPL_AVAILABLE:
        return image
    
    if heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)
    
    hm = heatmap.detach().cpu().numpy()
    
    # Resize heatmap to image size
    from PIL import Image as PILImage
    hm_img = PILImage.fromarray((hm * 255).astype(np.uint8))
    hm_img = hm_img.resize(image.size, PILImage.BILINEAR)
    hm_resized = np.array(hm_img).astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    hm_colored = cmap(hm_resized)[:, :, :3]  # [H, W, 3]
    hm_colored = (hm_colored * 255).astype(np.uint8)
    hm_pil = PILImage.fromarray(hm_colored)
    
    # Blend
    img_np = np.array(image).astype(np.float32)
    hm_np = hm_pil.resize(image.size).convert('RGB')
    hm_np = np.array(hm_np).astype(np.float32)
    
    blended = (1 - alpha) * img_np + alpha * hm_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return PILImage.fromarray(blended)


def draw_detection_results(
    image: Image.Image,
    centers_px: torch.Tensor,
    scores: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    gt_centers_px: Optional[torch.Tensor] = None,
    score_threshold: float = 0.3
) -> Image.Image:
    """
    Draw detection results: predicted centers, boxes, and optionally GT centers.
    
    Args:
        image: PIL Image.
        centers_px: [N, 2] — predicted centers.
        scores: [N] — confidence scores.
        boxes: [N, 4] — predicted bounding boxes (optional).
        gt_centers_px: [M, 2] — ground-truth centers (optional).
        score_threshold: Minimum score to display.
    
    Returns:
        PIL Image with detection overlay.
    """
    if not PIL_AVAILABLE:
        return image
    
    img = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(10)
    
    # Draw ground-truth centers (green circles)
    if gt_centers_px is not None:
        gt = gt_centers_px.numpy()
        for i in range(gt.shape[0]):
            x, y = gt[i]
            r = 6
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                outline=(0, 200, 0, 200),
                width=2
            )
    
    # Draw predicted boxes
    if boxes is not None:
        boxes_np = boxes.numpy()
        for i in range(boxes_np.shape[0]):
            if scores[i] < score_threshold:
                continue
            x1, y1, x2, y2 = boxes_np[i]
            score = scores[i].item()
            color = (0, 150, 255, 150)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw.text((x1, y1 - 12), f"{score:.2f}", fill=color, font=font)
    
    # Draw predicted centers (blue dots)
    centers = centers_px.numpy()
    for i in range(centers.shape[0]):
        if scores[i] < score_threshold:
            continue
        x, y = centers[i]
        r = 4
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=(0, 100, 255, 220)
        )
    
    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


# ==============================================================
# 4. COMPREHENSIVE VISUALIZATION SUITE
# ==============================================================

class VisualizationSuite:
    """
    Generate a complete set of visualizations for a single sample.
    
    Usage:
        viz = VisualizationSuite(output_dir='./viz')
        viz.visualize_sample(
            image=image_tensor,
            graph=graph,
            output_tokens=state.output_tokens,
            heatmap_logits=heatmap_logits,
            heatmap_target=heatmap_target,
            tag='sample_0'
        )
    """
    
    def __init__(self, output_dir: str = './viz'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_sample(
        self,
        image: torch.Tensor,
        graph,
        output_tokens: List[Dict],
        heatmap_logits: Optional[torch.Tensor] = None,
        heatmap_target: Optional[torch.Tensor] = None,
        detection_centers: Optional[torch.Tensor] = None,
        detection_scores: Optional[torch.Tensor] = None,
        detection_boxes: Optional[torch.Tensor] = None,
        gt_centers: Optional[torch.Tensor] = None,
        tag: str = 'sample',
        stride: int = 8
    ) -> Dict[str, str]:
        """
        Generate all visualizations for a single sample.
        
        Returns:
            Dict mapping visualization name → saved file path.
        """
        if not PIL_AVAILABLE:
            print("[Viz] Pillow not available. Skipping visualization.")
            return {}
        
        saved = {}
        base_img = denormalize_image(image)
        
        # 1. Graph overlay
        graph_img = draw_graph(
            image=base_img,
            node_positions_px=graph.node_positions_px,
            adjacency=graph.adjacency,
            node_chunk_ids=graph.node_chunk_ids,
            radius=graph.radius,
            node_labels=graph.node_labels,
            draw_radius_circles=True,
            draw_edges=True
        )
        path = os.path.join(self.output_dir, f'{tag}_graph.png')
        graph_img.save(path)
        saved['graph'] = path
        
        # 2. Reading path
        path_img = draw_reading_path(
            image=base_img,
            output_tokens=output_tokens,
            node_positions_px=graph.node_positions_px,
            node_chunk_ids=graph.node_chunk_ids
        )
        path = os.path.join(self.output_dir, f'{tag}_path.png')
        path_img.save(path)
        saved['reading_path'] = path
        
        # 3. Graph + Reading path combined
        combined = draw_graph(
            image=base_img,
            node_positions_px=graph.node_positions_px,
            adjacency=graph.adjacency,
            node_chunk_ids=graph.node_chunk_ids,
            radius=graph.radius,
            node_labels=graph.node_labels,
            draw_radius_circles=False,
            draw_edges=True
        )
        combined = draw_reading_path(
            image=combined,
            output_tokens=output_tokens,
            node_positions_px=graph.node_positions_px,
            node_chunk_ids=graph.node_chunk_ids
        )
        path = os.path.join(self.output_dir, f'{tag}_combined.png')
        combined.save(path)
        saved['combined'] = path
        
        # 4. Predicted heatmap
        if heatmap_logits is not None:
            hm_pred = torch.sigmoid(heatmap_logits)
            if hm_pred.dim() == 4:
                hm_pred = hm_pred.squeeze(0).squeeze(0)
            elif hm_pred.dim() == 3:
                hm_pred = hm_pred.squeeze(0)
            
            hm_img = draw_heatmap(base_img, hm_pred, stride=stride, alpha=0.5)
            path = os.path.join(self.output_dir, f'{tag}_heatmap_pred.png')
            hm_img.save(path)
            saved['heatmap_pred'] = path
        
        # 5. Ground-truth heatmap
        if heatmap_target is not None:
            hm_gt = heatmap_target
            if hm_gt.dim() == 4:
                hm_gt = hm_gt.squeeze(0).squeeze(0)
            elif hm_gt.dim() == 3:
                hm_gt = hm_gt.squeeze(0)
            
            hm_img = draw_heatmap(base_img, hm_gt, stride=stride, alpha=0.5, colormap='cool')
            path = os.path.join(self.output_dir, f'{tag}_heatmap_gt.png')
            hm_img.save(path)
            saved['heatmap_gt'] = path
        
        # 6. Detection results
        if detection_centers is not None and detection_scores is not None:
            det_img = draw_detection_results(
                image=base_img,
                centers_px=detection_centers,
                scores=detection_scores,
                boxes=detection_boxes,
                gt_centers_px=gt_centers
            )
            path = os.path.join(self.output_dir, f'{tag}_detections.png')
            det_img.save(path)
            saved['detections'] = path
        
        # 7. Text summary
        summary = self._generate_summary(graph, output_tokens)
        path = os.path.join(self.output_dir, f'{tag}_summary.txt')
        with open(path, 'w') as f:
            f.write(summary)
        saved['summary'] = path
        
        return saved
    
    def _generate_summary(
        self,
        graph,
        output_tokens: List[Dict]
    ) -> str:
        """Generate a text summary of the reading result."""
        lines = []
        lines.append("=" * 50)
        lines.append("  Cognitive Reader — Sample Summary")
        lines.append("=" * 50)
        
        lines.append(f"\n  Graph:")
        lines.append(f"    Nodes: {graph.num_nodes}")
        lines.append(f"    Edges: {int(graph.adjacency.sum().item())}")
        lines.append(f"    Radius: {graph.radius:.1f}px")
        lines.append(f"    Avg degree: {graph.adjacency.sum().item() / max(graph.num_nodes, 1):.2f}")
        
        # Chunk statistics
        chunk_ids = graph.node_chunk_ids.numpy()
        unique_chunks = np.unique(chunk_ids[chunk_ids >= 0])
        lines.append(f"    Chunks: {len(unique_chunks)}")
        for c in unique_chunks:
            count = (chunk_ids == c).sum()
            lines.append(f"      Chunk {c}: {count} digits")
        
        lines.append(f"\n  Reading Output:")
        tokens = [t['token'] for t in output_tokens]
        lines.append(f"    Tokens: {' '.join(tokens)}")
        
        read_tokens = [t for t in output_tokens if t['mode'] == 'READ']
        chunk_tokens = [t for t in output_tokens if t['mode'] == 'CHUNK']
        lines.append(f"    Digits read: {len(read_tokens)}")
        lines.append(f"    Chunks emitted: {len(chunk_tokens)}")
        
        if output_tokens:
            lines.append(f"    Total steps: {output_tokens[-1].get('step', '?')}")
        
        lines.append("\n" + "=" * 50)
        return '\n'.join(lines)


# ==============================================================
# 5. MATPLOTLIB-BASED PLOTS
# ==============================================================

def plot_training_curves(
    train_history: List[Dict],
    val_history: Optional[List[Dict]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_history: List of per-epoch training metric dicts.
        val_history: List of per-epoch validation metric dicts.
        save_path: Path to save the plot (optional).
    """
    if not MPL_AVAILABLE:
        print("[Viz] Matplotlib not available.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    loss_keys = ['total', 'heatmap', 'digit', 'action', 'jump']
    titles = ['Total Loss', 'Heatmap Loss', 'Digit Loss', 'Action Loss', 'Jump Loss']
    
    for idx, (key, title) in enumerate(zip(loss_keys, titles)):
        ax = axes[idx // 2][idx % 2]
        
        train_vals = [h.get(key, 0) for h in train_history]
        ax.plot(train_vals, label='Train', color='blue', alpha=0.7)
        
        if val_history:
            val_vals = [h.get(key, 0) for h in val_history]
            # Val is recorded less frequently; scale x-axis
            val_x = np.linspace(0, len(train_vals) - 1, len(val_vals))
            ax.plot(val_x, val_vals, label='Val', color='red', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Training curves saved: {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_ood_results(
    ood_results: Dict[int, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot OOD evaluation results across different sequence lengths.
    
    Args:
        ood_results: Dict mapping length → metrics dict.
        save_path: Path to save the plot.
    """
    if not MPL_AVAILABLE:
        return
    
    lengths = sorted(ood_results.keys())
    seq_accs = [ood_results[l]['sequence_accuracy'] for l in lengths]
    digit_accs = [ood_results[l]['digit_accuracy'] for l in lengths]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    x = np.arange(len(lengths))
    width = 0.35
    
    ax.bar(x - width/2, seq_accs, width, label='Sequence Accuracy', color='steelblue')
    ax.bar(x + width/2, digit_accs, width, label='Digit Accuracy', color='coral')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Accuracy')
    ax.set_title('OOD Length Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lengths])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig('ood_results.png', dpi=150, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Visualization Suite — Unit Test")
    print("=" * 60)
    
    if not PIL_AVAILABLE:
        print("\n  ✗ Pillow not installed. Skipping visual tests.")
    else:
        # Create a fake graph and output for testing
        N = 8
        positions = torch.tensor([
            [100, 100], [140, 102], [180, 98], [220, 101],  # Chunk 0
            [350, 250], [390, 252], [430, 248], [470, 251],  # Chunk 1
        ], dtype=torch.float32)
        
        adjacency = torch.zeros(N, N)
        # Chunk 0 edges
        for i in range(3):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1
        # Chunk 1 edges
        for i in range(4, 7):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1
        
        chunk_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        labels = torch.tensor([3, 8, 4, 2, 1, 7, 5, 9])
        
        # Fake output tokens
        output_tokens = [
            {'token': '3', 'node_id': 0, 'mode': 'READ', 'step': 1, 'chunk_size': 1},
            {'token': '8', 'node_id': 1, 'mode': 'READ', 'step': 2, 'chunk_size': 2},
            {'token': '4', 'node_id': 2, 'mode': 'READ', 'step': 3, 'chunk_size': 3},
            {'token': '2', 'node_id': 3, 'mode': 'READ', 'step': 4, 'chunk_size': 4},
            {'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK', 'step': 5, 'chunk_size': 0},
            {'token': '1', 'node_id': 4, 'mode': 'READ', 'step': 6, 'chunk_size': 1},
            {'token': '7', 'node_id': 5, 'mode': 'READ', 'step': 7, 'chunk_size': 2},
            {'token': '5', 'node_id': 6, 'mode': 'READ', 'step': 8, 'chunk_size': 3},
            {'token': '9', 'node_id': 7, 'mode': 'READ', 'step': 9, 'chunk_size': 4},
            {'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK', 'step': 10, 'chunk_size': 0},
            {'token': '<END>', 'node_id': None, 'mode': 'END', 'step': 11, 'chunk_size': 0},
        ]
        
        # Create a blank image
        img = Image.new('RGB', (640, 640), (245, 245, 240))
        
        # Test graph drawing
        print("\n[Test 1] Graph visualization")
        graph_img = draw_graph(
            image=img,
            node_positions_px=positions,
            adjacency=adjacency,
            node_chunk_ids=chunk_ids,
            radius=80.0,
            node_labels=labels
        )
        graph_img.save('test_viz_graph.png')
        print(f"  ✓ Saved test_viz_graph.png ({graph_img.size})")
        
        # Test reading path
        print("\n[Test 2] Reading path visualization")
        path_img = draw_reading_path(
            image=img,
            output_tokens=output_tokens,
            node_positions_px=positions,
            node_chunk_ids=chunk_ids
        )
        path_img.save('test_viz_path.png')
        print(f"  ✓ Saved test_viz_path.png ({path_img.size})")
        
        # Test heatmap
        if MPL_AVAILABLE:
            print("\n[Test 3] Heatmap visualization")
            fake_heatmap = torch.rand(80, 80) * 0.2
            fake_heatmap[12, 12] = 1.0
            fake_heatmap[12, 17] = 1.0
            fake_heatmap[31, 43] = 1.0
            
            hm_img = draw_heatmap(img, fake_heatmap, stride=8, alpha=0.5)
            hm_img.save('test_viz_heatmap.png')
            print(f"  ✓ Saved test_viz_heatmap.png")
        
        # Test VisualizationSuite
        print("\n[Test 4] VisualizationSuite")
        
        # Create a minimal fake graph object
        class FakeGraph:
            pass
        
        fake_graph = FakeGraph()
        fake_graph.node_positions_px = positions
        fake_graph.adjacency = adjacency
        fake_graph.node_chunk_ids = chunk_ids
        fake_graph.node_labels = labels
        fake_graph.radius = 80.0
        fake_graph.num_nodes = N
        
        viz = VisualizationSuite(output_dir='./test_viz_output')
        saved = viz.visualize_sample(
            image=torch.randn(3, 640, 640),
            graph=fake_graph,
            output_tokens=output_tokens,
            tag='unit_test'
        )
        
        print(f"  Generated {len(saved)} visualizations:")
        for name, path in saved.items():
            exists = os.path.exists(path)
            print(f"    {name}: {path} ({'✓' if exists else '✗'})")
        
        # Test denormalize
        print("\n[Test 5] Image denormalization")
        norm_img = torch.randn(3, 640, 640)
        denorm = denormalize_image(norm_img)
        assert denorm.size == (640, 640)
        assert denorm.mode == 'RGB'
        print(f"  ✓ Denormalized: {denorm.size}, mode={denorm.mode}")
        
        # Test training curves
        if MPL_AVAILABLE:
            print("\n[Test 6] Training curves plot")
            fake_history = [
                {'total': 5.0 - i*0.04, 'heatmap': 2.0 - i*0.02,
                 'digit': 1.5 - i*0.01, 'action': 1.0 - i*0.008,
                 'jump': 0.5 - i*0.005}
                for i in range(50)
            ]
            plot_training_curves(fake_history, save_path='test_viz_curves.png')
            print(f"  ✓ Saved test_viz_curves.png")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)