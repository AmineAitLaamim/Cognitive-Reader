"""
data/collate.py
Custom collate function for batching variable-sized spatial graphs.

THE CORE PROBLEM:
  Standard PyTorch DataLoaders stack tensors into fixed-size batches.
  But our graphs have a variable number of nodes N per image.
  The adjacency matrix is N×N, edge features are N×N×F, etc.
  
  Naive zero-padding creates "phantom nodes" at position (0, 0).
  If a real node is within radius r of (0, 0), the graph builder
  would create a fake edge to the phantom node. The controller
  would then try to route to a node that doesn't exist.

THE SOLUTION:
  1. Pad all node tensors to N_max with zeros.
  2. Pad all N×N tensors (adjacency, edges) to N_max×N_max with zeros.
  3. Create a node_mask [B, N_max]: 1 = real node, 0 = padding.
  4. Padded nodes have adjacency = 0 (no edges), labels = -1 (ignored),
     and positions = (-9999, -9999) (infinitely far from any real node).
  5. The controller uses node_mask to ignore padded nodes.
  6. Boxes are concatenated (not padded) with batch indices for RoI Align.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BatchedSample:
    """
    A batched sample ready for the training loop.
    
    Contains both batched tensors (for the backbone) and
    per-sample data (for the controller).
    """
    # === Batched tensors (standard stacking) ===
    images: torch.Tensor              # [B, 3, H, W]
    heatmap_targets: torch.Tensor     # [B, 1, H/8, W/8]
    
    # === Concatenated boxes (for RoI Align) ===
    boxes: torch.Tensor               # [N_total, 4] — (x1, y1, x2, y2)
    box_batch_indices: torch.Tensor   # [N_total] — which image each box belongs to
    
    # === Padded graph tensors ===
    node_positions_norm: torch.Tensor  # [B, N_max, 2]
    node_positions_px: torch.Tensor    # [B, N_max, 2]
    node_labels: torch.Tensor          # [B, N_max]
    node_chunk_ids: torch.Tensor       # [B, N_max]
    adjacency: torch.Tensor            # [B, N_max, N_max]
    edge_features: torch.Tensor        # [B, N_max, N_max, 3]
    edge_directions: torch.Tensor      # [B, N_max, N_max, 2]
    node_mask: torch.Tensor            # [B, N_max] — 1 = real, 0 = padding
    num_nodes: torch.Tensor            # [B] — actual node count per image
    
    # === Per-sample data (variable length, kept as lists) ===
    gt_sequences: List[List[Dict]]     # B lists of GT token dicts
    img_widths: List[int]              # B image widths
    img_heights: List[int]             # B image heights
    radii: List[float]                 # B threshold radii
    
    # === Metadata ===
    batch_size: int
    max_nodes: int


def collate_graphs(batch: List[Dict[str, Any]]) -> BatchedSample:
    """
    Custom collate function for the Cognitive Reader DataLoader.
    
    Args:
        batch: List of B sample dicts, each containing:
            'image': [3, H, W] tensor
            'boxes': [N, 4] tensor — (x1, y1, x2, y2)
            'node_positions_norm': [N, 2] tensor
            'node_positions_px': [N, 2] tensor
            'node_labels': [N] tensor
            'node_chunk_ids': [N] tensor
            'adjacency': [N, N] tensor
            'edge_features': [N, N, 3] tensor
            'edge_directions': [N, N, 2] tensor
            'gt_sequence': List[Dict]
            'heatmap_target': [1, H/8, W/8] tensor
            'img_width': int
            'img_height': int
            'radius': float
    
    Returns:
        BatchedSample with padded and batched tensors.
    """
    B = len(batch)
    
    # ============================================================
    # 1. Standard stacking: images and heatmap targets
    # ============================================================
    images = torch.stack([s['image'] for s in batch], dim=0)          # [B, 3, H, W]
    heatmap_targets = torch.stack([s['heatmap_target'] for s in batch], dim=0)  # [B, 1, H/8, W/8]
    
    # ============================================================
    # 2. Concatenate boxes with batch indices (for RoI Align)
    # ============================================================
    all_boxes = []
    all_batch_indices = []
    for i, s in enumerate(batch):
        boxes_i = s['boxes']  # [N_i, 4]
        N_i = boxes_i.shape[0]
        all_boxes.append(boxes_i)
        all_batch_indices.append(torch.full((N_i,), i, dtype=torch.long))
    
    boxes_cat = torch.cat(all_boxes, dim=0)           # [N_total, 4]
    box_batch_indices = torch.cat(all_batch_indices, dim=0)  # [N_total]
    
    # ============================================================
    # 3. Determine N_max and pad graph tensors
    # ============================================================
    num_nodes_list = [s['node_positions_norm'].shape[0] for s in batch]
    N_max = max(num_nodes_list)
    num_nodes = torch.tensor(num_nodes_list, dtype=torch.long)  # [B]
    
    # Initialize padded tensors
    node_positions_norm = torch.full((B, N_max, 2), -9999.0)  # Far away
    node_positions_px = torch.full((B, N_max, 2), -9999.0)
    node_labels = torch.full((B, N_max), -1, dtype=torch.long)  # Ignored by loss
    node_chunk_ids = torch.full((B, N_max), -1, dtype=torch.long)
    adjacency = torch.zeros(B, N_max, N_max)
    edge_features = torch.zeros(B, N_max, N_max, 3)
    edge_directions = torch.zeros(B, N_max, N_max, 2)
    node_mask = torch.zeros(B, N_max)  # 0 = padding
    
    for i, s in enumerate(batch):
        N_i = num_nodes_list[i]
        
        # Fill real node data
        node_positions_norm[i, :N_i] = s['node_positions_norm']
        node_positions_px[i, :N_i] = s['node_positions_px']
        node_labels[i, :N_i] = s['node_labels']
        node_chunk_ids[i, :N_i] = s['node_chunk_ids']
        adjacency[i, :N_i, :N_i] = s['adjacency']
        edge_features[i, :N_i, :N_i] = s['edge_features']
        edge_directions[i, :N_i, :N_i] = s['edge_directions']
        node_mask[i, :N_i] = 1.0  # Mark real nodes
    
    # ============================================================
    # 4. Per-sample data (kept as lists)
    # ============================================================
    gt_sequences = [s['gt_sequence'] for s in batch]
    img_widths = [s['img_width'] for s in batch]
    img_heights = [s['img_height'] for s in batch]
    radii = [s['radius'] for s in batch]
    
    return BatchedSample(
        images=images,
        heatmap_targets=heatmap_targets,
        boxes=boxes_cat,
        box_batch_indices=box_batch_indices,
        node_positions_norm=node_positions_norm,
        node_positions_px=node_positions_px,
        node_labels=node_labels,
        node_chunk_ids=node_chunk_ids,
        adjacency=adjacency,
        edge_features=edge_features,
        edge_directions=edge_directions,
        node_mask=node_mask,
        num_nodes=num_nodes,
        gt_sequences=gt_sequences,
        img_widths=img_widths,
        img_heights=img_heights,
        radii=radii,
        batch_size=B,
        max_nodes=N_max
    )


def unpad_graph(
    batched: BatchedSample,
    sample_idx: int,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Extract a single unpadded graph from a BatchedSample.
    
    Used by the training loop to process one graph at a time
    through the DualModeController.
    
    Args:
        batched: The BatchedSample from the DataLoader.
        sample_idx: Index of the sample in the batch (0 to B-1).
        device: Target device.
    
    Returns:
        Dict with unpadded tensors and metadata for one sample.
    """
    N = batched.num_nodes[sample_idx].item()
    
    # Build a SpatialGraph from the unpadded data
    from models.graph.builder import SpatialGraph
    
    graph = SpatialGraph(
        node_positions_norm=batched.node_positions_norm[sample_idx, :N].to(device),
        node_positions_px=batched.node_positions_px[sample_idx, :N].to(device),
        node_labels=batched.node_labels[sample_idx, :N].to(device),
        node_chunk_ids=batched.node_chunk_ids[sample_idx, :N].to(device),
        num_nodes=N,
        adjacency=batched.adjacency[sample_idx, :N, :N].to(device),
        edge_features=batched.edge_features[sample_idx, :N, :N].to(device),
        edge_directions=batched.edge_directions[sample_idx, :N, :N].to(device),
        img_width=batched.img_widths[sample_idx],
        img_height=batched.img_heights[sample_idx],
        radius=batched.radii[sample_idx],
        node_embeddings=None  # Filled by backbone
    )
    
    # Extract boxes for this sample
    box_mask = batched.box_batch_indices == sample_idx
    boxes = batched.boxes[box_mask].to(device)  # [N, 4]
    
    return {
        'graph': graph,
        'boxes': boxes,
        'image': batched.images[sample_idx:sample_idx+1].to(device),  # [1, 3, H, W]
        'heatmap_target': batched.heatmap_targets[sample_idx:sample_idx+1].to(device),
        'gt_sequence': batched.gt_sequences[sample_idx],
        'num_nodes': N
    }


def create_visited_mask_from_padding(
    node_mask: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create an initial visited mask that marks padded nodes as "visited".
    
    This prevents the controller from ever selecting a padded node.
    Padded nodes have node_mask = 0, so visited_mask = 1 for them.
    
    Args:
        node_mask: [B, N_max] — 1 = real node, 0 = padding.
    
    Returns:
        [B, N_max] — 1 = visited (or padding), 0 = unvisited.
    """
    # Padded nodes are always "visited" (1 - node_mask)
    # Real nodes start as unvisited (0)
    return (1.0 - node_mask).to(device)


def verify_no_phantom_edges(
    adjacency: torch.Tensor,
    node_mask: torch.Tensor
) -> bool:
    """
    Verify that no edges exist between real nodes and padded nodes.
    
    This is a safety check to ensure the padding strategy is correct.
    If this fails, the graph builder or collate function has a bug.
    
    Args:
        adjacency: [B, N_max, N_max]
        node_mask: [B, N_max]
    
    Returns:
        True if no phantom edges exist, False otherwise.
    """
    B, N_max, _ = adjacency.shape
    
    for b in range(B):
        real_mask = node_mask[b].bool()       # [N_max]
        pad_mask = ~real_mask                  # [N_max]
        
        if pad_mask.sum() == 0:
            continue  # No padding in this sample
        
        # Check edges from real nodes to padded nodes
        real_to_pad = adjacency[b][real_mask][:, pad_mask]
        if real_to_pad.sum() > 0:
            return False
        
        # Check edges from padded nodes to real nodes
        pad_to_real = adjacency[b][pad_mask][:, real_mask]
        if pad_to_real.sum() > 0:
            return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  collate_graphs Unit Test")
    print("=" * 60)
    
    # Simulate a batch of 3 samples with different node counts
    def make_fake_sample(N: int, H: int = 640, W: int = 640) -> Dict:
        """Create a fake sample for testing."""
        stride = 8
        return {
            'image': torch.randn(3, H, W),
            'boxes': torch.rand(N, 4) * torch.tensor([W, H, W, H]).float(),
            'node_positions_norm': torch.rand(N, 2),
            'node_positions_px': torch.rand(N, 2) * torch.tensor([W, H]).float(),
            'node_labels': torch.randint(0, 10, (N,)),
            'node_chunk_ids': torch.zeros(N, dtype=torch.long),
            'adjacency': (torch.rand(N, N) > 0.5).float() * (1 - torch.eye(N)),
            'edge_features': torch.randn(N, N, 3),
            'edge_directions': torch.randn(N, N, 2),
            'gt_sequence': [
                {'token': str(i % 10), 'node_id': i, 'mode': 'READ'}
                for i in range(N)
            ],
            'heatmap_target': torch.rand(1, H // stride, W // stride),
            'img_width': W,
            'img_height': H,
            'radius': 80.0
        }
    
    # Create batch with varying sizes
    batch = [
        make_fake_sample(N=5),
        make_fake_sample(N=12),
        make_fake_sample(N=3),
    ]
    
    # Collate
    batched = collate_graphs(batch)
    
    print(f"\n  Batch size:     {batched.batch_size}")
    print(f"  Max nodes:      {batched.max_nodes}")
    print(f"  Num nodes:      {batched.num_nodes.tolist()}")
    
    print(f"\n  Tensor shapes:")
    print(f"    images:              {batched.images.shape}")           # [3, 3, 640, 640]
    print(f"    heatmap_targets:     {batched.heatmap_targets.shape}")  # [3, 1, 80, 80]
    print(f"    boxes:               {batched.boxes.shape}")            # [20, 4]
    print(f"    box_batch_indices:   {batched.box_batch_indices.shape}") # [20]
    print(f"    node_positions_norm: {batched.node_positions_norm.shape}") # [3, 12, 2]
    print(f"    adjacency:           {batched.adjacency.shape}")        # [3, 12, 12]
    print(f"    edge_features:       {batched.edge_features.shape}")    # [3, 12, 12, 3]
    print(f"    node_mask:           {batched.node_mask.shape}")        # [3, 12]
    print(f"    node_labels:         {batched.node_labels.shape}")      # [3, 12]
    
    # Verify node_mask
    print(f"\n  Node masks:")
    for i in range(3):
        N_i = batched.num_nodes[i].item()
        mask = batched.node_mask[i]
        real_count = mask.sum().item()
        pad_count = (1 - mask).sum().item()
        print(f"    Sample {i}: {int(real_count)} real, {int(pad_count)} padded")
        assert real_count == N_i, f"Expected {N_i} real nodes, got {int(real_count)}"
    
    # Verify no phantom edges
    no_phantoms = verify_no_phantom_edges(batched.adjacency, batched.node_mask)
    print(f"\n  Phantom edge check: {'✓ PASSED' if no_phantoms else '✗ FAILED'}")
    assert no_phantoms, "Phantom edges detected!"
    
    # Verify padded positions are far away
    for i in range(3):
        N_i = batched.num_nodes[i].item()
        if N_i < batched.max_nodes:
            padded_pos = batched.node_positions_norm[i, N_i:]
            assert (padded_pos == -9999.0).all(), "Padded positions not set to -9999!"
    print(f"  Padded positions:   ✓ All set to -9999.0")
    
    # Verify padded labels are -1
    for i in range(3):
        N_i = batched.num_nodes[i].item()
        if N_i < batched.max_nodes:
            padded_labels = batched.node_labels[i, N_i:]
            assert (padded_labels == -1).all(), "Padded labels not set to -1!"
    print(f"  Padded labels:      ✓ All set to -1")
    
    # Test unpad_graph
    print(f"\n  Unpad Test:")
    for i in range(3):
        unpadded = unpad_graph(batched, i)
        N_i = batched.num_nodes[i].item()
        assert unpadded['graph'].num_nodes == N_i
        assert unpadded['graph'].adjacency.shape == (N_i, N_i)
        assert unpadded['boxes'].shape[0] == N_i
        print(f"    Sample {i}: {N_i} nodes, adjacency {unpadded['graph'].adjacency.shape}")
    
    # Test visited mask from padding
    print(f"\n  Visited Mask Test:")
    init_visited = create_visited_mask_from_padding(batched.node_mask)
    for i in range(3):
        N_i = batched.num_nodes[i].item()
        real_unvisited = (init_visited[i, :N_i] == 0).sum().item()
        pad_visited = (init_visited[i, N_i:] == 1).sum().item()
        print(f"    Sample {i}: {real_unvisited} real unvisited, {pad_visited} padded visited")
        assert real_unvisited == N_i
        assert pad_visited == batched.max_nodes - N_i
    
    # Verify box batch indices
    print(f"\n  Box Batch Indices:")
    for i in range(3):
        count = (batched.box_batch_indices == i).sum().item()
        N_i = batched.num_nodes[i].item()
        print(f"    Sample {i}: {count} boxes (expected {N_i})")
        assert count == N_i
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)