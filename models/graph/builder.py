"""
models/graph/builder.py
Threshold-Radius Spatial Graph Builder for the Cognitive Reader project.

Constructs a static 2D graph from digit bounding boxes.
The graph encodes:
  - Which digits are spatially close (adjacency matrix)
  - The geometric relationship between connected digits (edge features)
  - The absolute position of each digit (node positions)

The graph knows NOTHING about the controller, the visual backbone,
or the reading order. It is purely a geometric data structure.
"""

import torch
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpatialGraph:
    """
    Static spatial graph data structure.
    
    All tensors are single-image (no batch dimension).
    The collate function in data/collate.py handles batching.
    """
    # --- Node features ---
    node_positions_norm: torch.Tensor   # [N, 2] — normalized (x/W, y/H) in [0, 1]
    node_positions_px: torch.Tensor     # [N, 2] — raw pixel coordinates (x, y)
    node_labels: torch.Tensor           # [N] — digit labels (0-9), -1 if unknown
    node_chunk_ids: torch.Tensor        # [N] — chunk assignment, -1 if unknown
    num_nodes: int

    # --- Graph topology ---
    adjacency: torch.Tensor             # [N, N] — binary (1.0 if edge exists, 0.0 otherwise)
    edge_features: torch.Tensor         # [N, N, 3] — (delta_x_norm, delta_y_norm, distance_px)
    edge_directions: torch.Tensor       # [N, N, 2] — unit vector (cos_theta, sin_theta) from i to j

    # --- Metadata ---
    img_width: int
    img_height: int
    radius: float                       # threshold radius used to build this graph

    # --- Filled later by backbone ---
    node_embeddings: Optional[torch.Tensor] = None  # [N, D] — visual embeddings

    def to(self, device: torch.device) -> 'SpatialGraph':
        """Move all tensors to the specified device."""
        self.node_positions_norm = self.node_positions_norm.to(device)
        self.node_positions_px = self.node_positions_px.to(device)
        self.node_labels = self.node_labels.to(device)
        self.node_chunk_ids = self.node_chunk_ids.to(device)
        self.adjacency = self.adjacency.to(device)
        self.edge_features = self.edge_features.to(device)
        self.edge_directions = self.edge_directions.to(device)
        if self.node_embeddings is not None:
            self.node_embeddings = self.node_embeddings.to(device)
        return self

    def get_degree(self) -> torch.Tensor:
        """Return the degree (number of neighbors) for each node. [N]"""
        return self.adjacency.sum(dim=1)


class ThresholdRadiusGraphBuilder:
    """
    Builds a Threshold-Radius Spatial Graph from digit bounding boxes.
    
    Usage:
        builder = ThresholdRadiusGraphBuilder(radius=80.0, img_width=640, img_height=640)
        graph = builder.build_from_boxes(boxes, labels, chunk_ids)
    """

    def __init__(self, radius: float, img_width: int, img_height: int):
        """
        Args:
            radius: Threshold radius r (pixels). Nodes within this Euclidean
                    distance are connected by a directed edge.
            img_width: Image width in pixels (for coordinate normalization).
            img_height: Image height in pixels (for coordinate normalization).
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self.radius = radius
        self.img_width = img_width
        self.img_height = img_height

    def build_from_boxes(
        self,
        boxes: List[Dict],
        labels: Optional[List[str]] = None,
        chunk_ids: Optional[List[int]] = None
    ) -> SpatialGraph:
        """
        Build a spatial graph from a list of bounding boxes.

        Args:
            boxes: List of dicts, each with keys:
                   'center_x', 'center_y', 'w', 'h', 'node_id'
            labels: Optional list of digit labels ('0'-'9').
            chunk_ids: Optional list of chunk assignments (int).

        Returns:
            SpatialGraph data structure with all geometric information.
        """
        N = len(boxes)
        if N == 0:
            raise ValueError("Cannot build graph from empty box list")

        # --- Extract center coordinates ---
        centers_px = np.zeros((N, 2), dtype=np.float64)
        for i, box in enumerate(boxes):
            centers_px[i, 0] = box['center_x']
            centers_px[i, 1] = box['center_y']

        # --- Normalize coordinates to [0, 1] ---
        centers_norm = np.zeros((N, 2), dtype=np.float64)
        centers_norm[:, 0] = centers_px[:, 0] / self.img_width
        centers_norm[:, 1] = centers_px[:, 1] / self.img_height

        # --- Compute pairwise Euclidean distances (pixels) ---
        # Vectorized: dist[i, j] = ||center_i - center_j||_2
        diff = centers_px[:, np.newaxis, :] - centers_px[np.newaxis, :, :]  # [N, N, 2]
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # [N, N]

        # --- Build adjacency matrix ---
        # Edge exists if distance < radius (strict inequality)
        adjacency = (dist < self.radius).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)  # No self-loops

        # --- Compute edge features: [N, N, 3] ---
        # For each directed pair (i -> j):
        #   channel 0: delta_x normalized by image width
        #   channel 1: delta_y normalized by image height
        #   channel 2: raw Euclidean distance in pixels
        edge_features = np.zeros((N, N, 3), dtype=np.float64)
        edge_features[:, :, 0] = (centers_px[np.newaxis, :, 0] - centers_px[:, np.newaxis, 0]) / self.img_width
        edge_features[:, :, 1] = (centers_px[np.newaxis, :, 1] - centers_px[:, np.newaxis, 1]) / self.img_height
        edge_features[:, :, 2] = dist

        # --- Compute edge direction unit vectors: [N, N, 2] ---
        # (cos_theta, sin_theta) pointing from node i to node j
        # Avoid division by zero for self-loops and disconnected pairs
        edge_directions = np.zeros((N, N, 2), dtype=np.float64)
        safe_dist = np.where(dist > 1e-8, dist, 1.0)  # prevent div by zero
        edge_directions[:, :, 0] = (centers_px[np.newaxis, :, 0] - centers_px[:, np.newaxis, 0]) / safe_dist
        edge_directions[:, :, 1] = (centers_px[np.newaxis, :, 1] - centers_px[:, np.newaxis, 1]) / safe_dist
        # Zero out directions for self-loops
        mask = np.eye(N, dtype=bool)
        edge_directions[mask] = 0.0

        # --- Node labels ---
        if labels is not None:
            node_labels = np.array([int(l) for l in labels], dtype=np.int64)
        else:
            node_labels = np.full(N, -1, dtype=np.int64)

        # --- Chunk IDs ---
        if chunk_ids is not None:
            node_chunk_ids = np.array(chunk_ids, dtype=np.int64)
        else:
            node_chunk_ids = np.full(N, -1, dtype=np.int64)

        # --- Assemble SpatialGraph ---
        graph = SpatialGraph(
            node_positions_norm=torch.tensor(centers_norm, dtype=torch.float32),
            node_positions_px=torch.tensor(centers_px, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.long),
            node_chunk_ids=torch.tensor(node_chunk_ids, dtype=torch.long),
            num_nodes=N,
            adjacency=torch.tensor(adjacency, dtype=torch.float32),
            edge_features=torch.tensor(edge_features, dtype=torch.float32),
            edge_directions=torch.tensor(edge_directions, dtype=torch.float32),
            img_width=self.img_width,
            img_height=self.img_height,
            radius=self.radius,
            node_embeddings=None
        )

        return graph

    def get_unvisited_neighbors(
        self,
        graph: SpatialGraph,
        node_idx: int,
        visited_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get indices of unvisited neighbors within the threshold radius.

        Args:
            graph: The spatial graph.
            node_idx: Index of the current node.
            visited_mask: Binary tensor [N]. 1 = visited, 0 = unvisited.

        Returns:
            1D tensor of unvisited neighbor indices. Empty if none.
        """
        # adjacency[node_idx] is 1.0 for connected nodes, 0.0 otherwise
        connected = graph.adjacency[node_idx] > 0.5  # [N] boolean
        unvisited = visited_mask == 0                 # [N] boolean
        mask = connected & unvisited
        return torch.where(mask)[0]

    def get_all_unvisited(self, visited_mask: torch.Tensor) -> torch.Tensor:
        """
        Get indices of all unvisited nodes in the graph.
        Used by Mode 2 (Saccadic Jump) for global search.

        Args:
            visited_mask: Binary tensor [N]. 1 = visited, 0 = unvisited.

        Returns:
            1D tensor of unvisited node indices.
        """
        return torch.where(visited_mask == 0)[0]


class GraphValidator:
    """
    Validates graph topology against ground-truth chunk assignments.
    Use this during development to catch data generation bugs early.
    """

    @staticmethod
    def validate(graph: SpatialGraph) -> Dict:
        """
        Check that the graph topology is consistent with chunk labels.

        Rules:
          1. All nodes in the same chunk MUST be connected (path exists).
          2. Nodes in adjacent chunks SHOULD NOT have a direct edge.

        Returns:
            Dict with 'valid', 'intra_violations', 'inter_violations'.
        """
        N = graph.num_nodes
        chunk_ids = graph.node_chunk_ids.numpy()
        adj = graph.adjacency.numpy()

        intra_violations = []
        inter_violations = []

        for i in range(N):
            for j in range(i + 1, N):
                same_chunk = (chunk_ids[i] == chunk_ids[j])
                has_edge = (adj[i, j] > 0.5)

                if same_chunk and not has_edge:
                    # Check if they are adjacent in the chunk sequence
                    # (non-adjacent nodes in a long chunk may not have a direct edge)
                    # For now, flag all same-chunk disconnected pairs
                    intra_violations.append((i, j, chunk_ids[i]))

                elif not same_chunk and has_edge:
                    chunk_diff = abs(int(chunk_ids[i]) - int(chunk_ids[j]))
                    if chunk_diff == 1:
                        inter_violations.append((i, j, chunk_ids[i], chunk_ids[j]))

        return {
            'valid': len(intra_violations) == 0 and len(inter_violations) == 0,
            'intra_violations': intra_violations,
            'inter_violations': inter_violations,
            'num_intra_violations': len(intra_violations),
            'num_inter_violations': len(inter_violations),
            'total_nodes': N,
            'total_edges': int(adj.sum()),
            'avg_degree': float(adj.sum() / N) if N > 0 else 0.0
        }

    @staticmethod
    def print_report(validation: Dict) -> None:
        """Print a human-readable validation report."""
        status = "✓ VALID" if validation['valid'] else "✗ INVALID"
        print(f"\n{'='*50}")
        print(f"  Graph Validation: {status}")
        print(f"{'='*50}")
        print(f"  Total nodes:          {validation['total_nodes']}")
        print(f"  Total edges:          {validation['total_edges']}")
        print(f"  Average degree:       {validation['avg_degree']:.2f}")
        print(f"  Intra-chunk breaks:   {validation['num_intra_violations']}")
        print(f"  Inter-chunk leaks:    {validation['num_inter_violations']}")

        if validation['intra_violations']:
            print(f"\n  Intra-chunk violations (same chunk, no edge):")
            for i, j, c in validation['intra_violations'][:5]:
                print(f"    nodes ({i}, {j}) in chunk {c}")

        if validation['inter_violations']:
            print(f"\n  Inter-chunk violations (different chunks, edge exists):")
            for i, j, ci, cj in validation['inter_violations'][:5]:
                print(f"    nodes ({i}, {j}) in chunks ({ci}, {cj})")

        print(f"{'='*50}\n")


# --- Convenience: build graph from a GeneratedSample ---
def build_graph_from_sample(
    sample,
    radius: float,
    use_noise: bool = True
) -> SpatialGraph:
    """
    Build a SpatialGraph directly from a GeneratedSample (data/generator.py output).

    Args:
        sample: GeneratedSample from ConstrainedPolarGenerator.
        radius: Threshold radius r (pixels).
        use_noise: If True, use noisy centers (detector simulation).
                   If False, use perfect centers (oracle).

    Returns:
        SpatialGraph ready for the controller.
    """
    boxes = []
    labels = []
    chunk_ids = []

    for node in sample.nodes:
        cx = node.noisy_center_x if use_noise else node.center_x
        cy = node.noisy_center_y if use_noise else node.center_y

        boxes.append({
            'center_x': cx,
            'center_y': cy,
            'w': node.width,
            'h': node.height,
            'node_id': node.node_id
        })
        labels.append(node.label)
        chunk_ids.append(node.chunk_id)

    builder = ThresholdRadiusGraphBuilder(
        radius=radius,
        img_width=sample.img_width,
        img_height=sample.img_height
    )

    graph = builder.build_from_boxes(boxes, labels, chunk_ids)
    return graph


if __name__ == "__main__":
    # --- Unit test: build a graph from a generated sample ---
    import sys
    sys.path.append('..')  # Allow importing from data/

    # Simulate a small sample manually (no dependency on generator for this test)
    boxes = [
        {'center_x': 100, 'center_y': 100, 'w': 20, 'h': 30, 'node_id': 0},
        {'center_x': 140, 'center_y': 102, 'w': 20, 'h': 30, 'node_id': 1},
        {'center_x': 180, 'center_y': 98,  'w': 20, 'h': 30, 'node_id': 2},
        # Chunk boundary: large gap
        {'center_x': 350, 'center_y': 250, 'w': 20, 'h': 30, 'node_id': 3},
        {'center_x': 390, 'center_y': 252, 'w': 20, 'h': 30, 'node_id': 4},
    ]
    labels = ['1', '2', '3', '4', '5']
    chunk_ids = [0, 0, 0, 1, 1]

    builder = ThresholdRadiusGraphBuilder(radius=80.0, img_width=640, img_height=640)
    graph = builder.build_from_boxes(boxes, labels, chunk_ids)

    print(f"Nodes: {graph.num_nodes}")
    print(f"Adjacency:\n{graph.adjacency}")
    print(f"Node degrees: {graph.get_degree()}")

    # Validate
    validation = GraphValidator.validate(graph)
    GraphValidator.print_report(validation)

    # Test neighbor queries
    visited = torch.zeros(5)
    visited[0] = 1  # Mark node 0 as visited
    neighbors_of_1 = builder.get_unvisited_neighbors(graph, 1, visited)
    print(f"Unvisited neighbors of node 1: {neighbors_of_1.tolist()}")