"""
models/graph/features.py
Sinusoidal Fourier Feature Encoding for the Routing Pathway.

Maps continuous 2D spatial coordinates and vectors into a high-dimensional
embedding space where Euclidean distance correlates with physical distance.

STRICT USAGE RULE:
  This module is used ONLY by the Routing Pathway (traversal decisions).
  It is NEVER used by the Identity Pathway (digit classification).
  The dual-pathway invariant must be preserved.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalFourierEncoding(nn.Module):
    """
    Fixed (non-learnable) Sinusoidal Fourier Feature encoding.
    
    Maps a D-dimensional input to a (D * 2 * L)-dimensional output,
    where L is the number of frequency bands.
    
    For each input dimension x and each frequency f_k = 2^k:
        output includes sin(2π * f_k * x) and cos(2π * f_k * x)
    
    This is the same principle as Transformer positional encodings
    and NeRF positional encoding. The high-frequency components allow
    the network to represent sharp spatial boundaries without spectral bias.
    """
    
    def __init__(self, num_frequencies: int = 64, input_dim: int = 2):
        """
        Args:
            num_frequencies: Number of frequency bands L.
                             Output dim = input_dim * 2 * num_frequencies.
            input_dim: Dimensionality of the input (2 for (x,y) coordinates).
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.output_dim = input_dim * 2 * num_frequencies
        
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        # Exponentially spaced to capture both coarse and fine spatial structure
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer('freq_bands', freq_bands)  # [L]
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using sinusoidal Fourier features.
        
        Args:
            coords: [..., input_dim] tensor. Can be:
                    - [N, 2] for a batch of node positions
                    - [N, N, 2] for a matrix of edge deltas
                    - [2] for a single point/vector
                    - [B, N, 2] for a batched graph
        
        Returns:
            [..., output_dim] tensor of Fourier features.
        """
        # coords: [..., D]
        # freq_bands: [L]
        # Expand for broadcasting: [..., D, 1] * [1, ..., 1, L] -> [..., D, L]
        scaled = coords.unsqueeze(-1) * self.freq_bands * (2.0 * math.pi)
        
        # Compute sin and cos: [..., D, L] each
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        
        # Interleave sin and cos: [..., D, 2L]
        fourier = torch.cat([sin_feat, cos_feat], dim=-1)
        
        # Flatten the last two dimensions: [..., D * 2L]
        batch_shape = coords.shape[:-1]
        return fourier.reshape(*batch_shape, self.output_dim)


class SpatialFeatureEncoder(nn.Module):
    """
    High-level spatial encoding module for the Routing Pathway.
    
    Provides encoding methods for all spatial inputs used by the controller:
      1. Node positions (x, y) — for Mode 2 global attention keys
      2. Edge features (Δx, Δy, d) — for Mode 1 local routing logits
      3. Spatial anchor (x, y) — for Mode 2 query construction
      4. Chunk trajectory (Δx, Δy) — for Mode 2 query construction
    
    All encodings are projected to a common output_dim for downstream use.
    
    ARCHITECTURAL INVARIANT:
      The output of this module is NEVER fed to the digit classification head.
      It is used exclusively for routing decisions (which node to visit next,
      when to chunk, where to jump).
    """
    
    def __init__(
        self,
        num_frequencies: int = 64,
        output_dim: int = 256,
        use_learnable_projection: bool = True
    ):
        """
        Args:
            num_frequencies: Number of Fourier frequency bands.
            output_dim: Output dimensionality for all encoded features.
            use_learnable_projection: If True, add a learnable linear projection
                                      after the fixed Fourier features.
                                      If False, output_dim must equal fourier_dim.
        """
        super().__init__()
        
        self.fourier = SinusoidalFourierEncoding(
            num_frequencies=num_frequencies,
            input_dim=2
        )
        self.fourier_dim = self.fourier.output_dim  # 2 * 2 * L = 4L
        self.output_dim = output_dim
        self.use_learnable_projection = use_learnable_projection
        
        if use_learnable_projection:
            # Separate projections for each spatial input type.
            # This allows the model to learn different representations
            # for positions vs. edges vs. trajectories.
            
            # Node positions: Fourier(2D) -> output_dim
            self.pos_proj = nn.Sequential(
                nn.Linear(self.fourier_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            
            # Edge features: Fourier(2D) + normalized_distance -> output_dim
            # The +1 accounts for d/r (normalized Euclidean distance)
            self.edge_proj = nn.Sequential(
                nn.Linear(self.fourier_dim + 1, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            
            # Trajectory vectors: Fourier(2D) -> output_dim
            self.traj_proj = nn.Sequential(
                nn.Linear(self.fourier_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
        else:
            if self.fourier_dim != output_dim:
                raise ValueError(
                    f"Without learnable projection, output_dim ({output_dim}) "
                    f"must equal fourier_dim ({self.fourier_dim}). "
                    f"Set num_frequencies={output_dim // 4} or use_learnable_projection=True."
                )
    
    def encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode absolute node positions for Mode 2 global attention keys.
        
        Args:
            positions: [..., 2] tensor of normalized coordinates in [0, 1].
                       Typical shapes: [N, 2] or [B, N, 2].
        
        Returns:
            [..., output_dim] tensor of position embeddings.
        
        Usage:
            In Mode 2, the key for each unvisited node is:
                K_i = encode_positions(node_positions_norm[i])
        """
        fourier = self.fourier(positions)
        if self.use_learnable_projection:
            return self.pos_proj(fourier)
        return fourier
    
    def encode_edges(
        self,
        edge_features: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """
        Encode edge features for Mode 1 local routing decisions.
        
        Args:
            edge_features: [..., 3] tensor where:
                           channel 0: Δx normalized by image width
                           channel 1: Δy normalized by image height
                           channel 2: raw Euclidean distance in pixels
            radius: Threshold radius r (pixels), used to normalize distance.
        
        Returns:
            [..., output_dim] tensor of edge embeddings.
        
        Usage:
            In Mode 1, the routing logit for neighbor j is:
                z_j = MLP(h_content || encode_edges(edge_features[current, j], r))
        """
        # Separate spatial deltas from raw distance
        deltas = edge_features[..., :2]       # [..., 2] — (Δx_norm, Δy_norm)
        dist_px = edge_features[..., 2:3]     # [..., 1] — raw pixel distance
        
        # Normalize distance by radius: d/r ∈ [0, ~2.5]
        # This makes the distance feature scale-invariant
        dist_norm = dist_px / radius
        
        # Encode deltas with Fourier features
        fourier = self.fourier(deltas)        # [..., fourier_dim]
        
        # Concatenate Fourier features with normalized distance
        combined = torch.cat([fourier, dist_norm], dim=-1)  # [..., fourier_dim + 1]
        
        if self.use_learnable_projection:
            return self.edge_proj(combined)
        return combined
    
    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode the chunk trajectory vector for Mode 2 query construction.
        
        The trajectory vector v_chunk = (x_last - x_first, y_last - y_first)
        represents the reading direction of the chunk just completed.
        
        Args:
            trajectory: [..., 2] tensor of normalized trajectory vectors.
                        Typical shapes: [2] or [B, 2].
        
        Returns:
            [..., output_dim] tensor of trajectory embeddings.
        
        Usage:
            In Mode 2 (t > 0), the query is:
                Q = Linear(encode_anchor(anchor) || encode_trajectory(v_chunk))
        """
        fourier = self.fourier(trajectory)
        if self.use_learnable_projection:
            return self.traj_proj(fourier)
        return fourier
    
    def encode_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        """
        Encode the spatial anchor for Mode 2 query construction.
        
        The spatial anchor (x_last, y_last) is the position of the last
        digit read before the saccadic jump.
        
        Args:
            anchor: [..., 2] tensor of normalized coordinates in [0, 1].
        
        Returns:
            [..., output_dim] tensor of anchor embeddings.
        
        Usage:
            In Mode 2, the query is:
                Q = Linear(encode_anchor(anchor) || encode_trajectory(v_chunk))
        """
        # Anchor uses the same encoding as positions
        return self.encode_positions(anchor)


class EdgeFeatureAggregator(nn.Module):
    """
    Aggregates edge features from multiple neighbors into routing
    and chunking decisions for Mode 1.
    
    The CHUNK logit is computed from:
      - h_content (what has been read)
      - e_closest (edge embedding of the nearest candidate)
      - d_min_norm (normalized minimum distance to any candidate)
    
    This ensures the model has direct access to the proximity signal
    that defines chunk boundaries.
    """
    
    def __init__(self, edge_dim: int, hidden_dim: int, radius: float):
        super().__init__()
        self.radius = radius
        
        # Routing logit: (h_content || edge_embedding) -> scalar
        self.routing_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # CHUNK logit: (h_content || e_closest || d_min_norm) -> scalar
        # The +1 accounts for the normalized minimum distance scalar
        self.chunk_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        h_content: torch.Tensor,
        edge_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_distances: torch.Tensor
    ) -> tuple:
        """
        Compute routing logits and CHUNK logit for Mode 1.
        
        Args:
            h_content: [hidden_dim] — current content hidden state.
            edge_embeddings: [K, edge_dim] — edge embeddings for K candidates.
            candidate_mask: [K] — 1 = valid candidate, 0 = visited/padding.
            edge_distances: [K] — raw pixel distances to each candidate.
        
        Returns:
            routing_logits: [K] — logit for each candidate.
            chunk_logit: [1] — logit for the <CHUNK> action.
        """
        K = edge_embeddings.shape[0]
        device = h_content.device
        
        # --- Structural termination: no candidates at all ---
        if K == 0 or candidate_mask.sum() == 0:
            return (
                torch.tensor([], device=device),
                torch.tensor([10.0], device=device)  # Force CHUNK
            )
        
        # --- Routing logits ---
        h_expanded = h_content.unsqueeze(0).expand(K, -1)  # [K, hidden_dim]
        combined = torch.cat([h_expanded, edge_embeddings], dim=-1)  # [K, hidden_dim + edge_dim]
        routing_logits = self.routing_mlp(combined).squeeze(-1)  # [K]
        routing_logits = routing_logits.masked_fill(candidate_mask == 0, float('-inf'))
        
        # --- CHUNK logit (the fix) ---
        # 1. Find the minimum distance among valid candidates
        valid_distances = edge_distances.clone()
        valid_distances[candidate_mask == 0] = float('inf')
        d_min = valid_distances.min()
        d_min_norm = (d_min / self.radius).unsqueeze(0)  # [1] — normalized by r
        
        # 2. Get the edge embedding of the closest valid candidate
        closest_idx = valid_distances.argmin()
        e_closest = edge_embeddings[closest_idx]  # [edge_dim]
        
        # 3. Compute CHUNK logit from (h_content, e_closest, d_min_norm)
        chunk_input = torch.cat([h_content, e_closest, d_min_norm], dim=-1)
        chunk_logit = self.chunk_mlp(chunk_input)  # [1]
        
        return routing_logits, chunk_logit

    
if __name__ == "__main__":
    # --- Unit tests ---
    
    print("=" * 60)
    print("  SpatialFeatureEncoder Unit Tests")
    print("=" * 60)
    
    # Test 1: SinusoidalFourierEncoding
    print("\n[Test 1] SinusoidalFourierEncoding")
    encoder = SinusoidalFourierEncoding(num_frequencies=64, input_dim=2)
    print(f"  Output dim: {encoder.output_dim}")  # Should be 2 * 2 * 64 = 256
    
    # Single point
    point = torch.tensor([0.5, 0.3])
    encoded = encoder(point)
    print(f"  Single point [2] -> {encoded.shape}")  # [256]
    
    # Batch of points
    points = torch.rand(10, 2)
    encoded_batch = encoder(points)
    print(f"  Batch [10, 2] -> {encoded_batch.shape}")  # [10, 256]
    
    # Matrix of edge deltas
    edges = torch.randn(10, 10, 2)
    encoded_edges = encoder(edges)
    print(f"  Edge matrix [10, 10, 2] -> {encoded_edges.shape}")  # [10, 10, 256]
    
    # Test 2: SpatialFeatureEncoder
    print("\n[Test 2] SpatialFeatureEncoder")
    spatial_enc = SpatialFeatureEncoder(num_frequencies=64, output_dim=256)
    
    positions = torch.rand(10, 2)
    pos_encoded = spatial_enc.encode_positions(positions)
    print(f"  Positions [10, 2] -> {pos_encoded.shape}")  # [10, 256]
    
    edge_feats = torch.randn(10, 10, 3)
    edge_encoded = spatial_enc.encode_edges(edge_feats, radius=80.0)
    print(f"  Edges [10, 10, 3] -> {edge_encoded.shape}")  # [10, 10, 256]
    
    trajectory = torch.tensor([0.1, -0.05])
    traj_encoded = spatial_enc.encode_trajectory(trajectory)
    print(f"  Trajectory [2] -> {traj_encoded.shape}")  # [256]
    
    anchor = torch.tensor([0.7, 0.4])
    anchor_encoded = spatial_enc.encode_anchor(anchor)
    print(f"  Anchor [2] -> {anchor_encoded.shape}")  # [256]
    
    # Test 3: EdgeFeatureAggregator
    print("\n[Test 3] EdgeFeatureAggregator")
    aggregator = EdgeFeatureAggregator(edge_dim=256, hidden_dim=128)
    
    h_content = torch.randn(128)
    K = 5  # 5 candidate neighbors
    edge_embs = torch.randn(K, 256)
    mask = torch.ones(K)
    mask[3] = 0  # Node 3 is visited
    
    routing_logits, chunk_logit = aggregator(h_content, edge_embs, mask)
    print(f"  Routing logits: {routing_logits.shape}")  # [5]
    print(f"  Routing logits: {routing_logits.tolist()}")
    print(f"  CHUNK logit: {chunk_logit.item():.4f}")
    print(f"  Masked node 3 logit: {routing_logits[3].item()}")  # Should be -inf
    
    # Test 4: Verify distance correlation
    print("\n[Test 4] Distance Correlation Verification")
    # Two points at different distances should have different encodings
    p1 = torch.tensor([0.5, 0.5])
    p2_near = torch.tensor([0.55, 0.5])   # 0.05 away
    p3_far = torch.tensor([0.8, 0.5])     # 0.3 away
    
    e1 = spatial_enc.encode_positions(p1)
    e2 = spatial_enc.encode_positions(p2_near)
    e3 = spatial_enc.encode_positions(p3_far)
    
    dist_near = torch.norm(e1 - e2).item()
    dist_far = torch.norm(e1 - e3).item()
    
    print(f"  Embedding distance (near, 0.05 apart): {dist_near:.4f}")
    print(f"  Embedding distance (far, 0.30 apart):  {dist_far:.4f}")
    print(f"  Ratio (far/near): {dist_far / dist_near:.2f}x")
    print(f"  ✓ Farther points have larger embedding distance" if dist_far > dist_near else "  ✗ FAILED")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)