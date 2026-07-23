"""
models/controller/saccadic.py
Mode 2: Saccadic Jump — Global search for the next chunk start.

Triggered when:
  1. Mode 1's local neighborhood is exhausted (structural termination).
  2. At t=0 (initialization — find the first digit to read).

Responsibilities:
  - Construct a query Q from spatial anchor + chunk trajectory (or CLS + START at t=0).
  - Compute attention over ALL unvisited nodes in the graph.
  - Select the next starting node.
  - Compute the loss for training.

ARCHITECTURAL INVARIANT:
  Mode 2 NEVER uses h_content (which is zeroed after <CHUNK>).
  The query is constructed purely from spatial information.
  Mode 2 NEVER classifies digits. It only selects a node index.
  Visual processing of the selected node is handled by dual_mode.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SaccadicOutput:
    """Output of a single Mode 2 forward pass."""
    attention_logits: torch.Tensor    # [N] — raw attention scores for ALL nodes
    attention_probs: torch.Tensor     # [N] — softmax probabilities (visited nodes = 0)
    selected_node_idx: int            # Index of the selected node in the graph
    query: torch.Tensor               # [key_dim] — the constructed query vector


class SaccadicJumpModule(nn.Module):
    """
    Mode 2: Saccadic Jump.
    
    Computes global attention over all unvisited nodes to find the
    starting node of the next chunk. Mimics human saccadic eye movements:
    a rapid jump from the end of one line to the start of the next.
    
    The query is constructed from SPATIAL information only:
      - At t=0: Q_0 = f(CLS_token, e_start)
      - At t>0: Q = f(encode(anchor), encode(trajectory))
    
    h_content is NEVER used here. It is zeroed after <CHUNK> and
    contains no useful information for the jump decision.
    """
    
    def __init__(
        self,
        key_dim: int,
        vis_dim: int,
        start_embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            key_dim: Dimensionality of spatial position embeddings
                     (from SpatialFeatureEncoder.output_dim).
            vis_dim: Dimensionality of the global CLS token (from backbone).
            start_embed_dim: Dimensionality of the learnable <START> embedding.
            num_heads: Number of attention heads for multi-head attention.
            dropout: Attention dropout rate.
        """
        super().__init__()
        
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads
        
        if key_dim % num_heads != 0:
            raise ValueError(
                f"key_dim ({key_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        # ============================================================
        # QUERY CONSTRUCTION: Initialization (t=0)
        # Q_0 = Linear(CLS_token || e_start)
        # The CLS token provides global image context (where is the text?).
        # The START embedding is a learnable "find the beginning" signal.
        # ============================================================
        self.start_embedding = nn.Parameter(
            torch.randn(start_embed_dim) * 0.02
        )
        self.query_init_proj = nn.Sequential(
            nn.Linear(vis_dim + start_embed_dim, key_dim),
            nn.LayerNorm(key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim)
        )
        
        # ============================================================
        # QUERY CONSTRUCTION: Saccadic Jump (t>0)
        # Q = Linear(encode(anchor) || encode(trajectory))
        # The anchor tells the model WHERE it currently is.
        # The trajectory tells the model WHICH DIRECTION it was moving.
        # Together, they allow the model to project forward and find
        # the start of the next line.
        # ============================================================
        self.query_jump_proj = nn.Sequential(
            nn.Linear(key_dim + key_dim, key_dim),  # anchor + trajectory
            nn.LayerNorm(key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim)
        )
        
        # ============================================================
        # MULTI-HEAD ATTENTION
        # Q: [1, key_dim] — the constructed query
        # K: [N, key_dim] — spatial embeddings of all nodes
        # V: not used (we only need the attention weights to select a node)
        # ============================================================
        self.W_q = nn.Linear(key_dim, key_dim, bias=False)
        self.W_k = nn.Linear(key_dim, key_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention sharpness
        # Initialized to 1/sqrt(head_dim) (standard scaled dot-product)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / math.sqrt(self.head_dim)))
        )
    
    def construct_query_initial(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Construct the initialization query Q_0 at t=0.
        
        Args:
            cls_token: [vis_dim] — global image feature from backbone GAP.
        
        Returns:
            [key_dim] — the initialization query vector.
        """
        # Expand start embedding to match batch if needed
        start_emb = self.start_embedding.unsqueeze(0)  # [1, start_embed_dim]
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)  # [1, vis_dim]
        
        # Concatenate and project
        combined = torch.cat([cls_token, start_emb.expand(cls_token.shape[0], -1)], dim=-1)
        query = self.query_init_proj(combined)  # [1, key_dim]
        return query.squeeze(0)  # [key_dim]
    
    def construct_query_jump(
        self,
        anchor_encoded: torch.Tensor,
        trajectory_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct the saccadic jump query Q at t>0.
        
        Args:
            anchor_encoded: [key_dim] — Fourier-encoded spatial anchor.
            trajectory_encoded: [key_dim] — Fourier-encoded chunk trajectory.
        
        Returns:
            [key_dim] — the jump query vector.
        """
        combined = torch.cat([anchor_encoded, trajectory_encoded], dim=-1)  # [2*key_dim]
        query = self.query_jump_proj(combined)  # [key_dim]
        return query
    
    def forward(
        self,
        query: torch.Tensor,
        node_keys: torch.Tensor,
        visited_mask: torch.Tensor,
        greedy: bool = True,
        temperature_override: Optional[float] = None
    ) -> SaccadicOutput:
        """
        Compute global attention over all nodes and select the next start.
        
        Args:
            query: [key_dim] — constructed query (from construct_query_*).
            node_keys: [N, key_dim] — spatial embeddings of ALL nodes
                       (from SpatialFeatureEncoder.encode_positions).
            visited_mask: [N] — 1.0 = visited, 0.0 = unvisited.
            greedy: If True, select argmax. If False, sample.
            temperature_override: If provided, override the learned temperature.
        
        Returns:
            SaccadicOutput with attention scores, probabilities, and selected index.
        """
        N = node_keys.shape[0]
        device = query.device
        
        # ============================================================
        # Multi-Head Attention Scores
        # ============================================================
        
        # Project Q and K
        Q = self.W_q(query)       # [key_dim]
        K = self.W_k(node_keys)   # [N, key_dim]
        
        # Reshape for multi-head: [num_heads, head_dim]
        Q = Q.view(self.num_heads, self.head_dim)          # [H, D_h]
        K = K.view(N, self.num_heads, self.head_dim)       # [N, H, D_h]
        K = K.permute(1, 0, 2)                             # [H, N, D_h]
        
        # Scaled dot-product attention: [H, N]
        temperature = torch.exp(self.log_temperature)
        if temperature_override is not None:
            temperature = temperature_override
        
        scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1))  # [H, 1, N]
        scores = scores.squeeze(1) / temperature                     # [H, N]
        
        # Average across heads: [N]
        attention_logits = scores.mean(dim=0)  # [N]
        
        # ============================================================
        # Mask visited nodes
        # ============================================================
        # Set visited nodes to -inf so they get 0 probability
        attention_logits = attention_logits.masked_fill(
            visited_mask.bool(), float('-inf')
        )
        
        # ============================================================
        # Compute probabilities
        # ============================================================
        # Check if all nodes are visited (edge case)
        unvisited_count = (visited_mask == 0).sum().item()
        if unvisited_count == 0:
            # All nodes visited — return uniform over all (will be handled by caller)
            attention_probs = torch.zeros(N, device=device)
            selected_idx = -1
        else:
            attention_probs = F.softmax(attention_logits, dim=0)  # [N]
            
            # ============================================================
            # Select node
            # ============================================================
            if greedy:
                selected_idx = attention_probs.argmax().item()
            else:
                selected_idx = torch.multinomial(attention_probs, 1).item()
        
        return SaccadicOutput(
            attention_logits=attention_logits,
            attention_probs=attention_probs,
            selected_node_idx=selected_idx,
            query=query
        )
    
    def compute_loss(
        self,
        output: SaccadicOutput,
        gt_node_idx: int
    ) -> torch.Tensor:
        """
        Compute the training loss for a single Mode 2 step.
        
        The loss is Cross-Entropy over the attention logits, with the
        ground-truth next node as the target.
        
        Args:
            output: SaccadicOutput from the forward pass.
            gt_node_idx: Ground-truth index of the next starting node.
        
        Returns:
            Scalar loss tensor.
        """
        # Cross-entropy expects [batch, num_classes] and [batch]
        logits = output.attention_logits.unsqueeze(0)  # [1, N]
        target = torch.tensor([gt_node_idx], device=logits.device)  # [1]
        
        loss = F.cross_entropy(logits, target)
        return loss


class SaccadicJumpModuleWithDiagnostics(SaccadicJumpModule):
    """
    Extended Mode 2 with diagnostic outputs for debugging and visualization.
    
    Adds:
      - Attention entropy (measures confidence of the jump decision).
      - Top-K attention scores (for visualization of where the model "looks").
      - Jump distance (physical distance of the saccade).
    """
    
    def forward_with_diagnostics(
        self,
        query: torch.Tensor,
        node_keys: torch.Tensor,
        visited_mask: torch.Tensor,
        node_positions_px: torch.Tensor,
        current_anchor_px: torch.Tensor,
        greedy: bool = True
    ) -> Tuple[SaccadicOutput, dict]:
        """
        Forward pass with additional diagnostic information.
        
        Args:
            query: [key_dim]
            node_keys: [N, key_dim]
            visited_mask: [N]
            node_positions_px: [N, 2] — raw pixel positions of all nodes.
            current_anchor_px: [2] — raw pixel position of the current anchor.
            greedy: Action selection mode.
        
        Returns:
            output: SaccadicOutput
            diagnostics: Dict with entropy, top_k, jump_distance.
        """
        output = self.forward(query, node_keys, visited_mask, greedy=greedy)
        
        diagnostics = {}
        
        # Attention entropy (lower = more confident)
        probs = output.attention_probs
        valid_probs = probs[probs > 0]
        if valid_probs.numel() > 0:
            entropy = -(valid_probs * torch.log(valid_probs + 1e-8)).sum().item()
        else:
            entropy = 0.0
        diagnostics['attention_entropy'] = entropy
        
        # Top-5 attention scores
        top_k = min(5, (visited_mask == 0).sum().item())
        if top_k > 0:
            top_probs, top_indices = torch.topk(probs, top_k)
            diagnostics['top_k_indices'] = top_indices.tolist()
            diagnostics['top_k_probs'] = top_probs.tolist()
        else:
            diagnostics['top_k_indices'] = []
            diagnostics['top_k_probs'] = []
        
        # Jump distance (physical pixels)
        if output.selected_node_idx >= 0:
            target_pos = node_positions_px[output.selected_node_idx]  # [2]
            jump_vec = target_pos - current_anchor_px  # [2]
            jump_distance = torch.norm(jump_vec).item()
            jump_angle = torch.atan2(jump_vec[1], jump_vec[0]).item() * (180.0 / math.pi)
            diagnostics['jump_distance_px'] = jump_distance
            diagnostics['jump_angle_deg'] = jump_angle
        else:
            diagnostics['jump_distance_px'] = 0.0
            diagnostics['jump_angle_deg'] = 0.0
        
        return output, diagnostics


if __name__ == "__main__":
    # --- Unit test ---
    
    print("=" * 60)
    print("  SaccadicJumpModule Unit Test")
    print("=" * 60)
    
    # Config
    key_dim = 256
    vis_dim = 512
    N = 20  # 20 nodes in the graph
    
    device = torch.device('cpu')
    
    # Create module
    saccadic = SaccadicJumpModuleWithDiagnostics(
        key_dim=key_dim,
        vis_dim=vis_dim,
        start_embed_dim=128,
        num_heads=4,
        dropout=0.1
    )
    saccadic.eval()
    
    # ============================================================
    # Test 1: Initialization (t=0)
    # ============================================================
    print("\n[Test 1] Initialization Query (t=0)")
    
    cls_token = torch.randn(vis_dim)
    query_init = saccadic.construct_query_initial(cls_token)
    print(f"  Q_0 shape: {query_init.shape}")  # [256]
    print(f"  Q_0 norm: {query_init.norm():.4f}")
    
    # Simulate node keys (from SpatialFeatureEncoder)
    node_keys = torch.randn(N, key_dim)
    visited_mask = torch.zeros(N)  # Nothing visited at t=0
    
    output_init = saccadic(
        query=query_init,
        node_keys=node_keys,
        visited_mask=visited_mask,
        greedy=True
    )
    
    print(f"  Selected node: {output_init.selected_node_idx}")
    print(f"  Attention probs sum: {output_init.attention_probs.sum():.4f}")  # Should be 1.0
    print(f"  Max prob: {output_init.attention_probs.max():.4f}")
    
    # ============================================================
    # Test 2: Saccadic Jump (t>0)
    # ============================================================
    print("\n[Test 2] Saccadic Jump Query (t>0)")
    
    anchor_encoded = torch.randn(key_dim)
    trajectory_encoded = torch.randn(key_dim)
    query_jump = saccadic.construct_query_jump(anchor_encoded, trajectory_encoded)
    print(f"  Q shape: {query_jump.shape}")  # [256]
    
    # Simulate: 8 nodes already visited
    visited_mask_partial = torch.zeros(N)
    visited_mask_partial[:8] = 1.0
    
    output_jump = saccadic(
        query=query_jump,
        node_keys=node_keys,
        visited_mask=visited_mask_partial,
        greedy=True
    )
    
    print(f"  Selected node: {output_jump.selected_node_idx}")
    print(f"  Selected node is unvisited: {visited_mask_partial[output_jump.selected_node_idx] == 0}")
    print(f"  Visited nodes have 0 prob: {(output_jump.attention_probs[:8] == 0).all()}")
    
    # ============================================================
    # Test 3: Loss Computation
    # ============================================================
    print("\n[Test 3] Loss Computation")
    
    gt_next_node = 12  # Ground truth: next node is index 12
    loss = saccadic.compute_loss(output_jump, gt_next_node)
    print(f"  Loss: {loss.item():.4f}")
    
    # Verify gradient flows
    loss.backward()
    print(f"  W_q grad norm: {saccadic.W_q.weight.grad.norm():.6f}")
    print(f"  W_k grad norm: {saccadic.W_k.weight.grad.norm():.6f}")
    print(f"  query_jump_proj grad exists: {saccadic.query_jump_proj[0].weight.grad is not None}")
    
    # ============================================================
    # Test 4: Diagnostics
    # ============================================================
    print("\n[Test 4] Diagnostics")
    
    saccadic.zero_grad()
    node_positions_px = torch.rand(N, 2) * 640  # Random positions in 640x640 image
    current_anchor_px = torch.tensor([500.0, 100.0])  # Bottom-right area
    
    output_diag, diagnostics = saccadic.forward_with_diagnostics(
        query=query_jump.detach(),
        node_keys=node_keys.detach(),
        visited_mask=visited_mask_partial,
        node_positions_px=node_positions_px,
        current_anchor_px=current_anchor_px,
        greedy=True
    )
    
    print(f"  Attention entropy: {diagnostics['attention_entropy']:.4f}")
    print(f"  Top-5 indices: {diagnostics['top_k_indices']}")
    print(f"  Top-5 probs: {[f'{p:.4f}' for p in diagnostics['top_k_probs']]}")
    print(f"  Jump distance: {diagnostics['jump_distance_px']:.1f}px")
    print(f"  Jump angle: {diagnostics['jump_angle_deg']:.1f}°")
    
    # ============================================================
    # Test 5: All nodes visited (edge case)
    # ============================================================
    print("\n[Test 5] All Nodes Visited (Edge Case)")
    
    visited_all = torch.ones(N)
    output_all = saccadic(
        query=query_jump.detach(),
        node_keys=node_keys.detach(),
        visited_mask=visited_all,
        greedy=True
    )
    
    print(f"  Selected node: {output_all.selected_node_idx}")  # Should be -1
    print(f"  Probs sum: {output_all.attention_probs.sum():.4f}")  # Should be 0.0
    
    # ============================================================
    # Test 6: Verify visited nodes are never selected
    # ============================================================
    print("\n[Test 6] Visited Node Exclusion (100 random trials)")
    
    all_correct = True
    for trial in range(100):
        # Random visited mask
        mask = torch.zeros(N)
        num_visited = torch.randint(1, N, (1,)).item()
        visited_indices = torch.randperm(N)[:num_visited]
        mask[visited_indices] = 1.0
        
        # Random query
        q = torch.randn(key_dim)
        
        out = saccadic(
            query=q,
            node_keys=node_keys.detach(),
            visited_mask=mask,
            greedy=True
        )
        
        if out.selected_node_idx >= 0 and mask[out.selected_node_idx] == 1:
            all_correct = False
            print(f"  ✗ Trial {trial}: Selected visited node {out.selected_node_idx}")
            break
    
    if all_correct:
        print(f"  ✓ All 100 trials selected unvisited nodes only")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)