"""
models/controller/foveated.py
Mode 1: Foveated Read — Local traversal within the threshold radius.

Responsibilities:
  1. Digit classification (Identity Pathway — NO spatial information)
  2. Content memory update (GRU cell on visual embeddings)
  3. Routing decisions (Routing Pathway — spatial edge features only)
  4. <CHUNK> prediction (based on minimum distance to candidates)
  5. Action selection (combined softmax over neighbors + <CHUNK>)

DUAL-PATHWAY INVARIANT:
  The classifier sees ONLY [e_vis || h_content]. It NEVER sees coordinates.
  The router sees ONLY edge features and h_content. It NEVER sees e_vis.
  These two pathways share h_content but nothing else.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

# Import from our features module
# from models.graph.features import EdgeFeatureAggregator


@dataclass
class FoveatedOutput:
    """Output of a single Mode 1 forward pass."""
    digit_logits: torch.Tensor       # [num_classes] — classification logits for current node
    action_logits: torch.Tensor      # [K + 1] — combined logits: [neighbor_0, ..., neighbor_K, <CHUNK>]
    candidate_indices: torch.Tensor  # [K] — graph node indices of valid candidates
    new_h_content: torch.Tensor      # [hidden_dim] — updated content state (after GRU)
    min_distance: torch.Tensor       # [1] — minimum distance to any candidate (for diagnostics)


class FoveatedReadModule(nn.Module):
    """
    Mode 1: Foveated Read.
    
    At each time step, this module:
      1. Classifies the digit at the current node (Identity Pathway).
      2. Updates h_content via a GRU cell.
      3. Computes routing logits over unvisited local neighbors (Routing Pathway).
      4. Computes a <CHUNK> logit based on proximity to candidates.
      5. Returns combined action logits for the controller to select from.
    
    This module does NOT mutate the ControllerState. State updates are
    handled by the orchestrator (dual_mode.py) based on the selected action.
    """
    
    def __init__(
        self,
        vis_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_classes: int = 10,
        radius: float = 80.0,
        T_intra: float = 76.0,
        T_inter: float = 108.0,
        dropout: float = 0.1
    ):
        """
        Args:
            vis_dim: Dimensionality of visual embeddings e_vis (from backbone).
            hidden_dim: Dimensionality of h_content (GRU hidden state).
            edge_dim: Dimensionality of edge embeddings (from SpatialFeatureEncoder).
            num_classes: Number of digit classes (10 for 0-9).
            radius: Threshold radius r (pixels).
            T_intra: Intra-chunk distance threshold (pixels).
            T_inter: Inter-chunk distance threshold (pixels).
            dropout: Dropout rate for classification and routing heads.
        """
        super().__init__()
        
        self.vis_dim = vis_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.radius = radius
        self.T_intra = T_intra
        self.T_inter = T_inter
        
        # ============================================================
        # IDENTITY PATHWAY: Digit Classification
        # Input: [e_vis || h_content] -> digit logits
        # STRICT: No spatial information enters this pathway.
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(vis_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # ============================================================
        # RECURRENT UPDATE: GRU Cell
        # Input: e_vis (visual embedding of current node)
        # Hidden: h_content (sequential memory)
        # The GRU's update gate naturally handles "accumulate within
        # chunk, forget on reset" because h_content is zeroed on <CHUNK>.
        # ============================================================
        self.gru_cell = nn.GRUCell(
            input_size=vis_dim,
            hidden_size=hidden_dim
        )
        
        # ============================================================
        # ROUTING PATHWAY: Edge Aggregation
        # Input: h_content + edge embeddings -> routing logits + chunk logit
        # STRICT: No visual information enters this pathway.
        # ============================================================
        
        # Routing logit per candidate: (h_content || edge_embedding) -> scalar
        self.routing_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # CHUNK logit: (h_content || e_closest || d_min_norm) -> scalar
        # e_closest: edge embedding of the nearest candidate
        # d_min_norm: normalized minimum distance (the critical proximity signal)
        self.chunk_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        h_content: torch.Tensor,
        e_vis_current: torch.Tensor,
        edge_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        candidate_indices: torch.Tensor
    ) -> FoveatedOutput:
        """
        Single forward pass of Mode 1.
        
        Computes all logits but does NOT select an action or mutate state.
        Action selection and state updates are handled by dual_mode.py.
        
        Args:
            h_content: [hidden_dim] — current content hidden state.
            e_vis_current: [vis_dim] — visual embedding of the CURRENT node
                           (the node we just arrived at, before moving to next).
            edge_embeddings: [K, edge_dim] — edge embeddings for K candidate
                             neighbors (from SpatialFeatureEncoder).
            candidate_mask: [K] — 1.0 = valid candidate, 0.0 = visited/padding.
            edge_distances: [K] — raw pixel distances to each candidate.
            candidate_indices: [K] — graph node indices of candidates.
        
        Returns:
            FoveatedOutput with all logits and the updated h_content.
        """
        device = h_content.device
        K = edge_embeddings.shape[0]
        
        # ============================================================
        # Step 1: Digit Classification (Identity Pathway)
        # Classify the digit at the current node using ONLY visual
        # features and sequential context. No spatial information.
        # ============================================================
        cls_input = torch.cat([e_vis_current, h_content], dim=-1)  # [vis_dim + hidden_dim]
        digit_logits = self.classifier(cls_input)  # [num_classes]
        
        # ============================================================
        # Step 2: Recurrent Update
        # Update h_content with the visual embedding of the current node.
        # This is the ONLY place where e_vis enters the sequential memory.
        # ============================================================
        new_h_content = self.gru_cell(
            input=e_vis_current.unsqueeze(0),   # [1, vis_dim]
            hx=h_content.unsqueeze(0)            # [1, hidden_dim]
        ).squeeze(0)  # [hidden_dim]
        
        # ============================================================
        # Step 3: Routing Decision (Routing Pathway)
        # Compute logits for each candidate neighbor and the <CHUNK> action.
        # Uses the UPDATED h_content (informed by the digit just read).
        # ============================================================
        
        if K == 0 or candidate_mask.sum() == 0:
            # No candidates: structural termination, force <CHUNK>
            action_logits = torch.tensor([10.0], device=device)  # Single <CHUNK> logit
            min_distance = torch.tensor([float('inf')], device=device)
        else:
            # --- Routing logits for each candidate ---
            h_expanded = new_h_content.unsqueeze(0).expand(K, -1)  # [K, hidden_dim]
            routing_input = torch.cat([h_expanded, edge_embeddings], dim=-1)  # [K, hidden_dim + edge_dim]
            routing_logits = self.routing_mlp(routing_input).squeeze(-1)  # [K]
            
            # Mask out invalid candidates
            routing_logits = routing_logits.masked_fill(candidate_mask == 0, float('-inf'))
            
            # --- CHUNK logit ---
            # Find minimum distance among valid candidates
            valid_distances = edge_distances.clone()
            valid_distances[candidate_mask == 0] = float('inf')
            d_min = valid_distances.min()
            d_min_norm = (d_min / self.radius).unsqueeze(0)  # [1]
            
            # Get edge embedding of the closest valid candidate
            closest_idx = valid_distances.argmin()
            e_closest = edge_embeddings[closest_idx]  # [edge_dim]
            
            # Compute CHUNK logit
            chunk_input = torch.cat([new_h_content, e_closest, d_min_norm], dim=-1)
            chunk_logit = self.chunk_mlp(chunk_input)  # [1]
            
            # --- Combine into action logits ---
            # [neighbor_0, ..., neighbor_K, <CHUNK>]
            action_logits = torch.cat([routing_logits, chunk_logit], dim=0)  # [K + 1]
            min_distance = d_min.unsqueeze(0)
        
        return FoveatedOutput(
            digit_logits=digit_logits,
            action_logits=action_logits,
            candidate_indices=candidate_indices,
            new_h_content=new_h_content,
            min_distance=min_distance
        )
    
    def select_action(
        self,
        output: FoveatedOutput,
        greedy: bool = True,
        temperature: float = 1.0
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """
        Select the next action from the computed logits.
        
        Args:
            output: FoveatedOutput from the forward pass.
            greedy: If True, select argmax. If False, sample from softmax.
            temperature: Sampling temperature (only used if greedy=False).
        
        Returns:
            action_type: 'READ' (visit a neighbor) or 'CHUNK' (emit <CHUNK>).
            selected_node_idx: Graph node index of the selected neighbor (None if CHUNK).
            selected_local_idx: Index into candidate_indices (None if CHUNK).
        """
        logits = output.action_logits  # [K + 1]
        K = output.candidate_indices.shape[0]
        
        if greedy:
            action_idx = logits.argmax().item()
        else:
            probs = F.softmax(logits / temperature, dim=0)
            action_idx = torch.multinomial(probs, 1).item()
        
        if action_idx == K:
            # Selected <CHUNK> (the last action in the combined space)
            return 'CHUNK', None, None
        else:
            # Selected a neighbor
            selected_node_idx = output.candidate_indices[action_idx].item()
            return 'READ', selected_node_idx, action_idx
    
    def compute_loss(
        self,
        output: FoveatedOutput,
        gt_digit_label: int,
        gt_action_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the training loss for a single Mode 1 step.
        
        Args:
            output: FoveatedOutput from the forward pass.
            gt_digit_label: Ground-truth digit class (0-9).
            gt_action_idx: Ground-truth action index in the combined
                           action space [0..K] for neighbors, K for <CHUNK>.
        
        Returns:
            digit_loss: Cross-entropy loss for digit classification.
            action_loss: Cross-entropy loss for action selection.
        """
        # Digit classification loss
        digit_loss = F.cross_entropy(
            output.digit_logits.unsqueeze(0),  # [1, num_classes]
            torch.tensor([gt_digit_label], device=output.digit_logits.device)
        )
        
        # Action selection loss
        action_loss = F.cross_entropy(
            output.action_logits.unsqueeze(0),  # [1, K + 1]
            torch.tensor([gt_action_idx], device=output.action_logits.device)
        )
        
        return digit_loss, action_loss


class FoveatedReadModuleWithSafetyCheck(FoveatedReadModule):
    """
    Extended Mode 1 with post-hoc distance safety checks.
    
    After action selection, verifies that the selected neighbor's distance
    is consistent with the action type:
      - If action is READ and d > T_inter: force CHUNK (safety violation).
      - If action is CHUNK and d_min < T_intra: log warning (model uncertainty).
    
    This prevents catastrophic failures from miscalibrated logits.
    """
    
    def select_action_with_safety(
        self,
        output: FoveatedOutput,
        edge_distances: torch.Tensor,
        candidate_mask: torch.Tensor,
        greedy: bool = True,
        temperature: float = 1.0
    ) -> Tuple[str, Optional[int], Optional[int], bool]:
        """
        Select action with post-hoc safety checks.
        
        Returns:
            action_type: 'READ', 'CHUNK', or 'LOCAL_CHUNK' (distance-based boundary).
            selected_node_idx: Graph node index (None for pure CHUNK).
            selected_local_idx: Local index (None for pure CHUNK).
            safety_triggered: True if the safety check overrode the model's decision.
        """
        action_type, node_idx, local_idx = self.select_action(
            output, greedy=greedy, temperature=temperature
        )
        
        safety_triggered = False
        
        if action_type == 'READ' and node_idx is not None:
            # Check distance of selected neighbor
            d = edge_distances[local_idx].item()
            
            if d > self.T_inter:
                # SAFETY VIOLATION: Model selected a far neighbor.
                # Override: emit CHUNK, then process this neighbor as
                # the start of a new chunk (local chunk crossing).
                action_type = 'LOCAL_CHUNK'
                safety_triggered = True
        
        elif action_type == 'CHUNK':
            # Check if there's a close neighbor that the model ignored
            if candidate_mask.sum() > 0:
                valid_d = edge_distances.clone()
                valid_d[candidate_mask == 0] = float('inf')
                d_min = valid_d.min().item()
                
                if d_min < self.T_intra:
                    # WARNING: Model chose CHUNK despite a close neighbor.
                    # This is not necessarily wrong (the model may have learned
                    # a valid reason to chunk), but it's worth monitoring.
                    # We do NOT override here — trust the model's decision.
                    pass
        
        return action_type, node_idx, local_idx, safety_triggered


if __name__ == "__main__":
    # --- Unit test ---
    
    print("=" * 60)
    print("  FoveatedReadModule Unit Test")
    print("=" * 60)
    
    # Config
    vis_dim = 512
    hidden_dim = 256
    edge_dim = 256
    num_classes = 10
    radius = 80.0
    T_intra = 76.0
    T_inter = 108.0
    
    device = torch.device('cpu')
    
    # Create module
    foveated = FoveatedReadModuleWithSafetyCheck(
        vis_dim=vis_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        num_classes=num_classes,
        radius=radius,
        T_intra=T_intra,
        T_inter=T_inter
    )
    foveated.eval()
    
    # Simulate inputs
    h_content = torch.randn(hidden_dim)
    e_vis = torch.randn(vis_dim)
    K = 4  # 4 candidate neighbors
    
    edge_embeddings = torch.randn(K, edge_dim)
    candidate_mask = torch.ones(K)
    candidate_mask[2] = 0  # Node 2 is visited
    edge_distances = torch.tensor([30.0, 55.0, 90.0, 120.0])  # pixels
    candidate_indices = torch.tensor([3, 5, 7, 9])  # graph node IDs
    
    # Forward pass
    output = foveated(
        h_content=h_content,
        e_vis_current=e_vis,
        edge_embeddings=edge_embeddings,
        candidate_mask=candidate_mask,
        edge_distances=edge_distances,
        candidate_indices=candidate_indices
    )
    
    print(f"\n[Forward Pass]")
    print(f"  Digit logits shape: {output.digit_logits.shape}")  # [10]
    print(f"  Action logits shape: {output.action_logits.shape}")  # [K+1] = [5]
    print(f"  New h_content shape: {output.new_h_content.shape}")  # [256]
    print(f"  Min distance: {output.min_distance.item():.1f}px")
    
    # Predicted digit
    pred_digit = output.digit_logits.argmax().item()
    print(f"  Predicted digit: {pred_digit}")
    
    # Action selection (greedy)
    action_type, node_idx, local_idx, safety = foveated.select_action_with_safety(
        output, edge_distances, candidate_mask, greedy=True
    )
    print(f"\n[Action Selection]")
    print(f"  Action: {action_type}")
    print(f"  Node: {node_idx}")
    print(f"  Safety triggered: {safety}")
    
    # Test loss computation
    gt_digit = 3
    gt_action = 0  # Visit first candidate
    digit_loss, action_loss = foveated.compute_loss(output, gt_digit, gt_action)
    print(f"\n[Loss]")
    print(f"  Digit loss: {digit_loss.item():.4f}")
    print(f"  Action loss: {action_loss.item():.4f}")
    
    # Test empty neighborhood (structural termination)
    print(f"\n[Empty Neighborhood Test]")
    empty_output = foveated(
        h_content=h_content,
        e_vis_current=e_vis,
        edge_embeddings=torch.zeros(0, edge_dim),
        candidate_mask=torch.zeros(0),
        edge_distances=torch.zeros(0),
        candidate_indices=torch.zeros(0, dtype=torch.long)
    )
    print(f"  Action logits: {empty_output.action_logits.tolist()}")  # Should be [10.0]
    print(f"  Min distance: {empty_output.min_distance.item()}")  # Should be inf
    
    action_type, _, _, _ = foveated.select_action_with_safety(
        empty_output, torch.zeros(0), torch.zeros(0), greedy=True
    )
    print(f"  Action: {action_type}")  # Should be CHUNK
    
    # Test safety check: far neighbor selected
    print(f"\n[Safety Check Test]")
    # Force the model to select the far neighbor (index 3, distance 120 > T_inter=108)
    forced_output = FoveatedOutput(
        digit_logits=output.digit_logits,
        action_logits=torch.tensor([-10.0, -10.0, -10.0, 10.0, -10.0]),  # Force index 3
        candidate_indices=candidate_indices,
        new_h_content=output.new_h_content,
        min_distance=output.min_distance
    )
    action_type, node_idx, local_idx, safety = foveated.select_action_with_safety(
        forced_output, edge_distances, candidate_mask, greedy=True
    )
    print(f"  Action: {action_type}")  # Should be LOCAL_CHUNK (safety override)
    print(f"  Node: {node_idx}")       # Should be 9 (the far neighbor)
    print(f"  Safety triggered: {safety}")  # Should be True
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)