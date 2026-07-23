"""
models/controller/dual_mode.py
Dual-Mode Cognitive Controller — The top-level orchestrator.

Ties together:
  - ControllerState (state.py)
  - FoveatedReadModule (foveated.py) — Mode 1
  - SaccadicJumpModule (saccadic.py) — Mode 2
  - SpatialFeatureEncoder (features.py)
  - ThresholdRadiusGraphBuilder (builder.py)

Execution Flow:
  1. Backbone extracts e_vis for all nodes + CLS token (done externally).
  2. SpatialFeatureEncoder pre-computes node keys and edge embeddings.
  3. Mode 2 (Saccadic Jump) finds the first node.
  4. Mode 1 (Foveated Read) reads digits locally until <CHUNK>.
  5. Mode 2 jumps to the next chunk start.
  6. Repeat until all nodes are visited or max_steps is reached.

Two entry points:
  - forward_train(): Teacher-forced training with ground-truth actions.
  - forward_inference(): Autoregressive decoding with greedy/sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from models.controller.state import ControllerState, ControllerMode
from models.controller.foveated import FoveatedReadModule, FoveatedOutput
from models.controller.saccadic import SaccadicJumpModule, SaccadicOutput
from models.graph.features import SpatialFeatureEncoder
from models.graph.builder import SpatialGraph, ThresholdRadiusGraphBuilder


@dataclass
class TrainingStep:
    """A single ground-truth training step parsed from the GT sequence."""
    mode: str                    # 'JUMP' or 'READ'
    node_id: int                 # Node to arrive at
    digit_label: int             # Digit class (0-9) for READ steps, -1 for JUMP
    action: Optional[str]        # 'READ' or 'CHUNK' (departure action for READ steps)
    action_target_node: Optional[int]  # Target node for READ action, None for CHUNK


@dataclass
class DualModeOutput:
    """Complete output of a training or inference run."""
    total_loss: Optional[torch.Tensor]
    digit_loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    jump_loss: Optional[torch.Tensor]
    predicted_sequence: List[str]
    num_steps: int
    num_digits: int
    num_chunks: int
    state: ControllerState


class DualModeController(nn.Module):
    """
    The complete Dual-Mode Cognitive Controller.
    
    This is the top-level module that orchestrates Mode 1 and Mode 2,
    manages the ControllerState, and computes the training loss.
    """
    
    def __init__(
        self,
        vis_dim: int = 512,
        hidden_dim: int = 256,
        edge_dim: int = 256,
        key_dim: int = 256,
        num_classes: int = 10,
        radius: float = 80.0,
        T_intra: float = 76.0,
        T_inter: float = 108.0,
        num_frequencies: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.T_intra = T_intra
        self.T_inter = T_inter
        
        # Loss weights for multi-task training
        self.loss_weights = loss_weights or {
            'digit': 1.0,
            'action': 1.0,
            'jump': 1.0
        }
        
        # ============================================================
        # Sub-modules
        # ============================================================
        
        # Spatial Feature Encoder (Routing Pathway)
        self.spatial_encoder = SpatialFeatureEncoder(
            num_frequencies=num_frequencies,
            output_dim=key_dim,
            use_learnable_projection=True
        )
        
        # Mode 1: Foveated Read
        self.foveated = FoveatedReadModule(
            vis_dim=vis_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_classes=num_classes,
            radius=radius,
            T_intra=T_intra,
            T_inter=T_inter,
            dropout=dropout
        )
        
        # Mode 2: Saccadic Jump
        self.saccadic = SaccadicJumpModule(
            key_dim=key_dim,
            vis_dim=vis_dim,
            start_embed_dim=128,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Graph builder (used at inference to construct graph from detector output)
        self.graph_builder = ThresholdRadiusGraphBuilder(
            radius=radius,
            img_width=640,   # Will be overridden per-image
            img_height=640
        )
    
    # ==============================================================
    # FEATURE PRE-COMPUTATION
    # ==============================================================
    
    def _prepare_graph_features(
        self, graph: SpatialGraph
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-compute all spatial features for the graph.
        Called once per image, before the reading loop.
        
        Returns dict with:
          - node_keys: [N, key_dim] — for Mode 2 attention
          - edge_embeddings: [N, N, edge_dim] — for Mode 1 routing
          - edge_distances: [N, N] — raw pixel distances
        """
        with torch.no_grad():
            # Encode node positions for Mode 2 keys
            node_keys = self.spatial_encoder.encode_positions(
                graph.node_positions_norm
            )  # [N, key_dim]
            
            # Encode edge features for Mode 1 routing
            edge_embeddings = self.spatial_encoder.encode_edges(
                graph.edge_features,
                radius=self.radius
            )  # [N, N, edge_dim]
            
            # Extract raw pixel distances
            edge_distances = graph.edge_features[:, :, 2]  # [N, N]
        
        return {
            'node_keys': node_keys,
            'edge_embeddings': edge_embeddings,
            'edge_distances': edge_distances
        }
    
    # ==============================================================
    # GROUND-TRUTH PARSING
    # ==============================================================
    
    def parse_gt_sequence(
        self, gt_sequence: List[Dict]
    ) -> List[TrainingStep]:
        """
        Parse the ground-truth sequence from the data generator into
        a list of TrainingSteps for teacher-forced training.
        
        Input format (from data/generator.py):
          [{token: '3', node_id: 0, mode: 'READ'},
           {token: '8', node_id: 1, mode: 'READ'},
           {token: '<CHUNK>', node_id: None, mode: 'CHUNK'},
           {token: '1', node_id: 2, mode: 'READ'}, ...]
        
        Output format:
          [TrainingStep(mode='JUMP', node_id=0, digit_label=3, action='READ', target=1),
           TrainingStep(mode='READ', node_id=1, digit_label=8, action='CHUNK', target=None),
           TrainingStep(mode='JUMP', node_id=2, digit_label=1, action='READ', target=3), ...]
        """
        steps = []
        
        # Extract only digit tokens (skip <CHUNK> tokens)
        digit_tokens = [
            t for t in gt_sequence if t['mode'] == 'READ'
        ]
        
        # Build a mapping: node_id -> position in digit_tokens
        node_to_pos = {t['node_id']: i for i, t in enumerate(digit_tokens)}
        
        # Determine chunk boundaries from the full sequence
        # A <CHUNK> token between digit i and digit i+1 means they are in different chunks
        chunk_boundaries = set()  # Set of digit positions where a chunk boundary occurs
        digit_pos = 0
        for token in gt_sequence:
            if token['mode'] == 'CHUNK':
                if digit_pos > 0:
                    chunk_boundaries.add(digit_pos)
            elif token['mode'] == 'READ':
                digit_pos += 1
        
        # Build training steps
        for i, dt in enumerate(digit_tokens):
            node_id = dt['node_id']
            digit_label = int(dt['token'])
            
            # Determine if this is a JUMP step (first digit or after a chunk boundary)
            is_jump = (i == 0) or (i in chunk_boundaries)
            
            # Determine departure action
            if i + 1 in chunk_boundaries or i == len(digit_tokens) - 1:
                # Last digit in chunk or last digit overall
                action = 'CHUNK'
                action_target = None
            else:
                action = 'READ'
                action_target = digit_tokens[i + 1]['node_id']
            
            steps.append(TrainingStep(
                mode='JUMP' if is_jump else 'READ',
                node_id=node_id,
                digit_label=digit_label,
                action=action,
                action_target_node=action_target
            ))
        
        return steps
    
    # ==============================================================
    # TRAINING: TEACHER FORCING
    # ==============================================================
    
    def forward_train(
        self,
        graph: SpatialGraph,
        gt_sequence: List[Dict],
        cls_token: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ) -> DualModeOutput:
        """
        Teacher-forced training loop.
        
        Args:
            graph: SpatialGraph with node_embeddings already filled by backbone.
            gt_sequence: Ground-truth sequence from data generator.
            cls_token: [vis_dim] global image feature from backbone GAP.
            device: Torch device.
        
        Returns:
            DualModeOutput with total loss and diagnostics.
        """
        N = graph.num_nodes
        
        # Parse ground truth
        gt_steps = self.parse_gt_sequence(gt_sequence)
        
        # Pre-compute spatial features
        features = self._prepare_graph_features(graph)
        node_keys = features['node_keys'].to(device)
        edge_embeddings = features['edge_embeddings'].to(device)
        edge_distances = features['edge_distances'].to(device)
        
        # Initialize state
        state = ControllerState.initialize(
            num_nodes=N, hidden_dim=self.hidden_dim, device=device
        )
        
        # Accumulate losses
        total_digit_loss = torch.tensor(0.0, device=device)
        total_action_loss = torch.tensor(0.0, device=device)
        total_jump_loss = torch.tensor(0.0, device=device)
        num_digit_steps = 0
        num_action_steps = 0
        num_jump_steps = 0
        
        # Iterate through ground-truth steps
        for step_idx, gt_step in enumerate(gt_steps):
            
            if gt_step.mode == 'JUMP':
                # ============================================
                # MODE 2: Saccadic Jump
                # ============================================
                
                # Construct query
                if not state.initialized:
                    # t=0: use CLS + START
                    query = self.saccadic.construct_query_initial(cls_token.to(device))
                else:
                    # t>0: use anchor + trajectory
                    anchor_enc = self.spatial_encoder.encode_anchor(
                        state.get_anchor_for_query()
                    )
                    traj_enc = self.spatial_encoder.encode_trajectory(
                        state.get_trajectory_for_query()
                    )
                    query = self.saccadic.construct_query_jump(anchor_enc, traj_enc)
                
                # Forward pass
                saccadic_out = self.saccadic(
                    query=query,
                    node_keys=node_keys,
                    visited_mask=state.visited_mask,
                    greedy=True
                )
                
                # Jump loss
                jump_loss = self.saccadic.compute_loss(saccadic_out, gt_step.node_id)
                total_jump_loss = total_jump_loss + jump_loss
                num_jump_steps += 1
                
                # Update state: jump to the ground-truth node
                node_pos_norm = graph.node_positions_norm[gt_step.node_id].to(device)
                node_pos_px = graph.node_positions_px[gt_step.node_id].to(device)
                state.update_after_jump(
                    node_idx=gt_step.node_id,
                    node_pos_norm=node_pos_norm,
                    node_pos_px=node_pos_px
                )
                
                # Now read the digit at this node (Mode 1 classification)
                e_vis = graph.node_embeddings[gt_step.node_id].to(device)
                
                # Get unvisited neighbors for Mode 1
                neighbors = self.graph_builder.get_unvisited_neighbors(
                    graph, gt_step.node_id, state.visited_mask
                )
                
                if len(neighbors) > 0:
                    # Prepare edge features for candidates
                    cand_edge_emb = edge_embeddings[gt_step.node_id][neighbors]
                    cand_mask = torch.ones(len(neighbors), device=device)
                    cand_dist = edge_distances[gt_step.node_id][neighbors]
                    cand_indices = neighbors.to(device)
                else:
                    cand_edge_emb = torch.zeros(0, edge_embeddings.shape[-1], device=device)
                    cand_mask = torch.zeros(0, device=device)
                    cand_dist = torch.zeros(0, device=device)
                    cand_indices = torch.zeros(0, dtype=torch.long, device=device)
                
                # Mode 1 forward (for digit classification only at this step)
                foveated_out = self.foveated(
                    h_content=state.h_content,
                    e_vis_current=e_vis,
                    edge_embeddings=cand_edge_emb,
                    candidate_mask=cand_mask,
                    edge_distances=cand_dist,
                    candidate_indices=cand_indices
                )
                
                # Digit classification loss
                digit_loss = F.cross_entropy(
                    foveated_out.digit_logits.unsqueeze(0),
                    torch.tensor([gt_step.digit_label], device=device)
                )
                total_digit_loss = total_digit_loss + digit_loss
                num_digit_steps += 1
                
                # Update h_content (but don't change mode or visited mask again)
                state.h_content = foveated_out.new_h_content
                
                # Determine action loss
                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
                    # Find the index of the target node in candidates
                    target_local_idx = (cand_indices == gt_step.action_target_node).nonzero(as_tuple=True)
                    if len(target_local_idx[0]) > 0:
                        gt_action_idx = target_local_idx[0][0].item()
                    else:
                        gt_action_idx = len(cand_indices)  # <CHUNK> index (fallback)
                else:
                    gt_action_idx = len(cand_indices)  # <CHUNK> is the last action
                
                # Action loss
                action_loss = F.cross_entropy(
                    foveated_out.action_logits.unsqueeze(0),
                    torch.tensor([gt_action_idx], device=device)
                )
                total_action_loss = total_action_loss + action_loss
                num_action_steps += 1
                
                # If action is CHUNK, update state
                if gt_step.action == 'CHUNK':
                    state.update_after_chunk()
            
            elif gt_step.mode == 'READ':
                # ============================================
                # MODE 1: Foveated Read (continuation)
                # ============================================
                
                current_node = state.current_node
                e_vis = graph.node_embeddings[gt_step.node_id].to(device)
                
                # Get unvisited neighbors
                neighbors = self.graph_builder.get_unvisited_neighbors(
                    graph, current_node, state.visited_mask
                )
                
                if len(neighbors) > 0:
                    cand_edge_emb = edge_embeddings[current_node][neighbors]
                    cand_mask = torch.ones(len(neighbors), device=device)
                    cand_dist = edge_distances[current_node][neighbors]
                    cand_indices = neighbors.to(device)
                else:
                    cand_edge_emb = torch.zeros(0, edge_embeddings.shape[-1], device=device)
                    cand_mask = torch.zeros(0, device=device)
                    cand_dist = torch.zeros(0, device=device)
                    cand_indices = torch.zeros(0, dtype=torch.long, device=device)
                
                # Mode 1 forward
                foveated_out = self.foveated(
                    h_content=state.h_content,
                    e_vis_current=e_vis,
                    edge_embeddings=cand_edge_emb,
                    candidate_mask=cand_mask,
                    edge_distances=cand_dist,
                    candidate_indices=cand_indices
                )
                
                # Digit classification loss
                digit_loss = F.cross_entropy(
                    foveated_out.digit_logits.unsqueeze(0),
                    torch.tensor([gt_step.digit_label], device=device)
                )
                total_digit_loss = total_digit_loss + digit_loss
                num_digit_steps += 1
                
                # Determine ground-truth action index
                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
                    target_local_idx = (cand_indices == gt_step.action_target_node).nonzero(as_tuple=True)
                    if len(target_local_idx[0]) > 0:
                        gt_action_idx = target_local_idx[0][0].item()
                    else:
                        gt_action_idx = len(cand_indices)
                else:
                    gt_action_idx = len(cand_indices)  # <CHUNK>
                
                # Action loss
                action_loss = F.cross_entropy(
                    foveated_out.action_logits.unsqueeze(0),
                    torch.tensor([gt_action_idx], device=device)
                )
                total_action_loss = total_action_loss + action_loss
                num_action_steps += 1
                
                # Update state based on ground-truth action
                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
                    # Move to the target node
                    target_node = gt_step.action_target_node
                    node_pos_norm = graph.node_positions_norm[target_node].to(device)
                    node_pos_px = graph.node_positions_px[target_node].to(device)
                    
                    state.update_after_read(
                        node_idx=target_node,
                        node_pos_norm=node_pos_norm,
                        node_pos_px=node_pos_px,
                        new_h_content=foveated_out.new_h_content,
                        digit_token=str(gt_step.digit_label)
                    )
                else:
                    # CHUNK
                    state.h_content = foveated_out.new_h_content
                    state.update_after_chunk()
        
        # Compute weighted total loss
        avg_digit_loss = total_digit_loss / max(num_digit_steps, 1)
        avg_action_loss = total_action_loss / max(num_action_steps, 1)
        avg_jump_loss = total_jump_loss / max(num_jump_steps, 1)
        
        total_loss = (
            self.loss_weights['digit'] * avg_digit_loss +
            self.loss_weights['action'] * avg_action_loss +
            self.loss_weights['jump'] * avg_jump_loss
        )
        
        return DualModeOutput(
            total_loss=total_loss,
            digit_loss=avg_digit_loss,
            action_loss=avg_action_loss,
            jump_loss=avg_jump_loss,
            predicted_sequence=state.get_output_sequence(),
            num_steps=state.step,
            num_digits=state.total_digits_read,
            num_chunks=state.output_tokens.count('<CHUNK>'),
            state=state
        )
    
    # ==============================================================
    # INFERENCE: AUTOREGRESSIVE DECODING
    # ==============================================================
    
    @torch.no_grad()
    def forward_inference(
        self,
        graph: SpatialGraph,
        cls_token: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        max_steps: int = 500,
        greedy: bool = True,
        temperature: float = 1.0
    ) -> DualModeOutput:
        """
        Autoregressive decoding loop.
        
        Args:
            graph: SpatialGraph with node_embeddings filled by backbone.
            cls_token: [vis_dim] global image feature.
            device: Torch device.
            max_steps: Maximum number of steps before forced termination.
            greedy: If True, use argmax. If False, sample.
            temperature: Sampling temperature.
        
        Returns:
            DualModeOutput with predicted sequence.
        """
        N = graph.num_nodes
        
        # Pre-compute spatial features
        features = self._prepare_graph_features(graph)
        node_keys = features['node_keys'].to(device)
        edge_embeddings = features['edge_embeddings'].to(device)
        edge_distances = features['edge_distances'].to(device)
        
        # Initialize state
        state = ControllerState.initialize(
            num_nodes=N, hidden_dim=self.hidden_dim, device=device
        )
        
        step_count = 0
        
        while not state.terminated and step_count < max_steps:
            
            if state.mode == ControllerMode.SACCADIC_JUMP:
                # ============================================
                # MODE 2: Saccadic Jump
                # ============================================
                
                # Check if all nodes are visited
                if state.all_visited():
                    state.terminate()
                    break
                
                # Construct query
                if not state.initialized:
                    query = self.saccadic.construct_query_initial(cls_token.to(device))
                else:
                    anchor_enc = self.spatial_encoder.encode_anchor(
                        state.get_anchor_for_query()
                    )
                    traj_enc = self.spatial_encoder.encode_trajectory(
                        state.get_trajectory_for_query()
                    )
                    query = self.saccadic.construct_query_jump(anchor_enc, traj_enc)
                
                # Forward pass
                saccadic_out = self.saccadic(
                    query=query,
                    node_keys=node_keys,
                    visited_mask=state.visited_mask,
                    greedy=greedy
                )
                
                if saccadic_out.selected_node_idx < 0:
                    state.terminate()
                    break
                
                # Jump to selected node
                selected = saccadic_out.selected_node_idx
                node_pos_norm = graph.node_positions_norm[selected].to(device)
                node_pos_px = graph.node_positions_px[selected].to(device)
                state.update_after_jump(selected, node_pos_norm, node_pos_px)
                
                # Read the digit at the selected node
                e_vis = graph.node_embeddings[selected].to(device)
                
                # Get neighbors for Mode 1
                neighbors = self.graph_builder.get_unvisited_neighbors(
                    graph, selected, state.visited_mask
                )
                
                if len(neighbors) > 0:
                    cand_edge_emb = edge_embeddings[selected][neighbors]
                    cand_mask = torch.ones(len(neighbors), device=device)
                    cand_dist = edge_distances[selected][neighbors]
                    cand_indices = neighbors.to(device)
                else:
                    cand_edge_emb = torch.zeros(0, edge_embeddings.shape[-1], device=device)
                    cand_mask = torch.zeros(0, device=device)
                    cand_dist = torch.zeros(0, device=device)
                    cand_indices = torch.zeros(0, dtype=torch.long, device=device)
                
                # Classify digit
                foveated_out = self.foveated(
                    h_content=state.h_content,
                    e_vis_current=e_vis,
                    edge_embeddings=cand_edge_emb,
                    candidate_mask=cand_mask,
                    edge_distances=cand_dist,
                    candidate_indices=cand_indices
                )
                
                pred_digit = foveated_out.digit_logits.argmax().item()
                state.h_content = foveated_out.new_h_content
                
                # Record the digit
                state.output_tokens.append({
                    'token': str(pred_digit),
                    'node_id': selected,
                    'mode': 'READ',
                    'step': state.step,
                    'chunk_size': state.chunk_size
                })
                
                # Select action
                action_type, node_idx, local_idx = self.foveated.select_action(
                    foveated_out, greedy=greedy, temperature=temperature
                )
                
                if action_type == 'CHUNK':
                    state.update_after_chunk()
                elif action_type == 'READ' and node_idx is not None:
                    next_pos_norm = graph.node_positions_norm[node_idx].to(device)
                    next_pos_px = graph.node_positions_px[node_idx].to(device)
                    state.update_after_read(
                        node_idx=node_idx,
                        node_pos_norm=next_pos_norm,
                        node_pos_px=next_pos_px,
                        new_h_content=state.h_content,
                        digit_token=str(pred_digit)
                    )
            
            elif state.mode == ControllerMode.FOVEATED_READ:
                # ============================================
                # MODE 1: Foveated Read
                # ============================================
                
                current = state.current_node
                e_vis = graph.node_embeddings[current].to(device)
                
                # Get unvisited neighbors
                neighbors = self.graph_builder.get_unvisited_neighbors(
                    graph, current, state.visited_mask
                )
                
                if len(neighbors) > 0:
                    cand_edge_emb = edge_embeddings[current][neighbors]
                    cand_mask = torch.ones(len(neighbors), device=device)
                    cand_dist = edge_distances[current][neighbors]
                    cand_indices = neighbors.to(device)
                else:
                    cand_edge_emb = torch.zeros(0, edge_embeddings.shape[-1], device=device)
                    cand_mask = torch.zeros(0, device=device)
                    cand_dist = torch.zeros(0, device=device)
                    cand_indices = torch.zeros(0, dtype=torch.long, device=device)
                
                # Forward pass
                foveated_out = self.foveated(
                    h_content=state.h_content,
                    e_vis_current=e_vis,
                    edge_embeddings=cand_edge_emb,
                    candidate_mask=cand_mask,
                    edge_distances=cand_dist,
                    candidate_indices=cand_indices
                )
                
                # Select action
                action_type, node_idx, local_idx = self.foveated.select_action(
                    foveated_out, greedy=greedy, temperature=temperature
                )
                
                if action_type == 'CHUNK':
                    # Check if there are unvisited neighbors for local chunk crossing
                    if len(neighbors) > 0:
                        # Local chunk crossing: stay in Mode 1
                        closest_idx = cand_dist.argmin().item()
                        closest_node = cand_indices[closest_idx].item()
                        next_pos_norm = graph.node_positions_norm[closest_node].to(device)
                        next_pos_px = graph.node_positions_px[closest_node].to(device)
                        state.update_after_local_chunk(
                            closest_node, next_pos_norm, next_pos_px
                        )
                        # Read the digit at the new chunk start
                        e_vis_new = graph.node_embeddings[closest_node].to(device)
                        # (Digit will be classified in the next Mode 1 iteration)
                    else:
                        # Structural termination: switch to Mode 2
                        state.update_after_chunk()
                
                elif action_type == 'READ' and node_idx is not None:
                    # Visit the selected neighbor
                    next_pos_norm = graph.node_positions_norm[node_idx].to(device)
                    next_pos_px = graph.node_positions_px[node_idx].to(device)
                    
                    # Classify the digit at the neighbor
                    e_vis_next = graph.node_embeddings[node_idx].to(device)
                    
                    # We need to classify the digit at the NEXT node
                    # But the current foveated_out classified the CURRENT node
                    # So we just update state and classify in the next iteration
                    state.update_after_read(
                        node_idx=node_idx,
                        node_pos_norm=next_pos_norm,
                        node_pos_px=next_pos_px,
                        new_h_content=foveated_out.new_h_content,
                        digit_token=str(foveated_out.digit_logits.argmax().item())
                    )
            
            step_count += 1
        
        # Terminate if max_steps reached
        if not state.terminated:
            state.terminate()
        
        return DualModeOutput(
            total_loss=None,
            digit_loss=None,
            action_loss=None,
            jump_loss=None,
            predicted_sequence=state.get_output_sequence(),
            num_steps=state.step,
            num_digits=state.total_digits_read,
            num_chunks=sum(1 for t in state.output_tokens if t['token'] == '<CHUNK>'),
            state=state
        )


if __name__ == "__main__":
    print("=" * 60)
    print("  DualModeController — Structural Verification")
    print("=" * 60)
    
    # Verify module instantiation
    controller = DualModeController(
        vis_dim=512,
        hidden_dim=256,
        edge_dim=256,
        key_dim=256,
        num_classes=10,
        radius=80.0,
        T_intra=76.0,
        T_inter=108.0
    )
    
    total_params = sum(p.numel() for p in controller.parameters())
    trainable_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Verify sub-module structure
    print(f"\n  Sub-modules:")
    for name, module in controller.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name}: {params:,} params")
    
    # Verify GT parsing
    gt_seq = [
        {'token': '3', 'node_id': 0, 'mode': 'READ'},
        {'token': '8', 'node_id': 1, 'mode': 'READ'},
        {'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK'},
        {'token': '1', 'node_id': 2, 'mode': 'READ'},
        {'token': '2', 'node_id': 3, 'mode': 'READ'},
        {'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK'},
        {'token': '5', 'node_id': 4, 'mode': 'READ'},
    ]
    
    steps = controller.parse_gt_sequence(gt_seq)
    print(f"\n  GT Sequence Parsing:")
    print(f"    Input tokens:  {[t['token'] for t in gt_seq]}")
    print(f"    Parsed steps:  {len(steps)}")
    for i, s in enumerate(steps):
        print(f"      Step {i}: mode={s.mode}, node={s.node_id}, "
              f"digit={s.digit_label}, action={s.action}, target={s.action_target_node}")
    
    # Verify expected parsing
    assert steps[0].mode == 'JUMP' and steps[0].node_id == 0
    assert steps[1].mode == 'READ' and steps[1].action == 'CHUNK'
    assert steps[2].mode == 'JUMP' and steps[2].node_id == 2
    assert steps[3].mode == 'READ' and steps[3].action == 'CHUNK'
    assert steps[4].mode == 'JUMP' and steps[4].node_id == 4
    assert steps[4].action == 'CHUNK'  # Last digit
    
    print(f"\n  ✓ GT parsing verified")
    
    print("\n" + "=" * 60)
    print("  All structural checks passed.")
    print("=" * 60)