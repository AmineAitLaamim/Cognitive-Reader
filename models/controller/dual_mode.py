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

        self.loss_weights = loss_weights or {
            'digit': 1.0,
            'action': 1.0,
            'jump': 1.0
        }

        # ============================================================
        # Sub-modules
        # ============================================================

        self.spatial_encoder = SpatialFeatureEncoder(
            num_frequencies=num_frequencies,
            output_dim=key_dim,
            use_learnable_projection=True
        )

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

        self.saccadic = SaccadicJumpModule(
            key_dim=key_dim,
            vis_dim=vis_dim,
            start_embed_dim=128,
            num_heads=num_heads,
            dropout=dropout
        )

        self.graph_builder = ThresholdRadiusGraphBuilder(
            radius=radius,
            img_width=640,
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
        """
        with torch.no_grad():
            node_keys = self.spatial_encoder.encode_positions(
                graph.node_positions_norm
            )
            edge_embeddings = self.spatial_encoder.encode_edges(
                graph.edge_features,
                radius=self.radius
            )
            edge_distances = graph.edge_features[:, :, 2]

        return {
            'node_keys': node_keys,
            'edge_embeddings': edge_embeddings,
            'edge_distances': edge_distances
        }

    # ==============================================================
    # HELPER: Run foveated on a node and return output
    # ==============================================================

    def _run_foveated_at_node(
        self,
        graph: SpatialGraph,
        node_idx: int,
        state: ControllerState,
        edge_embeddings: torch.Tensor,
        edge_distances: torch.Tensor,
        device: torch.device
    ) -> Tuple[FoveatedOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the foveated module at a given node.

        Returns:
            foveated_out, cand_edge_emb, cand_mask, cand_indices
        """
        e_vis = graph.node_embeddings[node_idx].to(device)

        neighbors = self.graph_builder.get_unvisited_neighbors(
            graph, node_idx, state.visited_mask
        )

        if len(neighbors) > 0:
            cand_edge_emb = edge_embeddings[node_idx][neighbors]
            cand_mask = torch.ones(len(neighbors), device=device)
            cand_dist = edge_distances[node_idx][neighbors]
            cand_indices = neighbors.to(device)
        else:
            cand_edge_emb = torch.zeros(0, edge_embeddings.shape[-1], device=device)
            cand_mask = torch.zeros(0, device=device)
            cand_dist = torch.zeros(0, device=device)
            cand_indices = torch.zeros(0, dtype=torch.long, device=device)

        foveated_out = self.foveated(
            h_content=state.h_content,
            e_vis_current=e_vis,
            edge_embeddings=cand_edge_emb,
            candidate_mask=cand_mask,
            edge_distances=cand_dist,
            candidate_indices=cand_indices
        )

        return foveated_out, cand_edge_emb, cand_mask, cand_indices

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

        digit_tokens = [t for t in gt_sequence if t['mode'] == 'READ']

        chunk_boundaries = set()
        digit_pos = 0
        for token in gt_sequence:
            if token['mode'] == 'CHUNK':
                if digit_pos > 0:
                    chunk_boundaries.add(digit_pos)
            elif token['mode'] == 'READ':
                digit_pos += 1

        for i, dt in enumerate(digit_tokens):
            node_id = dt['node_id']
            digit_label = int(dt['token'])

            is_jump = (i == 0) or (i in chunk_boundaries)

            if i + 1 in chunk_boundaries or i == len(digit_tokens) - 1:
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

        gt_steps = self.parse_gt_sequence(gt_sequence)

        features = self._prepare_graph_features(graph)
        node_keys = features['node_keys'].to(device)
        edge_embeddings = features['edge_embeddings'].to(device)
        edge_distances = features['edge_distances'].to(device)

        state = ControllerState.initialize(
            num_nodes=N, hidden_dim=self.hidden_dim, device=device
        )

        total_digit_loss = torch.tensor(0.0, device=device)
        total_action_loss = torch.tensor(0.0, device=device)
        total_jump_loss = torch.tensor(0.0, device=device)
        num_digit_steps = 0
        num_action_steps = 0
        num_jump_steps = 0

        for step_idx, gt_step in enumerate(gt_steps):

            if gt_step.mode == 'JUMP':
                # ============================================
                # MODE 2: Saccadic Jump
                # ============================================

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

                saccadic_out = self.saccadic(
                    query=query,
                    node_keys=node_keys,
                    visited_mask=state.visited_mask,
                    greedy=True
                )

                jump_loss = self.saccadic.compute_loss(saccadic_out, gt_step.node_id)
                total_jump_loss = total_jump_loss + jump_loss
                num_jump_steps += 1

                # Jump to the ground-truth node
                node_pos_norm = graph.node_positions_norm[gt_step.node_id].to(device)
                node_pos_px = graph.node_positions_px[gt_step.node_id].to(device)
                state.update_after_jump(
                    node_idx=gt_step.node_id,
                    node_pos_norm=node_pos_norm,
                    node_pos_px=node_pos_px
                )

                # MANDATORY READ: classify digit at the jumped-to node
                foveated_out, cand_edge_emb, cand_mask, cand_indices = \
                    self._run_foveated_at_node(
                        graph, gt_step.node_id, state,
                        edge_embeddings, edge_distances, device
                    )

                digit_loss = F.cross_entropy(
                    foveated_out.digit_logits.unsqueeze(0),
                    torch.tensor([gt_step.digit_label], device=device)
                )
                total_digit_loss = total_digit_loss + digit_loss
                num_digit_steps += 1

                state.h_content = foveated_out.new_h_content

                # Determine ground-truth action index
                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
                    target_local_idx = (cand_indices == gt_step.action_target_node).nonzero(as_tuple=True)
                    if len(target_local_idx[0]) > 0:
                        gt_action_idx = target_local_idx[0][0].item()
                    else:
                        gt_action_idx = len(cand_indices)
                else:
                    gt_action_idx = len(cand_indices)  # <CHUNK>

                action_loss = F.cross_entropy(
                    foveated_out.action_logits.unsqueeze(0),
                    torch.tensor([gt_action_idx], device=device)
                )
                total_action_loss = total_action_loss + action_loss
                num_action_steps += 1

                if gt_step.action == 'CHUNK':
                    state.update_after_chunk()

            elif gt_step.mode == 'READ':
                # ============================================
                # MODE 1: Foveated Read (continuation)
                # ============================================

                current_node = state.current_node

                foveated_out, cand_edge_emb, cand_mask, cand_indices = \
                    self._run_foveated_at_node(
                        graph, gt_step.node_id, state,
                        edge_embeddings, edge_distances, device
                    )

                digit_loss = F.cross_entropy(
                    foveated_out.digit_logits.unsqueeze(0),
                    torch.tensor([gt_step.digit_label], device=device)
                )
                total_digit_loss = total_digit_loss + digit_loss
                num_digit_steps += 1

                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
                    target_local_idx = (cand_indices == gt_step.action_target_node).nonzero(as_tuple=True)
                    if len(target_local_idx[0]) > 0:
                        gt_action_idx = target_local_idx[0][0].item()
                    else:
                        gt_action_idx = len(cand_indices)
                else:
                    gt_action_idx = len(cand_indices)

                action_loss = F.cross_entropy(
                    foveated_out.action_logits.unsqueeze(0),
                    torch.tensor([gt_action_idx], device=device)
                )
                total_action_loss = total_action_loss + action_loss
                num_action_steps += 1

                if gt_step.action == 'READ' and gt_step.action_target_node is not None:
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
            num_chunks=sum(1 for t in state.output_tokens if t.get('token') == '<CHUNK>'),
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

        Architecture:
          Mode 2 (Saccadic Jump):
            1. Select target node via global attention.
            2. Jump to target (mark visited, update state).
            3. MANDATORY READ: classify digit at target, record it.
            4. Hand off to Mode 1 for action selection.

          Mode 1 (Foveated Read):
            1. Run foveated on current node -> action logits.
            2. Select action (visit neighbor or CHUNK).
            3. If READ: classify neighbor digit, let update_after_read record it.
            4. If CHUNK: let update_after_chunk record <CHUNK>, switch to Mode 2.

        Token recording rules:
          - Mode 2: explicitly appends digit (update_after_jump does NOT record).
          - Mode 1 READ: passes digit to update_after_read (which records it).
          - Mode 1 CHUNK: update_after_chunk records <CHUNK>.
          - No duplicate appends anywhere.
        """
        N = graph.num_nodes

        features = self._prepare_graph_features(graph)
        node_keys = features['node_keys'].to(device)
        edge_embeddings = features['edge_embeddings'].to(device)
        edge_distances = features['edge_distances'].to(device)

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

                # Forward pass: select target node
                saccadic_out = self.saccadic(
                    query=query,
                    node_keys=node_keys,
                    visited_mask=state.visited_mask,
                    greedy=greedy
                )

                if saccadic_out.selected_node_idx < 0:
                    state.terminate()
                    break

                selected = saccadic_out.selected_node_idx

                # Jump to selected node
                node_pos_norm = graph.node_positions_norm[selected].to(device)
                node_pos_px = graph.node_positions_px[selected].to(device)
                state.update_after_jump(selected, node_pos_norm, node_pos_px)

                # MANDATORY READ: classify digit at the jumped-to node
                foveated_out, _, _, _ = self._run_foveated_at_node(
                    graph, selected, state,
                    edge_embeddings, edge_distances, device
                )

                pred_digit = foveated_out.digit_logits.argmax().item()
                state.h_content = foveated_out.new_h_content

                # Record the digit (update_after_jump does NOT record digits)
                state.output_tokens.append({
                    'token': str(pred_digit),
                    'node_id': selected,
                    'mode': 'READ',
                    'step': state.step,
                    'chunk_size': state.chunk_size
                })
                state.total_digits_read += 1

                # Hand off to Mode 1 for action selection
                state.mode = ControllerMode.FOVEATED_READ

            elif state.mode == ControllerMode.FOVEATED_READ:
                # ============================================
                # MODE 1: Foveated Read — Action Selection
                # ============================================
                # The digit at current_node was already classified and
                # recorded (by Mode 2 or by the previous Mode 1 READ).
                # Here we only decide the next action.

                current = state.current_node

                # Run foveated for action logits
                foveated_out, cand_edge_emb, cand_mask, cand_indices = \
                    self._run_foveated_at_node(
                        graph, current, state,
                        edge_embeddings, edge_distances, device
                    )

                # Select action
                action_type, node_idx, local_idx = self.foveated.select_action(
                    foveated_out, greedy=greedy, temperature=temperature
                )

                if action_type == 'CHUNK':
                    # update_after_chunk records <CHUNK> — do NOT append separately
                    state.update_after_chunk()

                elif action_type == 'READ' and node_idx is not None:
                    # Classify neighbor's digit BEFORE moving
                    foveated_next, _, _, _ = self._run_foveated_at_node(
                        graph, node_idx, state,
                        edge_embeddings, edge_distances, device
                    )
                    pred_digit = foveated_next.digit_logits.argmax().item()

                    next_pos_norm = graph.node_positions_norm[node_idx].to(device)
                    next_pos_px = graph.node_positions_px[node_idx].to(device)

                    # update_after_read records the digit — do NOT append separately
                    state.update_after_read(
                        node_idx=node_idx,
                        node_pos_norm=next_pos_norm,
                        node_pos_px=next_pos_px,
                        new_h_content=foveated_next.new_h_content,
                        digit_token=str(pred_digit)
                    )

                    # Stay in Mode 1 for next action selection

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
            num_chunks=sum(1 for t in state.output_tokens if t.get('token') == '<CHUNK>'),
            state=state
        )


if __name__ == "__main__":
    print("=" * 60)
    print("  DualModeController — Structural Verification")
    print("=" * 60)

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

    assert steps[0].mode == 'JUMP' and steps[0].node_id == 0
    assert steps[1].mode == 'READ' and steps[1].action == 'CHUNK'
    assert steps[2].mode == 'JUMP' and steps[2].node_id == 2
    assert steps[3].mode == 'READ' and steps[3].action == 'CHUNK'
    assert steps[4].mode == 'JUMP' and steps[4].node_id == 4
    assert steps[4].action == 'CHUNK'

    print(f"\n  ✓ GT parsing verified")

    print("\n" + "=" * 60)
    print("  All structural checks passed.")
    print("=" * 60)