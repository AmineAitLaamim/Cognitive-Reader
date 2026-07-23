"""
models/controller/state.py
Mutable state management for the Dual-Mode Cognitive Controller.

This module defines the ControllerState dataclass, which holds ALL mutable
state that Mode 1 (Foveated Read) and Mode 2 (Saccadic Jump) operate on.

STATE OWNERSHIP RULES:
  - The controller (dual_mode.py) owns and mutates the state.
  - The graph (builder.py) is READ-ONLY. The state never modifies the graph.
  - The backbone (cnn.py) is READ-ONLY. The state never modifies embeddings.
  - h_content is the ONLY differentiable tensor in the state.
    All other tensors are non-differentiable bookkeeping.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class ControllerMode(IntEnum):
    """Operating mode of the Dual-Mode Controller."""
    FOVEATED_READ = 1    # Mode 1: Local traversal within threshold radius
    SACCADIC_JUMP = 2    # Mode 2: Global search for next chunk start


@dataclass
class ControllerState:
    """
    Complete mutable state of the Dual-Mode Controller.
    
    Lifecycle:
      1. Created via ControllerState.initialize(graph_size, hidden_dim, device)
      2. Mutated by the controller at each time step.
      3. Terminated when all nodes are visited or max_steps is reached.
    
    Differentiability:
      - h_content: DIFFERENTIABLE. Participates in backpropagation.
      - All other tensors: NON-DIFFERENTIABLE. Detached bookkeeping.
    """
    
    # === Content Memory (Differentiable) ===
    h_content: torch.Tensor              # [hidden_dim] — sequential memory of current chunk
    
    # === Spatial State (Non-differentiable) ===
    spatial_anchor_norm: torch.Tensor    # [2] — (x_last/W, y_last/H) normalized
    spatial_anchor_px: torch.Tensor      # [2] — (x_last, y_last) raw pixels
    chunk_start_norm: torch.Tensor       # [2] — (x_first/W, y_first/H) of current chunk
    chunk_start_px: torch.Tensor         # [2] — (x_first, y_first) of current chunk
    chunk_trajectory: torch.Tensor       # [2] — (x_last - x_first, y_last - y_first) normalized
    
    # === Graph State (Non-differentiable) ===
    visited_mask: torch.Tensor           # [N] — 1.0 = visited, 0.0 = unvisited
    current_node: int                    # Index of current node. -1 if none.
    num_nodes: int                       # Total nodes in the graph
    
    # === Mode ===
    mode: ControllerMode                 # Current operating mode
    
    # === Sequence Tracking ===
    step: int                            # Current time step (0-indexed)
    chunk_size: int                      # Digits read in current chunk
    total_digits_read: int               # Total digits read across all chunks
    output_tokens: List[Dict]            # Emitted tokens: [{token, node_id, mode, step}]
    
    # === Flags ===
    initialized: bool                    # False before first node is selected
    terminated: bool                     # True when reading is complete
    
    # === Device ===
    device: torch.device = field(default=torch.device('cpu'), repr=False)
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    
    @classmethod
    def initialize(
        cls,
        num_nodes: int,
        hidden_dim: int,
        device: torch.device = torch.device('cpu')
    ) -> 'ControllerState':
        """
        Create a fresh state for a new image.
        
        At t=0, before any digit is read:
          - h_content = 0 (empty memory)
          - spatial_anchor = (0, 0) (undefined, will be set after first jump)
          - visited_mask = all zeros (nothing visited)
          - mode = SACCADIC_JUMP (must find the first node globally)
          - initialized = False
        
        Args:
            num_nodes: Total number of nodes in the graph.
            hidden_dim: Dimensionality of h_content.
            device: Torch device.
        
        Returns:
            Fresh ControllerState.
        """
        return cls(
            h_content=torch.zeros(hidden_dim, device=device),
            spatial_anchor_norm=torch.zeros(2, device=device),
            spatial_anchor_px=torch.zeros(2, device=device),
            chunk_start_norm=torch.zeros(2, device=device),
            chunk_start_px=torch.zeros(2, device=device),
            chunk_trajectory=torch.zeros(2, device=device),
            visited_mask=torch.zeros(num_nodes, device=device),
            current_node=-1,
            num_nodes=num_nodes,
            mode=ControllerMode.SACCADIC_JUMP,
            step=0,
            chunk_size=0,
            total_digits_read=0,
            output_tokens=[],
            initialized=False,
            terminated=False,
            device=device
        )
    
    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------
    
    def update_after_jump(
        self,
        node_idx: int,
        node_pos_norm: torch.Tensor,
        node_pos_px: torch.Tensor
    ) -> None:
        """
        Update state after Mode 2 (Saccadic Jump) selects a new starting node.
        
        This is called:
          - At t=0, when the initial global search finds the first digit.
          - After a <CHUNK>, when the saccadic jump lands on the next chunk start.
        
        Effects:
          - Sets the spatial anchor and chunk start to the new node's position.
          - Resets the chunk trajectory to (0, 0).
          - Marks the new node as visited.
          - Switches mode to FOVEATED_READ.
          - Sets initialized = True.
        
        Args:
            node_idx: Index of the selected node in the graph.
            node_pos_norm: [2] normalized position (x/W, y/H).
            node_pos_px: [2] raw pixel position (x, y).
        """
        self.current_node = node_idx
        self.spatial_anchor_norm = node_pos_norm.detach().clone()
        self.spatial_anchor_px = node_pos_px.detach().clone()
        self.chunk_start_norm = node_pos_norm.detach().clone()
        self.chunk_start_px = node_pos_px.detach().clone()
        self.chunk_trajectory = torch.zeros(2, device=self.device)
        self.visited_mask[node_idx] = 1.0
        self.mode = ControllerMode.FOVEATED_READ
        self.chunk_size = 1
        self.total_digits_read += 1
        self.initialized = True
    
    def update_after_read(
        self,
        node_idx: int,
        node_pos_norm: torch.Tensor,
        node_pos_px: torch.Tensor,
        new_h_content: torch.Tensor,
        digit_token: str
    ) -> None:
        """
        Update state after Mode 1 (Foveated Read) reads a digit and moves
        to the next node within the same chunk.
        
        Effects:
          - Updates h_content with the new recurrent state.
          - Updates the spatial anchor to the new node's position.
          - Recomputes the chunk trajectory: (x_last - x_first, y_last - y_first).
          - Marks the new node as visited.
          - Increments chunk_size and total_digits_read.
          - Appends the digit token to output_tokens.
        
        Args:
            node_idx: Index of the node just read.
            node_pos_norm: [2] normalized position.
            node_pos_px: [2] raw pixel position.
            new_h_content: [hidden_dim] updated content hidden state (DIFFERENTIABLE).
            digit_token: The predicted digit character ('0'-'9').
        """
        # Update content memory (differentiable — do NOT detach)
        self.h_content = new_h_content
        
        # Update spatial state (non-differentiable — detach)
        self.spatial_anchor_norm = node_pos_norm.detach().clone()
        self.spatial_anchor_px = node_pos_px.detach().clone()
        
        # Recompute trajectory: from chunk start to current position
        self.chunk_trajectory = (
            self.spatial_anchor_norm - self.chunk_start_norm
        ).detach()
        
        # Mark visited
        self.visited_mask[node_idx] = 1.0
        self.current_node = node_idx
        
        # Update counters
        self.chunk_size += 1
        self.total_digits_read += 1
        self.step += 1
        
        # Record output
        self.output_tokens.append({
            'token': digit_token,
            'node_id': node_idx,
            'mode': 'READ',
            'step': self.step,
            'chunk_size': self.chunk_size
        })
    
    def update_after_chunk(self) -> None:
        """
        Update state after emitting a <CHUNK> token.
        
        Effects:
          - Resets h_content to zero (clears working memory).
          - Resets chunk_size to 0.
          - Resets chunk_trajectory to (0, 0).
          - Switches mode to SACCADIC_JUMP (must find next chunk start).
          - Does NOT reset the spatial anchor (needed for Mode 2 query).
          - Does NOT reset the visited mask (must remember what was read).
          - Appends <CHUNK> to output_tokens.
        
        CRITICAL: The spatial anchor is PRESERVED across chunk boundaries.
        Mode 2 needs it to construct the query Q = f(anchor, trajectory).
        """
        # Reset content memory (differentiable zero)
        self.h_content = torch.zeros_like(self.h_content)
        
        # Reset chunk-level state
        self.chunk_size = 0
        self.chunk_trajectory = torch.zeros(2, device=self.device)
        
        # Switch to global search mode
        self.mode = ControllerMode.SACCADIC_JUMP
        self.step += 1
        
        # Record output
        self.output_tokens.append({
            'token': '<CHUNK>',
            'node_id': None,
            'mode': 'CHUNK',
            'step': self.step,
            'chunk_size': 0
        })
    
    def update_after_local_chunk(
        self,
        next_node_idx: int,
        next_node_pos_norm: torch.Tensor,
        next_node_pos_px: torch.Tensor
    ) -> None:
        """
        Update state after emitting a <CHUNK> token due to a distance-based
        boundary (d > T_inter) while a local neighbor still exists.
        
        This is the "local chunk crossing" case: Mode 1 found the next node
        but it's across a chunk boundary. We emit <CHUNK>, reset h_content,
        and STAY IN MODE 1 to process the next node as the start of a new chunk.
        
        Effects:
          - Resets h_content to zero.
          - Sets chunk start to the next node's position.
          - Resets trajectory to (0, 0).
          - Marks the next node as visited.
          - Stays in FOVEATED_READ mode.
        
        Args:
            next_node_idx: Index of the node that triggered the boundary.
            next_node_pos_norm: [2] normalized position.
            next_node_pos_px: [2] raw pixel position.
        """
        # Reset content memory
        self.h_content = torch.zeros_like(self.h_content)
        
        # Set new chunk start
        self.chunk_start_norm = next_node_pos_norm.detach().clone()
        self.chunk_start_px = next_node_pos_px.detach().clone()
        self.spatial_anchor_norm = next_node_pos_norm.detach().clone()
        self.spatial_anchor_px = next_node_pos_px.detach().clone()
        self.chunk_trajectory = torch.zeros(2, device=self.device)
        
        # Mark visited and update counters
        self.visited_mask[next_node_idx] = 1.0
        self.current_node = next_node_idx
        self.chunk_size = 1
        self.total_digits_read += 1
        self.step += 1
        
        # Stay in Mode 1
        self.mode = ControllerMode.FOVEATED_READ
        
        # Record output
        self.output_tokens.append({
            'token': '<CHUNK>',
            'node_id': None,
            'mode': 'CHUNK',
            'step': self.step - 1,
            'chunk_size': 0
        })
    
    def terminate(self) -> None:
        """
        Mark the reading sequence as complete.
        Called when all nodes are visited or max_steps is reached.
        """
        self.terminated = True
        self.output_tokens.append({
            'token': '<END>',
            'node_id': None,
            'mode': 'END',
            'step': self.step,
            'chunk_size': 0
        })
    
    # ------------------------------------------------------------------
    # Query Methods (Read-only)
    # ------------------------------------------------------------------
    
    def get_unvisited_count(self) -> int:
        """Return the number of unvisited nodes."""
        return int((self.visited_mask == 0).sum().item())
    
    def all_visited(self) -> bool:
        """Check if all nodes have been visited."""
        return bool((self.visited_mask == 1).all().item())
    
    def is_chunk_start(self) -> bool:
        """Check if the current position is the start of a new chunk."""
        return self.chunk_size <= 1
    
    def get_trajectory_for_query(self) -> torch.Tensor:
        """
        Get the chunk trajectory vector for Mode 2 query construction.
        
        Returns:
            [2] tensor: (x_last - x_first, y_last - y_first) normalized.
            Returns (0, 0) if the chunk has only one digit (trajectory undefined).
        """
        return self.chunk_trajectory.clone()
    
    def get_anchor_for_query(self) -> torch.Tensor:
        """
        Get the spatial anchor for Mode 2 query construction.
        
        Returns:
            [2] tensor: (x_last/W, y_last/H) normalized.
        """
        return self.spatial_anchor_norm.clone()
    
    def get_output_sequence(self) -> List[str]:
        """
        Extract the predicted token sequence (for evaluation).
        
        Returns:
            List of token strings: ['3', '8', '<CHUNK>', '1', '2', '<END>']
        """
        return [t['token'] for t in self.output_tokens]
    
    def get_output_string(self) -> str:
        """
        Extract the predicted sequence as a single string.
        
        Returns:
            e.g., "38<CHUNK>12<END>"
        """
        return ''.join(self.get_output_sequence())
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    
    def to(self, device: torch.device) -> 'ControllerState':
        """Move all tensors to the specified device."""
        self.h_content = self.h_content.to(device)
        self.spatial_anchor_norm = self.spatial_anchor_norm.to(device)
        self.spatial_anchor_px = self.spatial_anchor_px.to(device)
        self.chunk_start_norm = self.chunk_start_norm.to(device)
        self.chunk_start_px = self.chunk_start_px.to(device)
        self.chunk_trajectory = self.chunk_trajectory.to(device)
        self.visited_mask = self.visited_mask.to(device)
        self.device = device
        return self
    
    def summary(self) -> str:
        """Return a human-readable summary of the current state."""
        lines = [
            f"ControllerState Summary:",
            f"  Step:              {self.step}",
            f"  Mode:              {self.mode.name}",
            f"  Current node:      {self.current_node}",
            f"  Chunk size:        {self.chunk_size}",
            f"  Total digits read: {self.total_digits_read}",
            f"  Unvisited nodes:   {self.get_unvisited_count()}/{self.num_nodes}",
            f"  Initialized:       {self.initialized}",
            f"  Terminated:        {self.terminated}",
            f"  Spatial anchor:    ({self.spatial_anchor_norm[0]:.3f}, {self.spatial_anchor_norm[1]:.3f})",
            f"  Chunk trajectory:  ({self.chunk_trajectory[0]:.3f}, {self.chunk_trajectory[1]:.3f})",
            f"  Output so far:     {self.get_output_string()[:50]}...",
        ]
        return '\n'.join(lines)


if __name__ == "__main__":
    # --- Unit test: simulate a reading sequence ---
    
    print("=" * 60)
    print("  ControllerState Unit Test")
    print("=" * 60)
    
    device = torch.device('cpu')
    N = 6       # 6 nodes in the graph
    D = 128     # hidden dim
    
    # Initialize
    state = ControllerState.initialize(num_nodes=N, hidden_dim=D, device=device)
    print(f"\n[Step 0] Initial state:")
    print(f"  Mode: {state.mode.name}")
    print(f"  Initialized: {state.initialized}")
    print(f"  Unvisited: {state.get_unvisited_count()}")
    
    # Simulate Mode 2 jump to node 0 (first digit)
    pos0_norm = torch.tensor([0.15, 0.12])
    pos0_px = torch.tensor([96.0, 76.8])
    state.update_after_jump(node_idx=0, node_pos_norm=pos0_norm, node_pos_px=pos0_px)
    print(f"\n[Step 1] After jump to node 0:")
    print(f"  Mode: {state.mode.name}")
    print(f"  Current node: {state.current_node}")
    print(f"  Chunk size: {state.chunk_size}")
    print(f"  Anchor: ({state.spatial_anchor_norm[0]:.3f}, {state.spatial_anchor_norm[1]:.3f})")
    
    # Simulate Mode 1 reads: node 1, node 2
    for i, (pos_norm, pos_px, digit) in enumerate([
        (torch.tensor([0.22, 0.13]), torch.tensor([140.8, 83.2]), '3'),
        (torch.tensor([0.29, 0.11]), torch.tensor([185.6, 70.4]), '8'),
    ]):
        new_h = torch.randn(D)  # Simulated recurrent update
        state.update_after_read(
            node_idx=i + 1,
            node_pos_norm=pos_norm,
            node_pos_px=pos_px,
            new_h_content=new_h,
            digit_token=digit
        )
        print(f"\n[Step {i+2}] After reading node {i+1} ('{digit}'):")
        print(f"  Chunk size: {state.chunk_size}")
        print(f"  Trajectory: ({state.chunk_trajectory[0]:.3f}, {state.chunk_trajectory[1]:.3f})")
        print(f"  Output: {state.get_output_string()}")
    
    # Simulate distance-based chunk boundary: next node (3) is far away
    pos3_norm = torch.tensor([0.55, 0.40])
    pos3_px = torch.tensor([352.0, 256.0])
    state.update_after_local_chunk(
        next_node_idx=3,
        next_node_pos_norm=pos3_norm,
        next_node_pos_px=pos3_px
    )
    print(f"\n[Step 4] After local chunk crossing to node 3:")
    print(f"  Mode: {state.mode.name}")
    print(f"  Chunk size: {state.chunk_size}")
    print(f"  h_content norm: {state.h_content.norm():.4f} (should be 0)")
    print(f"  Output: {state.get_output_string()}")
    
    # Simulate Mode 1 reads: node 4, node 5
    for i, (pos_norm, pos_px, digit) in enumerate([
        (torch.tensor([0.62, 0.41]), torch.tensor([396.8, 262.4]), '1'),
        (torch.tensor([0.69, 0.39]), torch.tensor([441.6, 249.6]), '2'),
    ]):
        new_h = torch.randn(D)
        state.update_after_read(
            node_idx=i + 4,
            node_pos_norm=pos_norm,
            node_pos_px=pos_px,
            new_h_content=new_h,
            digit_token=digit
        )
    
    # Terminate
    state.terminate()
    
    print(f"\n[Final] Complete sequence:")
    print(f"  Output: {state.get_output_string()}")
    print(f"  Total steps: {state.step}")
    print(f"  All visited: {state.all_visited()}")
    
    # Verify
    expected = "38<CHUNK>12<END>"
    actual = state.get_output_string()
    assert actual == expected, f"Expected '{expected}', got '{actual}'"
    print(f"\n  ✓ Sequence matches expected: '{expected}'")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)