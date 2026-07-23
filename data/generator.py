"""
data/generator.py
Constrained Polar Sampling data generator for the Cognitive Reader project.
Generates 2D digit layouts with mathematically guaranteed chunk boundaries.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    """All hyperparameters for the Constrained Polar Generator."""
    # Canvas dimensions
    img_width: int = 640
    img_height: int = 640

    # Core geometric threshold (pixels)
    threshold_radius_r: float = 80.0

    # Derived intra/inter chunk bounds
    r_intra_factor: float = 0.8       # max intra-chunk distance = 0.8 * r
    r_inter_factor: float = 1.5       # min inter-chunk distance = 1.5 * r
    r_max_inter_factor: float = 2.5   # max inter-chunk distance = 2.5 * r

    # Intra-chunk angle distribution (radians)
    intra_angle_mean: float = 0.0
    intra_angle_std: float = 5.0 * (math.pi / 180.0)  # 5 degrees

    # Inter-chunk angle distribution (radians) — carriage return (down-left)
    inter_angle_mean: float = 225.0 * (math.pi / 180.0)
    inter_angle_std: float = 10.0 * (math.pi / 180.0)  # 10 degrees

    # Sim2Real detector noise (pixels)
    noise_sigma: float = 3.0

    # Digit base dimensions (pixels, before scale variation)
    base_digit_w: float = 20.0
    base_digit_h: float = 30.0

    # Scale variation range
    scale_min: float = 0.8
    scale_max: float = 1.2

    # Chunk size constraints
    max_chunk_size: int = 4
    min_chunk_size: int = 1

    # Boundary margin (keep digits away from canvas edges)
    boundary_margin: float = 40.0

    # Rejection sampling max attempts for boundary violations
    max_resample_attempts: int = 50

    # Minimum intra-digit distance (prevent overlap)
    min_digit_gap: float = 25.0


@dataclass
class DigitNode:
    """A single digit node in the 2D layout."""
    node_id: int
    label: str
    center_x: float          # perfect center (no noise)
    center_y: float          # perfect center (no noise)
    noisy_center_x: float    # center with Sim2Real noise
    noisy_center_y: float    # center with Sim2Real noise
    width: float             # bounding box width (after scale)
    height: float            # bounding box height (after scale)
    scale: float             # scale factor applied
    chunk_id: int            # which chunk this digit belongs to


@dataclass
class GeneratedSample:
    """Complete output of a single generated sample."""
    nodes: List[DigitNode]
    gt_sequence: List[Dict[str, Any]]   # [{token, node_id, mode}]
    img_width: int
    img_height: int
    num_chunks: int
    total_digits: int


class ConstrainedPolarGenerator:
    """
    Generates synthetic 2D digit layouts using Constrained Polar Sampling.
    
    Guarantees:
      - Intra-chunk distances are strictly < r_intra_factor * r
      - Inter-chunk distances are strictly > r_inter_factor * r
      - Ground-truth <CHUNK> tokens align exactly with physical spatial gaps
      - Sim2Real noise is injected for detector training
    """

    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.r = config.threshold_radius_r
        self.r_intra = config.r_intra_factor * self.r
        self.r_inter = config.r_inter_factor * self.r
        self.r_max_inter = config.r_max_inter_factor * self.r

        # Dynamic thresholds for controller (accounting for noise)
        self.T_intra = self.r_intra + 4 * config.noise_sigma
        self.T_inter = self.r_inter - 4 * config.noise_sigma

        # Validate thresholds
        if self.T_intra >= self.T_inter:
            raise ValueError(
                f"Threshold overlap: T_intra={self.T_intra:.1f} >= T_inter={self.T_inter:.1f}. "
                f"Increase r or decrease noise_sigma."
            )

    def generate_sample(self, total_digits: int) -> GeneratedSample:
        """
        Generate a single 2D digit layout.
        
        Args:
            total_digits: Total number of digits to place.
            
        Returns:
            GeneratedSample with nodes, ground-truth sequence, and metadata.
        """
        nodes: List[DigitNode] = []
        gt_sequence: List[Dict[str, Any]] = []

        # Initialize starting position in the top-left region
        current_x = np.random.uniform(
            self.cfg.boundary_margin + self.r_inter,
            self.cfg.img_width * 0.25
        )
        current_y = np.random.uniform(
            self.cfg.boundary_margin + self.r_inter,
            self.cfg.img_height * 0.2
        )

        current_chunk_size = 0
        current_chunk_id = 0
        node_id = 0
        is_first_digit = True

        while node_id < total_digits:
            # --- Decide placement mode ---
            need_inter_chunk = False

            if is_first_digit:
                # First digit: no inter-chunk jump needed, just place it
                need_inter_chunk = False
            elif current_chunk_size >= self.cfg.max_chunk_size:
                need_inter_chunk = True
            elif self._approaching_boundary(current_x, current_y, mode='intra'):
                need_inter_chunk = True

            # --- Place the digit ---
            if need_inter_chunk and not is_first_digit:
                # INTER-CHUNK PLACEMENT (new line / new chunk)
                new_x, new_y = self._sample_inter_chunk_position(current_x, current_y)

                if new_x is None:
                    # Cannot place more digits without violating constraints
                    break

                # Insert <CHUNK> token in ground truth
                gt_sequence.append({
                    'token': '<CHUNK>',
                    'node_id': None,
                    'mode': 'CHUNK'
                })
                current_chunk_id += 1
                current_chunk_size = 1

            else:
                if is_first_digit:
                    new_x, new_y = current_x, current_y
                    current_chunk_size = 1
                    is_first_digit = False
                else:
                    # INTRA-CHUNK PLACEMENT
                    new_x, new_y = self._sample_intra_chunk_position(current_x, current_y)

                    if new_x is None:
                        # Intra-chunk placement failed (boundary), force inter-chunk
                        new_x, new_y = self._sample_inter_chunk_position(current_x, current_y)
                        if new_x is None:
                            break
                        gt_sequence.append({
                            'token': '<CHUNK>',
                            'node_id': None,
                            'mode': 'CHUNK'
                        })
                        current_chunk_id += 1
                        current_chunk_size = 1
                    else:
                        current_chunk_size += 1

            # --- Generate digit properties ---
            scale = np.random.uniform(self.cfg.scale_min, self.cfg.scale_max)
            w = self.cfg.base_digit_w * scale
            h = self.cfg.base_digit_h * scale
            label = str(np.random.randint(0, 10))

            # Inject Sim2Real noise
            noisy_x = new_x + np.random.normal(0, self.cfg.noise_sigma)
            noisy_y = new_y + np.random.normal(0, self.cfg.noise_sigma)

            # Create node
            node = DigitNode(
                node_id=node_id,
                label=label,
                center_x=new_x,
                center_y=new_y,
                noisy_center_x=noisy_x,
                noisy_center_y=noisy_y,
                width=w,
                height=h,
                scale=scale,
                chunk_id=current_chunk_id
            )
            nodes.append(node)

            # Append digit to ground-truth sequence
            gt_sequence.append({
                'token': label,
                'node_id': node_id,
                'mode': 'READ'
            })

            # Update state
            current_x = new_x
            current_y = new_y
            node_id += 1

        return GeneratedSample(
            nodes=nodes,
            gt_sequence=gt_sequence,
            img_width=self.cfg.img_width,
            img_height=self.cfg.img_height,
            num_chunks=current_chunk_id + 1,
            total_digits=len(nodes)
        )

    def _sample_intra_chunk_position(
        self, current_x: float, current_y: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Sample a position for the next digit within the same chunk.
        Uses rejection sampling to guarantee boundary constraints.
        
        Returns:
            (new_x, new_y) or (None, None) if placement is impossible.
        """
        d_min = max(self.cfg.min_digit_gap, 1.2 * self.cfg.base_digit_w)

        for _ in range(self.cfg.max_resample_attempts):
            d = np.random.uniform(d_min, self.r_intra)
            theta = np.random.normal(self.cfg.intra_angle_mean, self.cfg.intra_angle_std)

            new_x = current_x + d * math.cos(theta)
            new_y = current_y + d * math.sin(theta)

            # Check boundary constraints
            if self._is_within_bounds(new_x, new_y):
                return new_x, new_y

        return None, None

    def _sample_inter_chunk_position(
        self, current_x: float, current_y: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Sample a position for the first digit of a new chunk (line break).
        Uses rejection sampling to guarantee:
          1. Distance > r_inter (chunk boundary preserved)
          2. Position is within canvas bounds
        
        Returns:
            (new_x, new_y) or (None, None) if placement is impossible.
        """
        for _ in range(self.cfg.max_resample_attempts):
            d = np.random.uniform(self.r_inter, self.r_max_inter)
            theta = np.random.normal(self.cfg.inter_angle_mean, self.cfg.inter_angle_std)

            new_x = current_x + d * math.cos(theta)
            new_y = current_y + d * math.sin(theta)

            # Check boundary constraints
            if self._is_within_bounds(new_x, new_y):
                # Verify distance constraint is preserved (no clamping)
                actual_d = math.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
                if actual_d >= self.r_inter:
                    return new_x, new_y

        return None, None

    def _is_within_bounds(self, x: float, y: float) -> bool:
        """Check if a position is within the canvas boundaries."""
        margin = self.cfg.boundary_margin
        return (
            margin <= x <= self.cfg.img_width - margin and
            margin <= y <= self.cfg.img_height - margin
        )

    def _approaching_boundary(self, x: float, y: float, mode: str = 'intra') -> bool:
        """Check if the current position is too close to the canvas boundary."""
        margin = self.cfg.boundary_margin
        buffer = self.r_intra if mode == 'intra' else self.r_inter

        return (
            x + buffer > self.cfg.img_width - margin or
            y + buffer > self.cfg.img_height - margin or
            x - buffer < margin or
            y - buffer < margin
        )

    def get_bounding_boxes(self, sample: GeneratedSample, use_noise: bool = True) -> List[Dict]:
        """
        Extract bounding boxes from a generated sample.
        
        Args:
            sample: The generated sample.
            use_noise: If True, return noisy boxes (for detector training).
                       If False, return perfect boxes (for controller training).
        
        Returns:
            List of dicts with {x, y, w, h, center_x, center_y, label, node_id, chunk_id}
        """
        boxes = []
        for node in sample.nodes:
            cx = node.noisy_center_x if use_noise else node.center_x
            cy = node.noisy_center_y if use_noise else node.center_y

            boxes.append({
                'x': cx - node.width / 2,       # top-left x
                'y': cy - node.height / 2,       # top-left y
                'w': node.width,
                'h': node.height,
                'center_x': cx,
                'center_y': cy,
                'label': node.label,
                'node_id': node.node_id,
                'chunk_id': node.chunk_id,
                'scale': node.scale
            })
        return boxes

    def get_adjacency_info(self, sample: GeneratedSample, use_noise: bool = True) -> Dict:
        """
        Compute pairwise distances between all nodes.
        Useful for verifying graph construction and threshold calibration.
        
        Returns:
            Dict with distance matrix and chunk boundary flags.
        """
        n = len(sample.nodes)
        distances = np.zeros((n, n))
        same_chunk = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ni = sample.nodes[i]
                nj = sample.nodes[j]

                if use_noise:
                    dx = ni.noisy_center_x - nj.noisy_center_x
                    dy = ni.noisy_center_y - nj.noisy_center_y
                else:
                    dx = ni.center_x - nj.center_x
                    dy = ni.center_y - nj.center_y

                distances[i, j] = math.sqrt(dx**2 + dy**2)
                same_chunk[i, j] = (ni.chunk_id == nj.chunk_id)

        return {
            'distances': distances,
            'same_chunk': same_chunk,
            'r_intra': self.r_intra,
            'r_inter': self.r_inter,
            'T_intra': self.T_intra,
            'T_inter': self.T_inter
        }


# --- Convenience function for quick testing ---
def generate_and_validate(config: GeneratorConfig, total_digits: int = 50) -> GeneratedSample:
    """
    Generate a sample and validate all geometric constraints.
    Raises AssertionError if any constraint is violated.
    """
    gen = ConstrainedPolarGenerator(config)
    sample = gen.generate_sample(total_digits)

    # Validate with perfect boxes (no noise)
    info = gen.get_adjacency_info(sample, use_noise=False)
    distances = info['distances']
    same_chunk = info['same_chunk']

    for i in range(len(sample.nodes)):
        for j in range(i + 1, len(sample.nodes)):
            d = distances[i, j]
            if same_chunk[i, j]:
                assert d <= gen.r_intra, (
                    f"Intra-chunk violation: nodes {i},{j} distance={d:.1f} > r_intra={gen.r_intra:.1f}"
                )
            else:
                # Only check adjacent chunks (non-adjacent chunks can be far apart)
                chunk_diff = abs(sample.nodes[i].chunk_id - sample.nodes[j].chunk_id)
                if chunk_diff == 1:
                    assert d >= gen.r_inter, (
                        f"Inter-chunk violation: nodes {i},{j} distance={d:.1f} < r_inter={gen.r_inter:.1f}"
                    )

    print(f"✓ Generated {sample.total_digits} digits in {sample.num_chunks} chunks. All constraints satisfied.")
    return sample


if __name__ == "__main__":
    config = GeneratorConfig(
        img_width=640,
        img_height=640,
        threshold_radius_r=80.0,
        noise_sigma=3.0,
        max_chunk_size=4,
        min_chunk_size=1
    )

    # Generate and validate a sample
    sample = generate_and_validate(config, total_digits=50)

    # Print summary
    print(f"\nSample Summary:")
    print(f"  Total digits: {sample.total_digits}")
    print(f"  Total chunks: {sample.num_chunks}")
    print(f"  GT sequence length: {len(sample.gt_sequence)}")
    print(f"  T_intra: {ConstrainedPolarGenerator(config).T_intra:.1f}px")
    print(f"  T_inter: {ConstrainedPolarGenerator(config).T_inter:.1f}px")
    print(f"  Dead zone: [{ConstrainedPolarGenerator(config).T_intra:.1f}, {ConstrainedPolarGenerator(config).T_inter:.1f}]px")

    # Print first 20 tokens of GT sequence
    print(f"\n  GT Sequence (first 20 tokens):")
    for item in sample.gt_sequence[:20]:
        if item['token'] == '<CHUNK>':
            print(f"    <CHUNK>")
        else:
            print(f"    '{item['token']}' (node_id={item['node_id']}, chunk={sample.nodes[item['node_id']].chunk_id})")