"""
data/generator.py
Constrained Polar Sampling data generator for the Cognitive Reader project.
Generates 2D digit layouts with mathematically guaranteed chunk boundaries.

Contract (fixed):
  generate_sample(N) returns exactly N nodes, or raises ValueError if N cannot
  be placed on the canvas under the geometric constraints. It NEVER returns a
  silently-truncated layout (the old early-return on placement failure is gone).
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

    # Derived intra/inter chunk bounds (factors of r)
    r_intra_factor: float = 0.8       # max intra-chunk distance = 0.8 * r
    r_inter_factor: float = 1.5       # min inter-chunk distance = 1.5 * r
    r_max_inter_factor: float = 2.5   # max inter-chunk distance = 2.5 * r

    # Intra-chunk angle distribution (radians): controls chain straightness.
    intra_angle_std: float = 5.0 * (math.pi / 180.0)  # 5 degrees

    # [DEPRECATED, unused] The old fixed inter-chunk cone (225deg +/- 10deg) is
    # what marched the walk off the top edge and killed placement at 5 nodes.
    # Inter direction is now sampled over the full circle (see
    # _sample_inter_chunk_position). Kept here only so no existing constructor
    # call that passes these kwargs raises TypeError. Do not rely on them.
    intra_angle_mean: float = 0.0
    inter_angle_mean: float = 225.0 * (math.pi / 180.0)
    inter_angle_std: float = 10.0 * (math.pi / 180.0)

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

    # Rejection sampling max attempts per placement (full-circle search)
    max_resample_attempts: int = 80

    # Minimum intra-digit distance within a chain (prevent overlap)
    min_digit_gap: float = 25.0

    # Min separation between a new node and any node of a *different* chunk,
    # so clusters do not draw on top of each other.
    min_cluster_gap: float = 22.0

    # Outer retry budget. Each attempt uses a fresh random start and a fresh
    # per-chunk chain orientation; the longest successful layout wins.
    max_outer_attempts: int = 40


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

    Guarantees (preserved from the original design):
      - Intra-chunk consecutive distance < r_intra_factor * r
      - Inter-chunk distance (chunk k last -> chunk k+1 first) > r_inter_factor * r
      - Ground-truth <CHUNK> tokens align exactly with physical spatial gaps
      - Sim2Real noise is injected for detector training

    Contract:
      generate_sample(N) returns exactly N nodes, or raises ValueError if N
      cannot be placed on the canvas under the geometric constraints. It never
      returns a silently-truncated layout.
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

    # ==============================================================
    # PUBLIC
    # ==============================================================

    def generate_sample(self, total_digits: int) -> GeneratedSample:
        """
        Generate a 2D digit layout with exactly `total_digits` nodes.

        Uses an outer retry loop: each attempt draws a fresh random start and
        fresh per-chunk chain orientations, and the inter-chunk jump direction
        is sampled over the full circle so the walk can snake around the canvas
        instead of marching off one edge. The longest layout across attempts is
        remembered; if no attempt reaches `total_digits`, raises.
        """
        best_nodes: Optional[List[DigitNode]] = None
        best_gt: Optional[List[Dict[str, Any]]] = None
        best_chunks: int = 0

        for _attempt in range(self.cfg.max_outer_attempts):
            nodes, gt, n_chunks = self._try_place(total_digits)
            if len(nodes) >= total_digits:
                return GeneratedSample(
                    nodes=nodes,
                    gt_sequence=gt,
                    img_width=self.cfg.img_width,
                    img_height=self.cfg.img_height,
                    num_chunks=n_chunks,
                    total_digits=len(nodes),
                )
            if best_nodes is None or len(nodes) > len(best_nodes):
                best_nodes, best_gt, best_chunks = nodes, gt, n_chunks

        achieved = len(best_nodes) if best_nodes else 0
        raise ValueError(
            f"ConstrainedPolarGenerator could not place {total_digits} digits in "
            f"{self.cfg.max_outer_attempts} attempts (best achieved: {achieved}). "
            f"This is a canvas-capacity limit at threshold_radius_r={self.r:.0f} on a "
            f"{self.cfg.img_width}x{self.cfg.img_height} canvas, not a code bug. "
            f"Lower total_digits / dataset_config.max_digits, lower threshold_radius_r, "
            f"or enlarge the canvas."
        )

    # ==============================================================
    # SINGLE PLACEMENT ATTEMPT
    # ==============================================================

    def _try_place(
        self, total_digits: int
    ) -> Tuple[List[DigitNode], List[Dict[str, Any]], int]:
        """One random walk attempt. Returns (nodes, gt, num_chunks)."""
        nodes: List[DigitNode] = []
        gt: List[Dict[str, Any]] = []

        # Start anywhere in the usable canvas (old code pinned x to 160).
        m = self.cfg.boundary_margin
        current_x = np.random.uniform(m, self.cfg.img_width - m)
        current_y = np.random.uniform(m, self.cfg.img_height - m)

        chunk_size = 0
        chunk_id = 0
        node_id = 0
        is_first = True
        intra_base = np.random.uniform(0.0, 2 * math.pi)   # random chain heading

        while node_id < total_digits:
            need_inter = False
            if is_first:
                need_inter = False
            elif chunk_size >= self.cfg.max_chunk_size:
                need_inter = True
            elif self._approaching_boundary(current_x, current_y, mode='intra'):
                need_inter = True

            if need_inter and not is_first:
                # INTER-CHUNK PLACEMENT (new chunk)
                pos = self._sample_inter_chunk_position(current_x, current_y, nodes)
                if pos is None:
                    break                                   # stuck -> let outer retry roll
                new_x, new_y = pos
                gt.append({'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK'})
                chunk_id += 1
                chunk_size = 1
                intra_base = np.random.uniform(0.0, 2 * math.pi)   # new heading per chain
            else:
                if is_first:
                    new_x, new_y = current_x, current_y
                    chunk_size = 1
                    is_first = False
                else:
                    # INTRA-CHUNK PLACEMENT
                    pos = self._sample_intra_chunk_position(
                        current_x, current_y, nodes, chunk_id, intra_base
                    )
                    if pos is None:
                        # intra crowded/blocked -> force a chunk break
                        pos = self._sample_inter_chunk_position(current_x, current_y, nodes)
                        if pos is None:
                            break
                        new_x, new_y = pos
                        gt.append({'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK'})
                        chunk_id += 1
                        chunk_size = 1
                        intra_base = np.random.uniform(0.0, 2 * math.pi)
                    else:
                        new_x, new_y = pos
                        chunk_size += 1

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
                chunk_id=chunk_id,
            )
            nodes.append(node)

            # Append digit to ground-truth sequence
            gt.append({'token': label, 'node_id': node_id, 'mode': 'READ'})

            # Update state
            current_x = new_x
            current_y = new_y
            node_id += 1

        return nodes, gt, (chunk_id + 1)

    # ==============================================================
    # SAMPLING HELPERS
    # ==============================================================

    def _sample_intra_chunk_position(
        self, current_x: float, current_y: float,
        nodes: List[DigitNode], chunk_id: int, base_angle: float,
    ) -> Optional[Tuple[float, float]]:
        """Next digit in the same chunk. Distance < r_intra for any angle.
        Returns (x, y) on success or None on failure (NOT a (None, None) tuple,
        so callers' `if pos is None:` guards work)."""
        d_min = max(self.cfg.min_digit_gap, 1.2 * self.cfg.base_digit_w)
        for _ in range(self.cfg.max_resample_attempts):
            d = np.random.uniform(d_min, self.r_intra)
            theta = np.random.normal(base_angle, self.cfg.intra_angle_std)
            new_x = current_x + d * math.cos(theta)
            new_y = current_y + d * math.sin(theta)
            if not self._is_within_bounds(new_x, new_y):
                continue
            if self._too_close_to_other_chunks(new_x, new_y, nodes, chunk_id):
                continue
            return new_x, new_y
        return None   # [FIX2] was `return None, None` -> broke the `is None` guard

    def _sample_inter_chunk_position(
        self, current_x: float, current_y: float,
        nodes: List[DigitNode],
    ) -> Optional[Tuple[float, float]]:
        """
        First digit of a new chunk. Distance > r_inter for any angle.
        Direction is uniform on the full circle, so from a boundary point the
        in-bounds hemisphere is always searchable (the old 10deg cone was not).
        A soft global gap keeps clusters from overlapping older digits.
        Returns (x, y) on success or None on failure.
        """
        for _ in range(self.cfg.max_resample_attempts):
            d = np.random.uniform(self.r_inter, self.r_max_inter)
            theta = np.random.uniform(0.0, 2 * math.pi)
            new_x = current_x + d * math.cos(theta)
            new_y = current_y + d * math.sin(theta)
            if not self._is_within_bounds(new_x, new_y):
                continue
            if self._too_close_to_other_chunks(new_x, new_y, nodes, exclude_chunk=None):
                continue
            return new_x, new_y
        return None   # [FIX2] was `return None, None` -> broke the `is None` guard

    def _too_close_to_other_chunks(
        self, x: float, y: float,
        nodes: List[DigitNode], exclude_chunk: Optional[int],
    ) -> bool:
        """True if (x, y) is within min_cluster_gap of any node outside exclude_chunk."""
        gap = self.cfg.min_cluster_gap
        gap2 = gap * gap
        for n in nodes:
            if exclude_chunk is not None and n.chunk_id == exclude_chunk:
                continue
            dx = x - n.center_x
            dy = y - n.center_y
            if dx * dx + dy * dy < gap2:
                return True
        return False

    def _is_within_bounds(self, x: float, y: float) -> bool:
        """Check if a position is within the canvas boundaries."""
        m = self.cfg.boundary_margin
        return (m <= x <= self.cfg.img_width - m) and (m <= y <= self.cfg.img_height - m)

    def _approaching_boundary(self, x: float, y: float, mode: str = 'intra') -> bool:
        """Check if the current position is too close to the canvas boundary."""
        m = self.cfg.boundary_margin
        buffer = self.r_intra if mode == 'intra' else self.r_inter
        return (
            x + buffer > self.cfg.img_width - m or
            y + buffer > self.cfg.img_height - m or
            x - buffer < m or
            y - buffer < m
        )

    # ==============================================================
    # INTROSPECTION (unchanged)
    # ==============================================================

    def get_bounding_boxes(self, sample: GeneratedSample, use_noise: bool = True) -> List[Dict]:
        """
        Extract bounding boxes from a generated sample.

        Args:
            sample: The generated sample.
            use_noise: If True, return noisy boxes (for detector training).
                       If False, return perfect boxes (for controller training).

        Returns:
            List of dicts with {x, y, w, h, center_x, center_y, label, node_id, chunk_id, scale}
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

                distances[i, j] = math.sqrt(dx * dx + dy * dy)
                same_chunk[i, j] = (ni.chunk_id == nj.chunk_id)

        return {
            'distances': distances,
            'same_chunk': same_chunk,
            'r_intra': self.r_intra,
            'r_inter': self.r_inter,
            'T_intra': self.T_intra,
            'T_inter': self.T_inter
        }


# ------------------------------------------------------------------
# Boundary guard for every call site (dataset / eval / scripts).
# generate_sample now raises on infeasible N, so this is belt-and-braces:
# it turns any future silent-truncation regression into a hard crash at the
# exact sample that caused it, instead of a 50-epoch silent failure.
# ------------------------------------------------------------------
def generate_full(generator: ConstrainedPolarGenerator, total_digits: int) -> GeneratedSample:
    """Call generate_sample and assert the count contract at the boundary."""
    layout = generator.generate_sample(total_digits)
    assert len(layout.nodes) == total_digits, (
        f"generator returned {len(layout.nodes)} nodes, expected {total_digits}"
    )
    return layout


# --- Validation helper ---
def generate_and_validate(config: GeneratorConfig, total_digits: int = 50) -> GeneratedSample:
    """
    Generate a sample and validate the *consecutive-pair* geometric constraints.
    Raises AssertionError if any constraint is violated.

    Note: only consecutive node pairs are checked. A 4-node chain with ~45px
    steps has its 1<->3 pair at ~90px > r_intra; that pair is not an edge and
    was never meant to satisfy r_intra, so the old "every same-chunk pair"
    assertion was a false failure waiting to happen on long chains.
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
            ni, nj = sample.nodes[i], sample.nodes[j]
            # Consecutive intra-chunk pair must be within r_intra
            if same_chunk[i, j] and abs(ni.node_id - nj.node_id) == 1:
                assert d <= gen.r_intra, (
                    f"Intra-chunk violation: nodes {i},{j} distance={d:.1f} > r_intra={gen.r_intra:.1f}"
                )
            # Consecutive inter-chunk pair must exceed r_inter
            if (not same_chunk[i, j]) and abs(ni.chunk_id - nj.chunk_id) == 1 \
                    and abs(ni.node_id - nj.node_id) == 1:
                assert d >= gen.r_inter, (
                    f"Inter-chunk violation: nodes {i},{j} distance={d:.1f} < r_inter={gen.r_inter:.1f}"
                )

    print(f"OK Generated {sample.total_digits} digits in {sample.num_chunks} chunks. Constraints satisfied.")
    return sample


if __name__ == "__main__":
    # Built-in capacity probe: for each N, success rate over 100 seeds.
    # The feasible_max line is the number you set dataset_config.max_digits
    # and ood_eval_lengths to (or lower r / enlarge the canvas to raise it).
    config = GeneratorConfig(
        img_width=640,
        img_height=640,
        threshold_radius_r=80.0,
        noise_sigma=3.0,
        max_chunk_size=4,
        min_chunk_size=1
    )

    gen = ConstrainedPolarGenerator(config)
    print(f"r_intra={gen.r_intra:.0f}  r_inter={gen.r_inter:.0f}  "
          f"T_intra={gen.T_intra:.1f}  T_inter={gen.T_inter:.1f}\n")
    print(f"{'N':>4}  {'success%':>9}  {'min':>4}  {'max':>4}  {'feasible':>8}")

    feasible_max = 0
    for N in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]:
        ok = 0
        achieved = []
        for s in range(100):
            np.random.seed(s)
            try:
                lay = gen.generate_sample(N)
                achieved.append(len(lay.nodes))
                if len(lay.nodes) >= N:
                    ok += 1
            except ValueError:
                achieved.append(0)
        rate = ok  # out of 100
        feas = rate >= 95
        if feas:
            feasible_max = N
        print(f"{N:>4}  {rate:>8}%  {min(achieved):>4}  {max(achieved):>4}  {'YES' if feas else 'no':>8}")

    print(f"\n=> empirical feasible_max (>=95% success) at r=80/640: {feasible_max}")
    print("   set dataset_config.max_digits and ood_eval_lengths <= this value,")
    print("   OR lower threshold_radius_r / enlarge the canvas to raise it.")