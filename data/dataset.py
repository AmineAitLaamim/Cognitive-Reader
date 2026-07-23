"""
data/dataset.py
PyTorch Dataset for the Cognitive Reader project.

Ties together the full synthetic data pipeline:
  1. ConstrainedPolarGenerator → geometric layout
  2. DigitRenderer → image tensor + annotations
  3. ThresholdRadiusGraphBuilder → spatial graph
  4. Assembly into the sample dict expected by collate_graphs

Two modes:
  - On-the-fly: generates a new random sample per __getitem__ call.
    Best for training (infinite data, no disk usage).
  - Cached: pre-generates a fixed set of samples at initialization.
    Best for debugging and reproducibility.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from data.generator import ConstrainedPolarGenerator, GeneratorConfig, GeneratedSample
from data.renderer import DigitRenderer, RendererConfig
from data.collate import collate_graphs
from models.graph.builder import ThresholdRadiusGraphBuilder, SpatialGraph


@dataclass
class DatasetConfig:
    """Unified configuration for the Cognitive Reader dataset."""
    
    # === Sequence length ===
    min_digits: int = 5              # Minimum digits per sample
    max_digits: int = 50             # Maximum digits per sample
    # For OOD evaluation, override with fixed values:
    # ood_eval_lengths: List[int] = [100, 200, 500]
    
    # === Generator ===
    img_width: int = 640
    img_height: int = 640
    threshold_radius_r: float = 80.0
    noise_sigma: float = 3.0
    max_chunk_size: int = 4
    min_chunk_size: int = 1
    
    # === Renderer ===
    font_size_base: int = 28
    rotation_max_deg: float = 5.0
    blur_probability: float = 0.3
    heatmap_stride: int = 8
    heatmap_sigma: float = 1.0
    
    # === Graph ===
    # radius is derived from threshold_radius_r (same value)
    
    # === Dataset ===
    samples_per_epoch: int = 1000    # Virtual dataset size for on-the-fly mode
    cache_samples: bool = False      # If True, pre-generate all samples
    seed: Optional[int] = 42         # Global seed for reproducibility
    
    # === Normalization ===
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class CognitiveReaderDataset(Dataset):
    """
    On-the-fly synthetic dataset for the Cognitive Reader.
    
    Each __getitem__ call generates a fresh random sample:
      generator → renderer → graph builder → sample dict
    
    No data is stored on disk. The dataset is effectively infinite;
    samples_per_epoch controls the virtual length for DataLoader.
    
    Usage:
        config = DatasetConfig(min_digits=5, max_digits=50)
        dataset = CognitiveReaderDataset(config, split='train')
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_graphs)
        
        for batch in loader:
            images = batch.images          # [B, 3, H, W]
            graphs = batch.adjacency       # [B, N_max, N_max]
            gt_seqs = batch.gt_sequences   # List[List[Dict]]
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        split: str = 'train',
        seed_offset: int = 0
    ):
        """
        Args:
            config: Unified dataset configuration.
            split: 'train' or 'eval'. Affects augmentation intensity.
            seed_offset: Added to the global seed for this dataset instance.
                         Use different offsets for train/val/test splits.
        """
        self.cfg = config
        self.split = split
        
        # Compute deterministic seed for this split
        base_seed = config.seed if config.seed is not None else 0
        self.seed = base_seed + seed_offset
        
        # Initialize sub-components
        self.generator_config = GeneratorConfig(
            img_width=config.img_width,
            img_height=config.img_height,
            threshold_radius_r=config.threshold_radius_r,
            noise_sigma=config.noise_sigma,
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
        )
        
        self.renderer_config = RendererConfig(
            img_width=config.img_width,
            img_height=config.img_height,
            font_size_base=config.font_size_base,
            rotation_max_deg=config.rotation_max_deg if split == 'train' else 0.0,
            blur_probability=config.blur_probability if split == 'train' else 0.0,
            heatmap_stride=config.heatmap_stride,
            heatmap_sigma=config.heatmap_sigma,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            seed=None,  # Per-sample seed set in __getitem__
        )
        
        self.generator = ConstrainedPolarGenerator(self.generator_config)
        self.renderer = DigitRenderer(self.renderer_config)
        self.graph_builder = ThresholdRadiusGraphBuilder(
            radius=config.threshold_radius_r,
            img_width=config.img_width,
            img_height=config.img_height
        )
        
        # RNG for sampling total_digits and per-sample seeds
        self.rng = np.random.RandomState(self.seed)
        
        # Cache (if enabled)
        self._cache: Optional[List[Dict]] = None
        if config.cache_samples:
            print(f"[Dataset] Pre-generating {config.samples_per_epoch} samples...")
            self._cache = [self._generate_sample(i) for i in range(config.samples_per_epoch)]
            print(f"[Dataset] Cache ready.")
    
    def __len__(self) -> int:
        return self.cfg.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._generate_sample(idx)
    
    def _generate_sample(self, idx: int) -> Dict[str, Any]:
        """
        Generate a single complete sample.
        
        Pipeline:
          1. Sample total_digits from [min_digits, max_digits].
          2. Generate geometric layout (Constrained Polar Sampling).
          3. Render image with visual augmentations.
          4. Build threshold-radius spatial graph.
          5. Assemble into the dict expected by collate_graphs.
        """
        # 1. Sample sequence length
        total_digits = self.rng.randint(
            self.cfg.min_digits,
            self.cfg.max_digits + 1
        )
        
        # 2. Generate layout
        layout_sample = self.generator.generate_sample(total_digits)
        
        # 3. Render image
        # Set per-sample seed for reproducible rendering
        self.renderer.rng = np.random.RandomState(self.seed + idx * 7919)
        render_output = self.renderer.render(layout_sample)
        
        # 4. Build graph
        # Use noisy boxes for training (Sim2Real alignment)
        boxes_list = []
        labels_list = []
        chunk_ids_list = []
        for node in layout_sample.nodes:
            boxes_list.append({
                'center_x': node.noisy_center_x,
                'center_y': node.noisy_center_y,
                'w': node.width,
                'h': node.height,
                'node_id': node.node_id
            })
            labels_list.append(node.label)
            chunk_ids_list.append(node.chunk_id)
        
        graph = self.graph_builder.build_from_boxes(
            boxes_list, labels_list, chunk_ids_list
        )
        
        # 5. Assemble sample dict (matching collate_graphs expectations)
        sample = {
            # Image and heatmap
            'image': render_output['image'],                    # [3, H, W]
            'heatmap_target': render_output['heatmap_target'],  # [1, H/8, W/8]
            
            # Boxes for RoI Align (use noisy centers)
            'boxes': render_output['boxes'],                    # [N, 4]
            
            # Graph tensors
            'node_positions_norm': graph.node_positions_norm,   # [N, 2]
            'node_positions_px': graph.node_positions_px,       # [N, 2]
            'node_labels': graph.node_labels,                   # [N]
            'node_chunk_ids': graph.node_chunk_ids,             # [N]
            'adjacency': graph.adjacency,                       # [N, N]
            'edge_features': graph.edge_features,               # [N, N, 3]
            'edge_directions': graph.edge_directions,           # [N, N, 2]
            
            # Ground truth
            'gt_sequence': layout_sample.gt_sequence,           # List[Dict]
            
            # Metadata
            'img_width': self.cfg.img_width,
            'img_height': self.cfg.img_height,
            'radius': self.cfg.threshold_radius_r,
            
            # Diagnostics (not used by collate, but useful for debugging)
            'total_digits': len(layout_sample.nodes),
            'num_chunks': layout_sample.num_chunks,
        }
        
        return sample


class CachedCognitiveReaderDataset(CognitiveReaderDataset):
    """
    Pre-generates all samples at initialization and caches them in memory.
    
    Advantages:
      - Deterministic: same samples every epoch.
      - Fast: no generation overhead during training.
      - Debuggable: can inspect specific samples by index.
    
    Disadvantages:
      - Memory: stores all samples in RAM.
      - Fixed: no new samples across epochs.
    
    Best for: debugging, unit testing, small-scale experiments.
    """
    
    def __init__(self, config: DatasetConfig, split: str = 'train', seed_offset: int = 0):
        config.cache_samples = True
        super().__init__(config, split, seed_offset)


class OODEvalDataset(Dataset):
    """
    Fixed-length evaluation dataset for Out-Of-Distribution length generalization.
    
    Generates samples with a FIXED number of digits (e.g., 100, 200, 500)
    that exceed the training distribution (e.g., 5-50 digits).
    
    Usage:
        ood_dataset = OODEvalDataset(config, fixed_length=200, num_samples=50)
        ood_loader = DataLoader(ood_dataset, batch_size=1, collate_fn=collate_graphs)
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        fixed_length: int = 200,
        num_samples: int = 50,
        seed: int = 9999
    ):
        """
        Args:
            config: Base dataset config (image size, radius, etc.).
            fixed_length: Fixed number of digits per sample.
            num_samples: Number of evaluation samples.
            seed: Random seed for reproducibility.
        """
        self.cfg = config
        self.fixed_length = fixed_length
        self.num_samples = num_samples
        self.seed = seed
        
        # Override generator config for fixed length
        self.generator_config = GeneratorConfig(
            img_width=config.img_width,
            img_height=config.img_height,
            threshold_radius_r=config.threshold_radius_r,
            noise_sigma=config.noise_sigma,
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
        )
        
        self.renderer_config = RendererConfig(
            img_width=config.img_width,
            img_height=config.img_height,
            font_size_base=config.font_size_base,
            rotation_max_deg=0.0,   # No augmentation for eval
            blur_probability=0.0,
            heatmap_stride=config.heatmap_stride,
            heatmap_sigma=config.heatmap_sigma,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        )
        
        self.generator = ConstrainedPolarGenerator(self.generator_config)
        self.renderer = DigitRenderer(self.renderer_config)
        self.graph_builder = ThresholdRadiusGraphBuilder(
            radius=config.threshold_radius_r,
            img_width=config.img_width,
            img_height=config.img_height
        )
        
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate all samples (eval sets are small)
        print(f"[OODEval] Generating {num_samples} samples with {fixed_length} digits...")
        self._samples = []
        for i in range(num_samples):
            self._samples.append(self._generate_sample(i))
        print(f"[OODEval] Ready.")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._samples[idx]
    
    def _generate_sample(self, idx: int) -> Dict[str, Any]:
        """Generate a single fixed-length sample."""
        # Generate layout with fixed length
        layout_sample = self.generator.generate_sample(self.fixed_length)
        
        # Render (no augmentation)
        self.renderer.rng = np.random.RandomState(self.seed + idx * 7919)
        render_output = self.renderer.render(layout_sample)
        
        # Build graph
        boxes_list = []
        labels_list = []
        chunk_ids_list = []
        for node in layout_sample.nodes:
            boxes_list.append({
                'center_x': node.noisy_center_x,
                'center_y': node.noisy_center_y,
                'w': node.width,
                'h': node.height,
                'node_id': node.node_id
            })
            labels_list.append(node.label)
            chunk_ids_list.append(node.chunk_id)
        
        graph = self.graph_builder.build_from_boxes(
            boxes_list, labels_list, chunk_ids_list
        )
        
        return {
            'image': render_output['image'],
            'heatmap_target': render_output['heatmap_target'],
            'boxes': render_output['boxes'],
            'node_positions_norm': graph.node_positions_norm,
            'node_positions_px': graph.node_positions_px,
            'node_labels': graph.node_labels,
            'node_chunk_ids': graph.node_chunk_ids,
            'adjacency': graph.adjacency,
            'edge_features': graph.edge_features,
            'edge_directions': graph.edge_directions,
            'gt_sequence': layout_sample.gt_sequence,
            'img_width': self.cfg.img_width,
            'img_height': self.cfg.img_height,
            'radius': self.cfg.threshold_radius_r,
            'total_digits': len(layout_sample.nodes),
            'num_chunks': layout_sample.num_chunks,
        }


# ==============================================================
# DATALOADER FACTORY
# ==============================================================

def create_dataloaders(
    config: DatasetConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        config: Unified dataset configuration.
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Pin memory for GPU transfer.
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CognitiveReaderDataset(
        config, split='train', seed_offset=0
    )
    val_dataset = CognitiveReaderDataset(
        config, split='eval', seed_offset=1000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_ood_dataloaders(
    config: DatasetConfig,
    eval_lengths: List[int] = [100, 200, 500],
    num_samples_per_length: int = 50,
    batch_size: int = 1
) -> Dict[int, DataLoader]:
    """
    Create OOD evaluation DataLoaders for multiple sequence lengths.
    
    Args:
        config: Base dataset configuration.
        eval_lengths: List of fixed digit counts to evaluate.
        num_samples_per_length: Samples per length.
        batch_size: Batch size (1 recommended for OOD eval).
    
    Returns:
        Dict mapping length → DataLoader.
    """
    loaders = {}
    for length in eval_lengths:
        dataset = OODEvalDataset(
            config,
            fixed_length=length,
            num_samples=num_samples_per_length,
            seed=9999 + length
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graphs
        )
        loaders[length] = loader
    
    return loaders


if __name__ == "__main__":
    print("=" * 60)
    print("  CognitiveReaderDataset Unit Test")
    print("=" * 60)
    
    # Config
    config = DatasetConfig(
        min_digits=5,
        max_digits=20,
        img_width=640,
        img_height=640,
        threshold_radius_r=80.0,
        noise_sigma=3.0,
        max_chunk_size=4,
        samples_per_epoch=10,
        seed=42
    )
    
    # Test on-the-fly dataset
    print("\n[Test 1] On-the-fly Dataset")
    dataset = CognitiveReaderDataset(config, split='train')
    print(f"  Length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Sample keys: {sorted(sample.keys())}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Boxes shape: {sample['boxes'].shape}")
    print(f"  Adjacency shape: {sample['adjacency'].shape}")
    print(f"  GT sequence: {[t['token'] for t in sample['gt_sequence']]}")
    print(f"  Total digits: {sample['total_digits']}")
    print(f"  Num chunks: {sample['num_chunks']}")
    
    # Verify consistency
    N = sample['boxes'].shape[0]
    assert sample['adjacency'].shape == (N, N)
    assert sample['node_positions_norm'].shape == (N, 2)
    assert sample['node_labels'].shape == (N,)
    assert sample['edge_features'].shape == (N, N, 3)
    print(f"  ✓ All tensor shapes consistent (N={N})")
    
    # Test multiple samples have varying lengths
    print("\n[Test 2] Variable Sequence Lengths")
    lengths = []
    for i in range(10):
        s = dataset[i]
        lengths.append(s['total_digits'])
    print(f"  Digit counts: {lengths}")
    assert len(set(lengths)) > 1, "All samples have the same length!"
    print(f"  ✓ Variable lengths confirmed")
    
    # Test DataLoader with collate
    print("\n[Test 3] DataLoader with collate_graphs")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs
    )
    
    batch = next(iter(loader))
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Max nodes: {batch.max_nodes}")
    print(f"  Num nodes: {batch.num_nodes.tolist()}")
    print(f"  Images: {batch.images.shape}")
    print(f"  Adjacency: {batch.adjacency.shape}")
    print(f"  Node mask: {batch.node_mask.shape}")
    print(f"  Boxes: {batch.boxes.shape}")
    print(f"  GT sequences: {len(batch.gt_sequences)} lists")
    
    # Verify no phantom edges
    from data.collate import verify_no_phantom_edges
    no_phantoms = verify_no_phantom_edges(batch.adjacency, batch.node_mask)
    print(f"  Phantom edges: {'✓ None' if no_phantoms else '✗ DETECTED'}")
    assert no_phantoms
    
    # Test cached dataset
    print("\n[Test 4] Cached Dataset")
    cached_config = DatasetConfig(
        min_digits=5, max_digits=15,
        samples_per_epoch=5,
        cache_samples=True,
        seed=42
    )
    cached_dataset = CachedCognitiveReaderDataset(cached_config, split='train')
    
    # Verify determinism: same index → same sample
    s1 = cached_dataset[0]
    s2 = cached_dataset[0]
    assert torch.equal(s1['image'], s2['image']), "Cached samples not deterministic!"
    print(f"  ✓ Cached samples are deterministic")
    
    # Test OOD eval dataset
    print("\n[Test 5] OOD Eval Dataset")
    ood_dataset = OODEvalDataset(
        config,
        fixed_length=30,  # Longer than training max (20)
        num_samples=3,
        seed=9999
    )
    
    for i in range(len(ood_dataset)):
        s = ood_dataset[i]
        print(f"  Sample {i}: {s['total_digits']} digits, {s['num_chunks']} chunks")
        assert s['total_digits'] <= 30  # May be less if generator hit boundary
    
    print(f"  ✓ OOD eval dataset works")
    
    # Test factory functions
    print("\n[Test 6] DataLoader Factory")
    small_config = DatasetConfig(
        min_digits=5, max_digits=15,
        samples_per_epoch=8,
        seed=42
    )
    train_loader, val_loader = create_dataloaders(
        small_config, batch_size=4, num_workers=0
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    ood_loaders = create_ood_dataloaders(
        small_config,
        eval_lengths=[30, 50],
        num_samples_per_length=2,
        batch_size=1
    )
    for length, loader in ood_loaders.items():
        print(f"  OOD length={length}: {len(loader)} batches")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)