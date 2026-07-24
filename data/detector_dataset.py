# data/detector_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.generator import ConstrainedPolarGenerator, GeneratorConfig
from data.renderer import DigitRenderer, RendererConfig


class DetectorDataset(Dataset):
    """Images + heatmap targets for detector pre-training."""

    def __init__(self, num_samples, dataset_config):
        self.num_samples = num_samples
        self.gen_config = GeneratorConfig(
            img_width=dataset_config.img_width,
            img_height=dataset_config.img_height,
            threshold_radius_r=dataset_config.threshold_radius_r,
            noise_sigma=dataset_config.noise_sigma,
            max_chunk_size=dataset_config.max_chunk_size,
        )
        self.render_config = RendererConfig(
            img_width=dataset_config.img_width,
            img_height=dataset_config.img_height,
            seed=None,
        )
        self.generator = ConstrainedPolarGenerator(self.gen_config)
        self.renderer = DigitRenderer(self.render_config)
        self.rng = np.random.RandomState(42)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # clamp to the joint curriculum so the two stages stay consistent
        hi = min(51, getattr(self, '_max_digits', 50) + 1)
        total_digits = self.rng.randint(5, hi)
        layout = self.generator.generate_sample(total_digits)  # raises if infeasible
        self.renderer.rng = np.random.RandomState(idx)
        render_out = self.renderer.render(layout)
        return {
            'image': render_out['image'],
            'heatmap_target': render_out['heatmap_target'],
        }


def make_det_loaders(dataset_config, train_n=500, val_n=50,
                     batch_size=8, num_workers=2):
    train_ds = DetectorDataset(train_n, dataset_config)
    val_ds = DetectorDataset(val_n, dataset_config)
    for ds in (train_ds, val_ds):
        ds._max_digits = dataset_config.max_digits
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ld, val_ld