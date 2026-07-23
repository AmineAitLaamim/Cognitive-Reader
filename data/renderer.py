"""
data/renderer.py
Synthetic digit image renderer for the Cognitive Reader project.

Takes the geometric layout from ConstrainedPolarGenerator and renders
actual digit glyphs onto a canvas image with visual augmentations.

Pipeline:
  1. Create blank canvas with textured background.
  2. For each digit node: render glyph with random font, size, rotation.
  3. Apply global augmentations: noise, blur, color jitter.
  4. Convert to normalized tensor.
  5. Generate heatmap target for detector training.

The renderer is fully configurable and deterministic given a seed.
"""

import torch
import numpy as np
import math
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class RendererConfig:
    """Configuration for the digit renderer."""
    # Canvas
    img_width: int = 640
    img_height: int = 640
    background_color: Tuple[int, int, int] = (245, 245, 240)  # Off-white
    
    # Font
    font_dirs: List[str] = field(default_factory=lambda: [
        '/usr/share/fonts/truetype',
        '/usr/share/fonts',
        '/System/Library/Fonts',
        'C:\\Windows\\Fonts',
        './fonts',
    ])
    font_size_base: int = 28          # Base font size in pixels
    font_size_jitter: float = 0.15    # ±15% size variation
    
    # Glyph rendering
    rotation_max_deg: float = 5.0     # Max rotation in degrees
    digit_color_mean: int = 30        # Mean digit color (0=black, 255=white)
    digit_color_std: int = 20         # Color variation std
    
    # Background
    bg_noise_std: float = 3.0         # Background Gaussian noise std
    bg_texture_strength: float = 0.02 # Subtle texture overlay strength
    
    # Global augmentations
    blur_probability: float = 0.3     # Probability of applying blur
    blur_radius_range: Tuple[float, float] = (0.3, 0.8)
    brightness_jitter: float = 0.05   # ±5% brightness
    contrast_jitter: float = 0.05     # ±5% contrast
    
    # Heatmap
    heatmap_stride: int = 8
    heatmap_sigma: float = 1.0        # Gaussian sigma in feature map pixels
    
    # Normalization
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)  # ImageNet
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Reproducibility
    seed: Optional[int] = None


class FontManager:
    """
    Manages a pool of TrueType fonts for digit rendering.
    Falls back to PIL default font if no TrueType fonts are found.
    """
    
    def __init__(self, font_dirs: List[str], font_size_base: int = 28):
        self.font_size_base = font_size_base
        self.font_paths: List[str] = []
        self._fonts_cache: Dict[int, Any] = {}  # size -> font object cache
        
        # Scan font directories for .ttf and .otf files
        for font_dir in font_dirs:
            if not os.path.isdir(font_dir):
                continue
            for root, dirs, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith(('.ttf', '.otf')):
                        self.font_paths.append(os.path.join(root, f))
        
        # Deduplicate
        self.font_paths = sorted(set(self.font_paths))
        
        if len(self.font_paths) == 0:
            print("[FontManager] No TrueType fonts found. Using PIL default font.")
        else:
            print(f"[FontManager] Found {len(self.font_paths)} fonts.")
    
    def get_font(self, size: int, rng: np.random.RandomState) -> Any:
        """
        Get a random font at the specified size.
        
        Args:
            size: Font size in pixels.
            rng: Random state for reproducibility.
        
        Returns:
            PIL ImageFont object.
        """
        if not PIL_AVAILABLE:
            return None
        
        if len(self.font_paths) == 0:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
        
        # Pick a random font
        font_path = self.font_paths[rng.randint(len(self.font_paths))]
        
        # Cache key: (path, size)
        cache_key = (font_path, size)
        if cache_key in self._fonts_cache:
            return self._fonts_cache[cache_key]
        
        try:
            font = ImageFont.truetype(font_path, size)
            self._fonts_cache[cache_key] = font
            return font
        except Exception:
            # Font file might be corrupted; try another
            try:
                return ImageFont.load_default()
            except Exception:
                return None


class DigitRenderer:
    """
    Renders synthetic digit images from geometric layouts.
    
    Usage:
        generator = ConstrainedPolarGenerator(config)
        sample = generator.generate_sample(total_digits=50)
        
        renderer = DigitRenderer(renderer_config)
        output = renderer.render(sample)
        
        image = output['image']           # [3, H, W] tensor
        boxes = output['boxes']           # [N, 4] tensor
        heatmap = output['heatmap_target'] # [1, H/8, W/8] tensor
    """
    
    def __init__(self, config: RendererConfig):
        self.cfg = config
        
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for digit rendering. "
                "Install with: pip install Pillow"
            )
        
        self.font_manager = FontManager(
            font_dirs=config.font_dirs,
            font_size_base=config.font_size_base
        )
        
        self.rng = np.random.RandomState(config.seed)
    
    def render(self, sample) -> Dict[str, Any]:
        """
        Render a GeneratedSample into an image tensor with annotations.
        
        Args:
            sample: GeneratedSample from ConstrainedPolarGenerator.
        
        Returns:
            Dict with:
              'image': [3, H, W] normalized tensor
              'boxes': [N, 4] tensor (x1, y1, x2, y2)
              'node_positions_norm': [N, 2] tensor
              'node_positions_px': [N, 2] tensor
              'node_labels': [N] tensor
              'node_chunk_ids': [N] tensor
              'heatmap_target': [1, H/8, W/8] tensor
              'gt_sequence': List[Dict]
              'img_width': int
              'img_height': int
              'radius': float
        """
        H = self.cfg.img_height
        W = self.cfg.img_width
        
        # 1. Create canvas
        canvas = self._create_canvas(H, W)
        draw = ImageDraw.Draw(canvas)
        
        # 2. Render each digit
        for node in sample.nodes:
            self._render_digit(
                canvas=canvas,
                draw=draw,
                digit=node.label,
                center_x=node.center_x,
                center_y=node.center_y,
                box_w=node.width,
                box_h=node.height,
                scale=node.scale
            )
        
        # 3. Apply global augmentations
        canvas = self._apply_augmentations(canvas)
        
        # 4. Convert to tensor
        image_tensor = self._to_tensor(canvas)  # [3, H, W]
        
        # 5. Build bounding boxes [N, 4]
        boxes = self._build_boxes(sample)  # [N, 4]
        
        # 6. Generate heatmap target
        centers_px = torch.tensor([
            [node.center_x, node.center_y] for node in sample.nodes
        ], dtype=torch.float32)
        heatmap_target = self._generate_heatmap(centers_px, H, W)  # [1, H/8, W/8]
        
        # 7. Build node tensors
        N = len(sample.nodes)
        node_positions_px = torch.tensor([
            [node.center_x, node.center_y] for node in sample.nodes
        ], dtype=torch.float32)
        
        node_positions_norm = node_positions_px.clone()
        node_positions_norm[:, 0] /= W
        node_positions_norm[:, 1] /= H
        
        node_labels = torch.tensor(
            [int(node.label) for node in sample.nodes], dtype=torch.long
        )
        node_chunk_ids = torch.tensor(
            [node.chunk_id for node in sample.nodes], dtype=torch.long
        )
        
        return {
            'image': image_tensor,
            'boxes': boxes,
            'node_positions_norm': node_positions_norm,
            'node_positions_px': node_positions_px,
            'node_labels': node_labels,
            'node_chunk_ids': node_chunk_ids,
            'heatmap_target': heatmap_target,
            'gt_sequence': sample.gt_sequence,
            'img_width': W,
            'img_height': H,
            'radius': 80.0  # Will be overridden by the dataset
        }
    
    def _create_canvas(self, H: int, W: int) -> Image.Image:
        """Create a canvas with textured background."""
        bg = self.cfg.background_color
        
        # Base color with slight random variation
        bg_r = max(0, min(255, bg[0] + self.rng.randint(-5, 6)))
        bg_g = max(0, min(255, bg[1] + self.rng.randint(-5, 6)))
        bg_b = max(0, min(255, bg[2] + self.rng.randint(-5, 6)))
        
        canvas = Image.new('RGB', (W, H), (bg_r, bg_g, bg_b))
        
        # Add subtle background noise
        if self.cfg.bg_noise_std > 0:
            noise = self.rng.normal(
                0, self.cfg.bg_noise_std, (H, W, 3)
            ).astype(np.float32)
            canvas_np = np.array(canvas, dtype=np.float32) + noise
            canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(canvas_np)
        
        # Add subtle texture (random faint lines or dots)
        if self.cfg.bg_texture_strength > 0:
            draw = ImageDraw.Draw(canvas)
            num_textures = self.rng.randint(5, 20)
            for _ in range(num_textures):
                x1 = self.rng.randint(0, W)
                y1 = self.rng.randint(0, H)
                x2 = x1 + self.rng.randint(-50, 51)
                y2 = y1 + self.rng.randint(-50, 51)
                gray = self.rng.randint(200, 240)
                draw.line([(x1, y1), (x2, y2)], fill=(gray, gray, gray), width=1)
        
        return canvas
    
    def _render_digit(
        self,
        canvas: Image.Image,
        draw: ImageDraw.Draw,
        digit: str,
        center_x: float,
        center_y: float,
        box_w: float,
        box_h: float,
        scale: float
    ) -> None:
        """
        Render a single digit glyph onto the canvas.
        
        The digit is rendered on a temporary transparent image,
        optionally rotated, then pasted onto the canvas at the
        correct position.
        """
        # Font size: base * scale * jitter
        jitter = 1.0 + self.rng.uniform(
            -self.cfg.font_size_jitter, self.cfg.font_size_jitter
        )
        font_size = max(8, int(self.cfg.font_size_base * scale * jitter))
        
        # Get random font
        font = self.font_manager.get_font(font_size, self.rng)
        
        # Digit color: dark with variation
        color_val = max(0, min(255, int(
            self.rng.normal(self.cfg.digit_color_mean, self.cfg.digit_color_std)
        )))
        digit_color = (color_val, color_val, color_val)
        
        # Render digit on a temporary image (with padding for rotation)
        pad = int(font_size * 0.5)
        tmp_size = font_size + 2 * pad
        tmp_img = Image.new('RGBA', (tmp_size, tmp_size), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp_img)
        
        # Get text bounding box for centering
        if font is not None:
            try:
                bbox = tmp_draw.textbbox((0, 0), digit, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = (tmp_size - text_w) / 2 - bbox[0]
                text_y = (tmp_size - text_h) / 2 - bbox[1]
            except Exception:
                text_x = pad
                text_y = pad
        else:
            text_x = pad
            text_y = pad
        
        tmp_draw.text(
            (text_x, text_y), digit,
            fill=digit_color + (255,),  # RGBA
            font=font
        )
        
        # Apply rotation
        rotation_deg = self.rng.uniform(
            -self.cfg.rotation_max_deg, self.cfg.rotation_max_deg
        )
        if abs(rotation_deg) > 0.5:
            tmp_img = tmp_img.rotate(
                rotation_deg,
                resample=Image.BICUBIC,
                expand=False,
                center=(tmp_size // 2, tmp_size // 2)
            )
        
        # Scale the temporary image to match the bounding box
        target_w = max(1, int(box_w * 1.3))  # Slight oversize for context
        target_h = max(1, int(box_h * 1.3))
        tmp_img = tmp_img.resize((target_w, target_h), Image.BICUBIC)
        
        # Paste onto canvas at the correct position
        paste_x = int(center_x - target_w / 2)
        paste_y = int(center_y - target_h / 2)
        
        # Use alpha channel as mask for compositing
        canvas.paste(tmp_img, (paste_x, paste_y), tmp_img)
    
    def _apply_augmentations(self, canvas: Image.Image) -> Image.Image:
        """Apply global visual augmentations to the canvas."""
        
        # Random blur
        if self.rng.random() < self.cfg.blur_probability:
            blur_radius = self.rng.uniform(*self.cfg.blur_radius_range)
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Brightness and contrast jitter
        canvas_np = np.array(canvas, dtype=np.float32)
        
        # Brightness
        brightness_factor = 1.0 + self.rng.uniform(
            -self.cfg.brightness_jitter, self.cfg.brightness_jitter
        )
        canvas_np *= brightness_factor
        
        # Contrast
        contrast_factor = 1.0 + self.rng.uniform(
            -self.cfg.contrast_jitter, self.cfg.contrast_jitter
        )
        mean_val = canvas_np.mean()
        canvas_np = (canvas_np - mean_val) * contrast_factor + mean_val
        
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(canvas_np)
        
        return canvas
    
    def _to_tensor(self, canvas: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to normalized tensor [3, H, W].
        
        Applies ImageNet normalization (mean/std).
        """
        # Convert to numpy: [H, W, 3] uint8
        img_np = np.array(canvas, dtype=np.float32) / 255.0  # [0, 1]
        
        # Normalize with ImageNet stats
        mean = np.array(self.cfg.normalize_mean, dtype=np.float32)
        std = np.array(self.cfg.normalize_std, dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # Convert to tensor: [3, H, W]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return tensor
    
    def _build_boxes(self, sample) -> torch.Tensor:
        """
        Build bounding box tensor [N, 4] from sample nodes.
        Format: (x1, y1, x2, y2) in pixel coordinates.
        """
        boxes = []
        for node in sample.nodes:
            x1 = node.center_x - node.width / 2
            y1 = node.center_y - node.height / 2
            x2 = node.center_x + node.width / 2
            y2 = node.center_y + node.height / 2
            boxes.append([x1, y1, x2, y2])
        
        if len(boxes) == 0:
            return torch.zeros(0, 4)
        
        return torch.tensor(boxes, dtype=torch.float32)
    
    def _generate_heatmap(
        self,
        centers_px: torch.Tensor,
        img_height: int,
        img_width: int
    ) -> torch.Tensor:
        """
        Generate ground-truth heatmap with Gaussian blobs.
        
        Args:
            centers_px: [N, 2] — digit centers in pixels (x, y).
            img_height: Image height.
            img_width: Image width.
        
        Returns:
            [1, H/stride, W/stride] heatmap tensor.
        """
        stride = self.cfg.heatmap_stride
        sigma = self.cfg.heatmap_sigma
        h = img_height // stride
        w = img_width // stride
        
        heatmap = torch.zeros(1, h, w)
        
        N = centers_px.shape[0]
        radius = int(3 * sigma)
        
        for i in range(N):
            cx = centers_px[i, 0].item() / stride
            cy = centers_px[i, 1].item() / stride
            
            cx_int = int(round(cx))
            cy_int = int(round(cy))
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y = cy_int + dy
                    x = cx_int + dx
                    if 0 <= y < h and 0 <= x < w:
                        val = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                        heatmap[0, y, x] = max(heatmap[0, y, x].item(), val)
        
        return heatmap


class SimpleDigitRenderer(DigitRenderer):
    """
    Lightweight renderer that uses PIL's default font only.
    Faster than DigitRenderer but less visually diverse.
    Useful for quick prototyping and debugging.
    """
    
    def __init__(self, config: RendererConfig):
        # Override font settings for simplicity
        config.font_dirs = []  # Force default font
        config.rotation_max_deg = 0.0  # No rotation
        config.blur_probability = 0.0  # No blur
        super().__init__(config)
    
    def _render_digit(self, canvas, draw, digit, center_x, center_y, box_w, box_h, scale):
        """Simplified rendering: draw text directly on canvas."""
        font_size = max(8, int(self.cfg.font_size_base * scale))
        
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        color_val = max(0, min(255, int(
            self.rng.normal(self.cfg.digit_color_mean, self.cfg.digit_color_std)
        )))
        
        # Approximate text position (top-left from center)
        text_x = int(center_x - box_w / 2)
        text_y = int(center_y - box_h / 2)
        
        draw.text(
            (text_x, text_y), digit,
            fill=(color_val, color_val, color_val),
            font=font
        )


if __name__ == "__main__":
    print("=" * 60)
    print("  DigitRenderer Unit Test")
    print("=" * 60)
    
    if not PIL_AVAILABLE:
        print("\n  ✗ Pillow not installed. Skipping render tests.")
        print("    Install with: pip install Pillow")
    else:
        # Create a fake sample (mimicking GeneratedSample)
        from dataclasses import dataclass
        
        @dataclass
        class FakeNode:
            node_id: int
            label: str
            center_x: float
            center_y: float
            noisy_center_x: float
            noisy_center_y: float
            width: float
            height: float
            scale: float
            chunk_id: int
        
        @dataclass
        class FakeSample:
            nodes: List
            gt_sequence: List
            img_width: int
            img_height: int
            num_chunks: int
            total_digits: int
        
        # Create 6 digits in 2 chunks
        nodes = [
            FakeNode(0, '3', 100, 100, 101, 99, 20, 30, 1.0, 0),
            FakeNode(1, '8', 140, 102, 141, 101, 20, 30, 1.0, 0),
            FakeNode(2, '4', 180, 98, 179, 99, 20, 30, 1.0, 0),
            FakeNode(3, '1', 350, 250, 351, 249, 20, 30, 1.0, 1),
            FakeNode(4, '7', 390, 252, 389, 253, 20, 30, 1.0, 1),
            FakeNode(5, '2', 430, 248, 431, 247, 20, 30, 1.0, 1),
        ]
        
        gt_seq = [
            {'token': '3', 'node_id': 0, 'mode': 'READ'},
            {'token': '8', 'node_id': 1, 'mode': 'READ'},
            {'token': '4', 'node_id': 2, 'mode': 'READ'},
            {'token': '<CHUNK>', 'node_id': None, 'mode': 'CHUNK'},
            {'token': '1', 'node_id': 3, 'mode': 'READ'},
            {'token': '7', 'node_id': 4, 'mode': 'READ'},
            {'token': '2', 'node_id': 5, 'mode': 'READ'},
        ]
        
        sample = FakeSample(
            nodes=nodes,
            gt_sequence=gt_seq,
            img_width=640,
            img_height=640,
            num_chunks=2,
            total_digits=6
        )
        
        # Render
        config = RendererConfig(
            img_width=640,
            img_height=640,
            seed=42
        )
        
        renderer = DigitRenderer(config)
        output = renderer.render(sample)
        
        print(f"\n  Rendered image:  {output['image'].shape}")       # [3, 640, 640]
        print(f"  Bounding boxes:  {output['boxes'].shape}")         # [6, 4]
        print(f"  Heatmap target:  {output['heatmap_target'].shape}") # [1, 80, 80]
        print(f"  Node positions:  {output['node_positions_px'].shape}") # [6, 2]
        print(f"  Node labels:     {output['node_labels'].tolist()}")
        print(f"  GT sequence:     {[t['token'] for t in output['gt_sequence']]}")
        
        # Verify shapes
        assert output['image'].shape == (3, 640, 640)
        assert output['boxes'].shape == (6, 4)
        assert output['heatmap_target'].shape == (1, 80, 80)
        assert output['node_positions_px'].shape == (6, 2)
        assert output['node_labels'].shape == (6,)
        
        # Verify heatmap has peaks at digit locations
        hm = output['heatmap_target']
        assert hm.max() > 0.9, "Heatmap should have peaks near 1.0"
        print(f"  Heatmap max:     {hm.max():.4f}")
        print(f"  Heatmap nonzero: {(hm > 0).sum().item()} pixels")
        
        # Verify boxes are valid (x1 < x2, y1 < y2)
        boxes = output['boxes']
        assert (boxes[:, 0] < boxes[:, 2]).all(), "x1 must be < x2"
        assert (boxes[:, 1] < boxes[:, 3]).all(), "y1 must be < y2"
        print(f"  Box validity:    ✓ All boxes have x1<x2, y1<y2")
        
        # Save rendered image for visual inspection
        try:
            # Denormalize for saving
            img_tensor = output['image']
            mean = torch.tensor(config.normalize_mean).view(3, 1, 1)
            std = torch.tensor(config.normalize_std).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = img_denorm.clamp(0, 1)
            img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save('test_render.png')
            print(f"\n  ✓ Saved test_render.png for visual inspection")
        except Exception as e:
            print(f"\n  Could not save image: {e}")
        
        # Test SimpleDigitRenderer
        print(f"\n  SimpleDigitRenderer Test:")
        simple_config = RendererConfig(img_width=640, img_height=640, seed=42)
        simple_renderer = SimpleDigitRenderer(simple_config)
        simple_output = simple_renderer.render(sample)
        assert simple_output['image'].shape == (3, 640, 640)
        print(f"    ✓ Simple renderer works")
    
    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)