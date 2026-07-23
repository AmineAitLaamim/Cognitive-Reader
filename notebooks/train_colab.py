# %% [markdown]
# # Cognitive Reader — Training on Google Colab
#
# This notebook trains the Dual-Mode Cognitive Controller end-to-end.
#
# **Pipeline:**
# 1. Setup environment (GPU, dependencies, project files)
# 2. Detector pre-training (heatmap head)
# 3. Joint training (backbone + controller)
# 4. OOD evaluation
# 5. Visualization
#
# **Runtime:** ~2-4 hours on Colab Pro (A100), ~6-8 hours on free tier (T4)

# %% [markdown]
# ## 1. Environment Setup

# %%
# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow.")
    print("Go to Runtime -> Change runtime type -> GPU")

# %%
# Install dependencies
# !pip install -q torch torchvision
# !pip install -q Pillow matplotlib tensorboard pyyaml tqdm

# %%
# Mount Google Drive (for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Set project path on Drive
PROJECT_DIR = '/content/drive/MyDrive/cognitive_reader'
# !mkdir -p {PROJECT_DIR}

# %%
# Clone project files
# !git clone https://github.com/AmineAitLaamim/Cognitive-Reader {PROJECT_DIR} 2>/dev/null || true

# Verify all required directories are present
import os
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")
required = ['data', 'models', 'train', 'eval', 'utils']
missing = [d for d in required if not os.path.isdir(d)]
if missing:
    print(f"ERROR: Missing directories: {missing}")
    print("Upload project files manually or check the git URL.")
else:
    print(f"All directories present: {os.listdir('.')}")

# %%
# Add project to Python path
import sys
sys.path.insert(0, PROJECT_DIR)

# Verify imports
from data.generator import ConstrainedPolarGenerator, GeneratorConfig
from data.renderer import DigitRenderer, RendererConfig
from data.dataset import CognitiveReaderDataset, DatasetConfig, create_dataloaders
from data.collate import collate_graphs
from models.backbone.cnn import VisualBackbone
from models.controller.dual_mode import DualModeController
from models.detector.heatmap import HeatmapHead, DetectorTrainer, DetectorTrainerConfig
from models.detector.postprocess import DigitDetector, PostProcessConfig
from models.graph.builder import ThresholdRadiusGraphBuilder
from train.trainer import Trainer, TrainerConfig
from train.optimizer import OptimizerPackage
from train.losses import LossPackage, LossWeights
from eval.metrics import compute_all_metrics, ood_generalization_analysis
from utils.viz import VisualizationSuite, denormalize_image
from utils.logger import TrainingLogger, LoggerConfig

print("All imports successful")

# %% [markdown]
# ## 2. Configuration

# %%
# Training configuration (optimized for Colab T4/A100)
dataset_config = DatasetConfig(
    min_digits=5,
    max_digits=50,
    img_width=640,
    img_height=640,
    threshold_radius_r=80.0,
    noise_sigma=3.0,
    max_chunk_size=4,
    min_chunk_size=1,
    samples_per_epoch=500,
    seed=42,
)

trainer_config = TrainerConfig(
    learning_rate=1e-4,
    backbone_lr=1e-5,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    num_epochs=50,
    warmup_epochs=3,
    scheduler='cosine',
    batch_size=4,
    num_workers=2,
    heatmap_loss_weight=1.0,
    digit_loss_weight=1.0,
    action_loss_weight=1.0,
    jump_loss_weight=1.0,
    val_every_n_epochs=1,
    ood_eval_every_n_epochs=10,
    ood_eval_lengths=[100, 200],
    ood_eval_samples=10,
    checkpoint_dir=f'{PROJECT_DIR}/checkpoints',
    metrics_dir=f'{PROJECT_DIR}/metrics',
    save_every_n_epochs=10,
    log_every_n_steps=25,
    use_amp=torch.cuda.is_available(),
    seed=42,
)

print(f"Dataset: {dataset_config.min_digits}-{dataset_config.max_digits} digits, "
      f"{dataset_config.samples_per_epoch} samples/epoch")
print(f"Training: {trainer_config.num_epochs} epochs, batch={trainer_config.batch_size}, "
      f"lr={trainer_config.learning_rate}")
print(f"AMP: {trainer_config.use_amp}")

# %% [markdown]
# ## 3. Data Preview

# %%
# Generate and visualize a sample
import matplotlib.pyplot as plt
import numpy as np

gen_config = GeneratorConfig(
    img_width=640, img_height=640,
    threshold_radius_r=80.0, noise_sigma=3.0,
    max_chunk_size=4,
)
generator = ConstrainedPolarGenerator(gen_config)
sample = generator.generate_sample(total_digits=20)

render_config = RendererConfig(img_width=640, img_height=640, seed=42)
renderer = DigitRenderer(render_config)
render_output = renderer.render(sample)

# Display image
img_tensor = render_output['image']
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(img_display)
axes[0].set_title(f'Generated Image ({sample.total_digits} digits, {sample.num_chunks} chunks)')
axes[0].axis('off')

# Draw bounding boxes
for node in sample.nodes:
    x, y = node.center_x, node.center_y
    w, h = node.width, node.height
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                          linewidth=1, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x, y - h/2 - 3, node.label, color='red', fontsize=8, ha='center')

# Display heatmap target
hm = render_output['heatmap_target'].squeeze().numpy()
axes[1].imshow(hm, cmap='hot')
axes[1].set_title('Heatmap Target (digit centers)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/sample_preview.png', dpi=150, bbox_inches='tight')
plt.show()

# Print ground truth sequence
gt_tokens = [t['token'] for t in sample.gt_sequence]
print(f"GT sequence: {' '.join(gt_tokens)}")
print(f"Nodes: {sample.total_digits}, Chunks: {sample.num_chunks}")

# %% [markdown]
# ## 4. Detector Pre-Training
#
# Pre-train the heatmap head so the controller receives reasonable
# bounding boxes from epoch 0.

# %%
# Create a simple dataset for detector pre-training
from torch.utils.data import Dataset, DataLoader


class DetectorDataset(Dataset):
    """Simple dataset that yields images + heatmap targets."""

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
        total_digits = self.rng.randint(5, 51)
        layout = self.generator.generate_sample(total_digits)
        self.renderer.rng = np.random.RandomState(idx)
        render_out = self.renderer.render(layout)
        return {
            'image': render_out['image'],
            'heatmap_target': render_out['heatmap_target'],
        }


# Create dataloaders
det_train_dataset = DetectorDataset(500, dataset_config)
det_val_dataset   = DetectorDataset(50,  dataset_config)

det_train_loader = DataLoader(det_train_dataset, batch_size=8, shuffle=True,  num_workers=2)
det_val_loader   = DataLoader(det_val_dataset,   batch_size=8, shuffle=False, num_workers=2)

print(f"Detector train: {len(det_train_dataset)} samples")
print(f"Detector val:   {len(det_val_dataset)} samples")

# %%
# Pre-train detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone_for_detector = VisualBackbone(
    vis_dim=512, roi_output_size=7,
    pretrained=True, enable_heatmap=False,
).to(device)

det_trainer_config = DetectorTrainerConfig(
    num_epochs=10,
    learning_rate=1e-4,
    backbone_lr=1e-5,
    batch_size=8,
    freeze_backbone_epochs=3,
    checkpoint_dir=f'{PROJECT_DIR}/checkpoints/detector',
    device=str(device),
)

det_trainer = DetectorTrainer(
    config=det_trainer_config,
    backbone=backbone_for_detector,
    in_channels=512,
    hidden_channels=128,
    stride=8,
)

# Run pre-training
det_trainer.fit(det_train_loader, det_val_loader)

print("\nDetector pre-training complete")

# %% [markdown]
# ## 5. Joint Training
#
# Train the full model: backbone + heatmap head + controller.

# %%
# Initialize the full trainer
trainer = Trainer(trainer_config, dataset_config)

# Load pre-trained backbone weights
det_trainer.load_into_backbone(trainer.backbone)
print("Pre-trained backbone loaded into full model")

# %%
# ============================================================
# AUTO-RESUME: Continue from last checkpoint if interrupted
# ============================================================
import glob

ckpt_dir = f'{PROJECT_DIR}/checkpoints'
resume_path = None

if os.path.isdir(ckpt_dir):
    periodic = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
    if periodic:
        periodic.sort(key=lambda p: int(os.path.basename(p)
                      .replace('checkpoint_epoch_', '')
                      .replace('.pt', '')))
        resume_path = periodic[-1]
    else:
        best = os.path.join(ckpt_dir, 'checkpoint_best.pt')
        if os.path.exists(best):
            resume_path = best

if resume_path:
    trainer.resume_from_checkpoint(resume_path)
    print(f"\nResuming from: {os.path.basename(resume_path)}")
    print(f"  Starting at epoch {trainer.current_epoch}/{trainer_config.num_epochs}")
else:
    print("\nNo checkpoint found. Starting from epoch 0.")

# %%
# Run training
trainer.fit()

print("\nJoint training complete")

# %% [markdown]
# ## 6. Training Curves

# %%
# Plot training curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

loss_keys = ['total', 'heatmap', 'digit', 'action', 'jump']
titles    = ['Total Loss', 'Heatmap Loss', 'Digit Loss', 'Action Loss', 'Jump Loss']

for idx, (key, title) in enumerate(zip(loss_keys, titles)):
    ax = axes[idx // 3][idx % 3]
    train_vals = [h.get(key, 0) for h in trainer.train_history]
    ax.plot(train_vals, label='Train', color='blue', alpha=0.7)

    if trainer.val_history:
        val_vals = [h.get(key, 0) for h in trainer.val_history]
        val_x = np.linspace(0, len(train_vals) - 1, len(val_vals))
        ax.plot(val_x, val_vals, label='Val', color='red', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1][2].axis('off')

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. OOD Evaluation

# %%
# Run OOD evaluation
from evaluate import load_model, evaluate_length

model = load_model(
    checkpoint_path=f'{PROJECT_DIR}/checkpoints/checkpoint_best.pt',
    device=device,
    radius=80.0,
    noise_sigma=3.0,
    r_infer_multiplier=1.2,
)

eval_lengths = [50, 100, 200, 500]
results_by_length = {}

for length in eval_lengths:
    print(f"\nEvaluating length={length}...")
    result = evaluate_length(
        length=length,
        num_samples=20,
        model=model,
        device=device,
        img_size=640,
        radius=80.0,
        noise_sigma=3.0,
        max_chunk_size=4,
        base_seed=9999,
        max_steps_multiplier=3,
        quiet=False,
    )
    results_by_length[length] = result['summary']

# %%
# OOD analysis and plot
ood_analysis = ood_generalization_analysis(results_by_length)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

lengths    = sorted(results_by_length.keys())
seq_accs   = [results_by_length[l].get('exact_match',    0) for l in lengths]
digit_accs = [results_by_length[l].get('digit_accuracy', 0) for l in lengths]
chunk_f1s  = [results_by_length[l].get('chunk_f1',       0) for l in lengths]

x     = np.arange(len(lengths))
width = 0.25

ax.bar(x - width, seq_accs,   width, label='Exact Match',    color='steelblue')
ax.bar(x,         digit_accs, width, label='Digit Accuracy',  color='coral')
ax.bar(x + width, chunk_f1s,  width, label='Chunk F1',        color='seagreen')

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Score')
ax.set_title('OOD Length Generalization')
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in lengths])
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

if ood_analysis['critical_length']:
    ax.axvline(
        x=lengths.index(ood_analysis['critical_length']),
        color='red', linestyle='--', alpha=0.7,
        label=f"Critical length: {ood_analysis['critical_length']}",
    )
    ax.legend()

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/ood_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nOOD Analysis:")
print(f"  Critical length:  {ood_analysis['critical_length']}")
print(f"  Seq degradation:  {ood_analysis['seq_degradation_rate']:.4f}")
print(f"  Accuracy at max:  {ood_analysis['accuracy_at_max_length']:.4f}")

# %% [markdown]
# ## 8. Visualization

# %%
# Visualize a sample prediction
from utils.viz import draw_graph, draw_reading_path
from evaluate import generate_eval_sample, evaluate_single_sample

test_sample = generate_eval_sample(
    total_digits=30, img_size=640, radius=80.0,
    noise_sigma=3.0, max_chunk_size=4, seed=12345,
)

result = evaluate_single_sample(
    sample=test_sample, model=model, device=device,
    max_steps=100, greedy=True,
)

base_img = denormalize_image(test_sample['image'])

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

axes[0].imshow(np.array(base_img))
axes[0].set_title('Input Image')
axes[0].axis('off')

graph_img = draw_graph(
    image=base_img,
    node_positions_px=test_sample['graph'].node_positions_px,
    adjacency=test_sample['graph'].adjacency,
    node_chunk_ids=test_sample['graph'].node_chunk_ids,
    radius=80.0,
    node_labels=test_sample['graph'].node_labels,
)
axes[1].imshow(np.array(graph_img))
axes[1].set_title('Spatial Graph')
axes[1].axis('off')

path_img = draw_reading_path(
    image=base_img,
    output_tokens=result['output_tokens'],
    node_positions_px=test_sample['graph'].node_positions_px,
    node_chunk_ids=test_sample['graph'].node_chunk_ids,
)
axes[2].imshow(np.array(path_img))
axes[2].set_title('Reading Path')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/prediction_viz.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"GT:   {result['gt_string']}")
print(f"PRED: {result['pred_string']}")
print(f"Exact match: {result['metrics']['exact_match']}")
print(f"Digit acc:   {result['metrics']['digit_accuracy']:.4f}")
print(f"Chunk F1:    {result['metrics']['chunk_f1']:.4f}")

# %% [markdown]
# ## 9. Save Results

# %%
# Save all results to Drive
import json

results_data = {
    'training': {
        'config': {
            'num_epochs':    trainer_config.num_epochs,
            'batch_size':    trainer_config.batch_size,
            'learning_rate': trainer_config.learning_rate,
        },
        'final_train_loss': trainer.train_history[-1] if trainer.train_history else {},
        'best_val_loss':    trainer.ckpt_mgr.best_val_loss,
    },
    'ood_evaluation': {
        str(k): v for k, v in results_by_length.items()
    },
    'ood_analysis': ood_analysis,
}

results_path = f'{PROJECT_DIR}/eval_results.json'
with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2, default=str)

print(f"Results saved to:        {results_path}")
print(f"Checkpoints saved to:    {PROJECT_DIR}/checkpoints/")
print(f"Metrics saved to:        {PROJECT_DIR}/metrics/")
print(f"Visualizations saved to: {PROJECT_DIR}/")

# %% [markdown]
# ## 10. Download Checkpoint (Optional)

# %%
# Download the best checkpoint to local machine
from google.colab import files

best_ckpt = f'{PROJECT_DIR}/checkpoints/checkpoint_best.pt'
if os.path.exists(best_ckpt):
    print(f"Downloading: {best_ckpt}")
    files.download(best_ckpt)
else:
    print("No checkpoint found.")

# %% [markdown]
# ---
# **Training complete.** The model checkpoint is saved on your Google Drive at:
# `MyDrive/cognitive_reader/checkpoints/checkpoint_best.pt`
#
# To run inference locally:
# ```python
# from eval.inference import load_inference_pipeline
# pipeline = load_inference_pipeline('checkpoint_best.pt')
# result = pipeline.run(image_tensor)
# print(result.predicted_string)
# ```