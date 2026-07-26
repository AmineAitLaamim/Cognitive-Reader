# %% [markdown]
# # Cognitive Reader — Fine-Tune with Scheduled Sampling
#
# Fine-tunes from the epoch-20 checkpoint (best chunking: pure_frac ~0.75)
# using scheduled sampling to combat exposure bias.
#
# The model's chunking degraded from ep21 to ep49 because it was trained
# purely on the GT trajectory (teacher forcing) but deployed on its own
# divergent trajectory at inference. Scheduled sampling randomly uses the
# model's own READ/CHUNK prediction to update h_content during training,
# teaching it to handle its own divergent recurrent state.
#
# Pipeline:
# 1. Environment setup (clone-or-PULL, imports)
# 2. Configuration (fine-tune LR, scheduled sampling 0 -> 0.5 over 20 epochs)
# 3. GATE 1 (verify generator)
# 4. Load detector from Drive (skip 9-min pretrain)
# 5. Load ep20 checkpoint + rebuild optimizer/scheduler for fine-tune
# 6. Fine-tune (30 epochs)
# 7. Training curves
# 8. OOD evaluation at 30/40/50/60
# 9. Contiguity probe (pure_frac / intra_changes / cross_repeats)
# 10. Save to Drive
#
# MANGLE-PROOF: no leading underscores, no dunders, no triple-quoted strings,
# no underscore before * or ' in code cells.

# %% [markdown]
# ## 1. Environment Setup

# %%
import torch
try:
    import importlib.metadata as imd
    torchver = imd.version('torch')
except Exception:
    torchver = 'unknown'
print('PyTorch version:', torchver)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print('WARNING: No GPU. Runtime -> Change runtime type -> GPU')

# %%
!pip install -q torch torchvision
!pip install -q Pillow matplotlib tensorboard pyyaml tqdm
!pip install -q jupytext

# %%
import os
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/Cognitive-Reader'
DRIVE_DIR = '/content/drive/MyDrive/cognitive_reader'
RUN_DIR = '/content/run'

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
print('Local run dir:', RUN_DIR)
print('Drive dir:', DRIVE_DIR)

# %%
repo = 'https://github.com/AmineAitLaamim/Cognitive-Reader'
if os.path.isdir(os.path.join(PROJECT_DIR, '.git')):
    print('repo present -> pulling latest')
    os.system('git -C %s pull' % PROJECT_DIR)
else:
    print('first run -> cloning repo')
    os.system('git clone -q %s %s' % (repo, PROJECT_DIR))

os.chdir(PROJECT_DIR)
print('Working directory:', os.getcwd())
required = ['data', 'models', 'train', 'eval', 'utils']
missing = [d for d in required if not os.path.isdir(d)]
print('Missing dirs:', missing if missing else 'none')

# %%
import sys
sys.path.insert(0, PROJECT_DIR)

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
print('All imports successful')

# %% [markdown]
# ## 2. Configuration (fine-tune specific)
#
# Key differences from the training notebook:
# - num_epochs=30 (fine-tune for 30 more epochs)
# - learning_rate=3e-5 (lower than training's 1e-4)
# - backbone_lr=3e-6 (lower than training's 1e-5)
# - scheduled_sampling_max=0.5 (grows from 0 to 0.5 over 20 epochs)
# - checkpoint_dir on Drive in a SEPARATE folder (checkpoints_finetune)

# %%
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
    learning_rate=3e-5,
    backbone_lr=3e-6,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    num_epochs=30,
    warmup_epochs=3,
    scheduler='cosine',
    batch_size=4,
    num_workers=2,
    heatmap_loss_weight=1.0,
    digit_loss_weight=1.0,
    action_loss_weight=1.0,
    jump_loss_weight=1.0,
    scheduled_sampling_max=0.5,
    scheduled_sampling_ramp_epochs=20,
    val_every_n_epochs=1,
    ood_eval_every_n_epochs=10,
    ood_eval_lengths=[60],
    ood_eval_samples=10,
    checkpoint_dir=DRIVE_DIR + '/checkpoints_finetune',
    metrics_dir=DRIVE_DIR + '/metrics_finetune',
    log_dir=RUN_DIR + '/logs_finetune',
    save_every_n_epochs=10,
    log_every_n_steps=25,
    use_amp=torch.cuda.is_available(),
    seed=42,
)

print('Fine-tune config:')
print('  Epochs: %d' % trainer_config.num_epochs)
print('  LR: %s (backbone: %s)' % (trainer_config.learning_rate, trainer_config.backbone_lr))
print('  Scheduled sampling: 0 -> %.1f over %d epochs' % (
    trainer_config.scheduled_sampling_max, trainer_config.scheduled_sampling_ramp_epochs))
print('  Checkpoints -> Drive:', trainer_config.checkpoint_dir)

# %% [markdown]
# ## 3. GATE 1 (verify generator)

# %%
gen_config = GeneratorConfig(
    img_width=640, img_height=640,
    threshold_radius_r=80.0, noise_sigma=3.0, max_chunk_size=4,
)
generator = ConstrainedPolarGenerator(gen_config)
sample = generator.generate_sample(total_digits=20)

assert sample.total_digits == 20, (
    'GATE 1 FAIL: generator returned %d nodes for a request of 20. '
    'Re-run clone-or-pull, then Restart runtime.' % sample.total_digits
)
print('GATE 1 OK: Nodes = %d' % sample.total_digits)

# %% [markdown]
# ## 4. Load detector from Drive (skip pretrain)

# %%
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone_for_detector = VisualBackbone(
    vis_dim=512, roi_output_size=7, pretrained=True, enable_heatmap=False,
).to(device)

det_trainer_config = DetectorTrainerConfig(
    num_epochs=10,
    learning_rate=1e-4,
    backbone_lr=1e-5,
    batch_size=8,
    freeze_backbone_epochs=3,
    checkpoint_dir=DRIVE_DIR + '/checkpoints/detector',
    device=str(device),
)

det_trainer = DetectorTrainer(
    config=det_trainer_config,
    backbone=backbone_for_detector,
    in_channels=512,
    hidden_channels=128,
    stride=8,
)

det_ckpts = sorted(glob.glob(DRIVE_DIR + '/checkpoints/detector/detector*.pt'))
if det_ckpts:
    det_path = det_ckpts[-1]
    det_trainer.load_checkpoint(det_path)
    print('Detector loaded from Drive:', det_path)
    print('Skipping pretrain (saved ~9 min)')
else:
    print('WARNING: No detector checkpoint on Drive. Pretraining from scratch.')
    from data.detector_dataset import make_det_loaders
    det_train_loader, det_val_loader = make_det_loaders(dataset_config)
    det_trainer.fit(det_train_loader, det_val_loader)
    print('Detector pre-training complete')

print('Detector ready. Best val loss:', det_trainer.best_val_loss)

# %% [markdown]
# ## 5. Load ep20 checkpoint + rebuild optimizer for fine-tune
#
# We load ONLY the model weights (backbone + controller) from the ep20
# checkpoint. The optimizer and scheduler are rebuilt fresh with the
# fine-tune LR (3e-5 / 3e-6) and cosine schedule for 30 epochs.
# The epoch counter is reset to 0 so the scheduled sampling ramp
# (0 -> 0.5 over 20 epochs) starts from the beginning of the fine-tune.

# %%
trainer = Trainer(trainer_config, dataset_config)
det_trainer.load_into_backbone(trainer.backbone)
print('Pre-trained detector backbone loaded')

# Load the ep20 checkpoint (model weights only)
ep20_path = DRIVE_DIR + '/checkpoints/checkpoint_epoch_20.pt'
if not os.path.exists(ep20_path):
    # Try alternate names
    alt_paths = [
        DRIVE_DIR + '/checkpoints/checkpoint_epoch_19.pt',
        DRIVE_DIR + '/checkpoints/checkpoint_best.pt',
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            ep20_path = alt
            break

assert os.path.exists(ep20_path), (
    'Cannot find ep20 checkpoint. Looked for:\n'
    '  %s\n'
    'Make sure the checkpoint is on Drive.' % ep20_path
)

ckpt = torch.load(ep20_path, map_location=device)
trainer.backbone.load_state_dict(ckpt['backbone_state_dict'])
trainer.controller.load_state_dict(ckpt['controller_state_dict'])
print('Loaded model weights from:', ep20_path)
print('  Checkpoint epoch:', ckpt.get('epoch', 'unknown'))

# Reset epoch counter so scheduled sampling ramps from 0
trainer.current_epoch = 0
trainer.global_step = 0
trainer.train_history = []
trainer.val_history = []

# Rebuild optimizer with fine-tune LR
trainer.optimizer = torch.optim.AdamW([
    {'params': trainer.backbone.parameters(), 'lr': trainer_config.backbone_lr, 'weight_decay': trainer_config.weight_decay},
    {'params': trainer.controller.parameters(), 'lr': trainer_config.learning_rate, 'weight_decay': trainer_config.weight_decay},
])

# Rebuild scheduler for 30 epochs
trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    trainer.optimizer,
    T_max=trainer_config.num_epochs - trainer_config.warmup_epochs,
    eta_min=1e-7
)

# Rebuild scaler
from torch.cuda.amp import GradScaler
trainer.scaler = GradScaler(enabled=trainer_config.use_amp)

print('Optimizer rebuilt: LR=%s, backbone_LR=%s' % (
    trainer_config.learning_rate, trainer_config.backbone_lr))
print('Scheduler rebuilt: T_max=%d' % (
    trainer_config.num_epochs - trainer_config.warmup_epochs))
print('Fine-tune ready: %d epochs, scheduled sampling 0 -> %.1f' % (
    trainer_config.num_epochs, trainer_config.scheduled_sampling_max))

# %% [markdown]
# ## 6. Fine-tune

# %%
trainer.fit()
print('Fine-tuning complete')
print('Checkpoints saved to Drive:', trainer_config.checkpoint_dir)

# %% [markdown]
# ## 7. Training Curves

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
loss_keys = ['total', 'heatmap', 'digit', 'action', 'jump']
titles = ['Total Loss', 'Heatmap Loss', 'Digit Loss', 'Action Loss', 'Jump Loss']
for idx, pair in enumerate(zip(loss_keys, titles)):
    key, title = pair
    ax = axes[idx // 3][idx % 3]
    train_vals = [h.get(key, 0) for h in trainer.train_history]
    ax.plot(train_vals, label='Train', color='blue', alpha=0.7)
    if trainer.val_history:
        val_vals = [h.get(key, 0) for h in trainer.val_history]
        val_x = np.linspace(0, len(train_vals) - 1, len(val_vals))
        ax.plot(val_x, val_vals, label='Val', color='red', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Fine-tune Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/finetune_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. OOD Evaluation

# %%
from evaluate import load_model, evaluate_length

model = load_model(
    checkpoint_path=DRIVE_DIR + '/checkpoints_finetune/checkpoint_best.pt',
    device=device, radius=80.0, noise_sigma=3.0, r_infer_multiplier=1.2,
)

eval_lengths = [30, 40, 50, 60]
results_by_length = {}
for length in eval_lengths:
    print('Evaluating length=%d...' % length)
    result = evaluate_length(
        length=length, num_samples=20, model=model, device=device,
        img_size=640, radius=80.0, noise_sigma=3.0, max_chunk_size=4,
        base_seed=9999, max_steps_multiplier=3, quiet=False,
    )
    results_by_length[length] = result['summary']

# %%
ood_analysis = ood_generalization_analysis(results_by_length)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
lengths = sorted(results_by_length.keys())
seq_accs = [results_by_length[l].get('exact_match', 0) for l in lengths]
digit_accs = [results_by_length[l].get('digit_accuracy', 0) for l in lengths]
chunk_f1s = [results_by_length[l].get('chunk_f1', 0) for l in lengths]
x = np.arange(len(lengths))
width = 0.25
ax.bar(x - width, seq_accs, width, label='Exact Match', color='steelblue')
ax.bar(x, digit_accs, width, label='Digit Accuracy', color='coral')
ax.bar(x + width, chunk_f1s, width, label='Chunk F1', color='seagreen')
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Score')
ax.set_title('OOD After Fine-Tune (Scheduled Sampling)')
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in lengths])
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/finetune_ood.png', dpi=150, bbox_inches='tight')
plt.show()
print('OOD Analysis:')
print('  Critical length: ', ood_analysis['critical_length'])
print('  Seq degradation:  %.4f' % ood_analysis['seq_degradation_rate'])
print('  Accuracy at max:  %.4f' % ood_analysis['accuracy_at_max_length'])

# %% [markdown]
# ## 9. Contiguity Probe
#
# This is the KEY evaluation. It measures whether the model reads chunks
# contiguously (pure_frac), how often it crosses chunk boundaries within
# a segment (intra_changes), and how often it revisits chunks (cross_repeats).
#
# Compare to the pre-fine-tune results:
#   ep20: pure_frac=0.75/0.60, intra_changes=5.2/16.6, cross_repeats=4.0/7.8
#   ep49: pure_frac=0.66/0.52, intra_changes=6.8/19.0, cross_repeats=3.2/6.4
#
# Target: pure_frac >= 0.8 at length 60.

# %%
from evaluate import generate_eval_sample

ctrl = model['controller']
bb = model['backbone']
ctrl.eval(); bb.eval()
dev = next(bb.parameters()).device

def probe(seed, N):
    s = generate_eval_sample(total_digits=N, img_size=640, radius=80.0,
                             noise_sigma=3.0, max_chunk_size=4, seed=seed)
    with torch.no_grad():
        bout = bb(s['image'].unsqueeze(0).to(dev), s['boxes'].to(dev))
    g = s['graph']; g.node_embeddings = bout['node_embeddings']; g = g.to(dev)
    with torch.no_grad():
        inf = ctrl.forward_inference(graph=g, cls_token=bout['cls_token'].squeeze(0),
                                     device=dev, max_steps=N*4, greedy=True)
    cids = g.node_chunk_ids.cpu().numpy(); Nn = g.num_nodes
    segs, cur = [], []
    for t in inf.state.output_tokens:
        if t.get('mode') == 'READ' and t.get('node_id') is not None: cur.append(t['node_id'])
        elif t.get('token') == '<CHUNK>': segs.append(cur); cur = []
    if cur: segs.append(cur)
    segs = [seg for seg in segs if seg]
    read_order = [n for seg in segs for n in seg]
    complete = (len(set(read_order)) == Nn and len(read_order) == Nn)
    gt = {i: int(g.node_labels[i]) for i in range(Nn)}
    dc = dt = 0
    for t in inf.state.output_tokens:
        if t.get('mode') == 'READ' and t.get('node_id') is not None:
            nid = t['node_id']; gv = gt.get(nid)
            if gv is not None: dt += 1; dc += int(int(t['token']) == gv)
    pure = intra = 0
    for seg in segs:
        ids = [int(cids[n]) for n in seg]
        ch = sum(1 for i in range(1, len(ids)) if ids[i] != ids[i-1])
        intra += ch; pure += (ch == 0)
    seen, order, repeats = set(), [], 0
    for seg in segs:
        c = int(cids[seg[0]])
        if order and order[-1] != c and c in seen: repeats += 1
        order.append(c); seen.add(c)
    return dict(complete=complete, pernode_digit=(dc/max(dt,1)),
                pure_frac=(pure/max(len(segs),1)), intra_changes=intra,
                cross_repeats=repeats)

print('=' * 60)
print('  CONTINUITY PROBE (fine-tuned model)')
print('=' * 60)
for N in (30, 60):
    rows = [probe(sd, N) for sd in (1,2,3,4,5)]
    avg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print('--- length %d ---' % N)
    print('  complete:', '%.2f' % avg['complete'])
    print('  per-node digit acc:', '%.2f' % avg['pernode_digit'])
    print('  pure_frac:', '%.2f' % avg['pure_frac'],
          ' (ep20: 0.75/0.60, ep49: 0.66/0.52, target: >=0.80)')
    print('  intra_changes:', '%.1f' % avg['intra_changes'],
          ' (ep20: 5.2/16.6, ep49: 6.8/19.0, target: <=2)')
    print('  cross_repeats:', '%.1f' % avg['cross_repeats'],
          ' (ep20: 4.0/7.8, ep49: 3.2/6.4, target: <=1)')

# %% [markdown]
# ## 10. Save to Drive

# %%
import json
results_data = {
    'finetune': {
        'base_checkpoint': ep20_path,
        'num_epochs': trainer_config.num_epochs,
        'learning_rate': trainer_config.learning_rate,
        'backbone_lr': trainer_config.backbone_lr,
        'scheduled_sampling_max': trainer_config.scheduled_sampling_max,
        'scheduled_sampling_ramp_epochs': trainer_config.scheduled_sampling_ramp_epochs,
        'best_val_loss': trainer.ckpt_mgr.best_val_loss,
    },
    'ood_evaluation': {str(k): v for k, v in results_by_length.items()},
    'ood_analysis': ood_analysis,
}
results_path = DRIVE_DIR + '/finetune_results.json'
with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2, default=str)
print('Fine-tune results saved to Drive:', results_path)
print('Checkpoints on Drive:', DRIVE_DIR + '/checkpoints_finetune/')

# %% [markdown]
# ---
# Decision rule after fine-tune:
#   pure_frac >= 0.8 at length 60
#       -> exposure bias fixed; architecture proven; Plan B (bigger canvas) justified.
#   pure_frac 0.6-0.8 at length 60
#       -> scheduled sampling helped but not enough; try higher sampling_max (0.7)
#          or longer ramp, or combine with action_loss_weight=2.0.
#   pure_frac < 0.6 at length 60 (no improvement over ep20)
#       -> scheduled sampling alone is insufficient; the action head needs a
#          structural fix (e.g., explicit chunk-boundary feature, or a separate
#          chunk-boundary classifier trained on the model's own trajectory).

# %%
from google.colab import files
best_ckpt = DRIVE_DIR + '/checkpoints_finetune/checkpoint_best.pt'
if os.path.exists(best_ckpt):
    print('Downloading:', best_ckpt)
    files.download(best_ckpt)
else:
    print('No fine-tune checkpoint found.')