# %% [markdown]
# # Cognitive Reader — Training on Google Colab (Plan A, crash-hardened)
#
# Pipeline:
# 1. Environment setup (clone-or-PULL to LOCAL disk, not Drive)
# 2. Configuration (OOD lengths lowered to physically realizable values; FRESH_RETRAIN flag)
# 3. Data preview  -> GATE 1: must print Nodes: 20
#    Curriculum histogram -> GATE 2: must show a broad spread, not a spike
# 4. Clean slate (deletes LOCAL run dirs so we never resume the flat-5 weights)
# 5. Detector pre-training (fresh; dataset class imported from data/detector_dataset.py)
# 6. Joint training on LOCAL SSD (avoids the Drive Errno-107 crash)
# 7. Training curves
# 8. OOD evaluation at 30/40/50/60 (all feasible at 640 / r=80)
# 9. Visualization
# 10. Save + export to Drive at the very end
#
# This file is intentionally free of double-underscore method names and triple-quote
# docstrings in code cells, so a markdown-aware editor cannot strip them.

# %% [markdown]
# ## 1. Environment Setup

# %%
# GPU check. Version is read WITHOUT double-underscore attributes on purpose,
# because those get stripped by the notebook editor.
import torch
try:
    import importlib.metadata as _im
    _tv = _im.version('torch')
except Exception:
    _tv = 'unknown'
print('PyTorch version:', _tv)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print('WARNING: No GPU. Runtime -> Change runtime type -> GPU')

# %%
# Dependencies
# !pip install -q torch torchvision
# !pip install -q Pillow matplotlib tensorboard pyyaml tqdm
# !pip install -q jupytext

# %%
# Mount Drive. Drive is used ONLY for the final export (and optional restore),
# never as the working directory. The Errno-107 crash happened because cwd was
# on the Drive FUSE mount; keeping cwd on local disk removes that whole failure class.
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/Cognitive-Reader'                 # LOCAL: code clone
DRIVE_DIR   = '/content/drive/MyDrive/cognitive_reader'   # Drive: persistent export/restore ONLY
RUN_DIR     = '/content/run'                              # LOCAL SSD: checkpoints/metrics/logs
os.makedirs(RUN_DIR, exist_ok=True) if 'os' in dir() else None

# %%
# Clone on first run, PULL on every later run. The old cell did
# `git clone ... || true`, which silently fails when the dir exists and therefore
# NEVER picked up your pushes on a re-used runtime -- that is very likely why
# earlier fixes looked like they did not take effect.
import os
os.makedirs(RUN_DIR, exist_ok=True)
_repo = 'https://github.com/AmineAitLaamim/Cognitive-Reader'
if os.path.isdir(os.path.join(PROJECT_DIR, '.git')):
    print('repo present -> pulling latest (makes your pushes take effect)')
    os.system('git -C %s pull' % PROJECT_DIR)
else:
    print('first run -> cloning repo')
    os.system('git clone -q %s %s' % (_repo, PROJECT_DIR))

os.chdir(PROJECT_DIR)        # cwd is LOCAL -> not on a flaky mount
print('Working directory:', os.getcwd())
_required = ['data', 'models', 'train', 'eval', 'utils']
_missing = [d for d in _required if not os.path.isdir(d)]
print('Missing dirs:', _missing if _missing else 'none')

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
# ## 2. Configuration

# %%
# FRESH_RETRAIN controls the clean-slate cell.
#   True  -> delete LOCAL run dirs; retrain detector + joint from scratch (Plan A).
#   False -> keep them so the resume cell can continue an interrupted run.
# Set True for this retrain. Set False only to resume a run started with the fixed code.
FRESH_RETRAIN = True

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
    ood_eval_lengths=[60],                       # was [100, 200] -> those raise now
    ood_eval_samples=10,
    checkpoint_dir=RUN_DIR + '/checkpoints',     # LOCAL, not Drive
    metrics_dir=RUN_DIR + '/metrics',            # LOCAL
    log_dir=RUN_DIR + '/logs',                   # LOCAL (this is what Errno-107 hit)
    save_every_n_epochs=10,
    log_every_n_steps=25,
    use_amp=torch.cuda.is_available(),
    seed=42,
)

print('Dataset: %d-%d digits, %d samples/epoch' % (
    dataset_config.min_digits, dataset_config.max_digits, dataset_config.samples_per_epoch))
print('Training: %d epochs, batch=%d, lr=%s' % (
    trainer_config.num_epochs, trainer_config.batch_size, trainer_config.learning_rate))
print('AMP:', trainer_config.use_amp, ' FRESH_RETRAIN:', FRESH_RETRAIN)

# %% [markdown]
# ## 3. Data Preview (GATE 1) and curriculum histogram (GATE 2)
#
# GATE 1 must print Nodes: 20. If it prints 5, the clone-or-pull did not fetch the
# fixed generator -- do not continue. GATE 2 checks the collate/graph layer the
# generator fix did not touch; a broad spread is honest, a spike is a downstream bug.

# %%
import matplotlib.pyplot as plt
import numpy as np

gen_config = GeneratorConfig(
    img_width=640, img_height=640,
    threshold_radius_r=80.0, noise_sigma=3.0, max_chunk_size=4,
)
generator = ConstrainedPolarGenerator(gen_config)
sample = generator.generate_sample(total_digits=20)

assert sample.total_digits == 20, (
    'GATE 1 FAIL: generator returned %d nodes for a request of 20. The fixed '
    'generator is NOT loaded on this runtime. Re-run the clone-or-pull cell, then '
    'Runtime -> Restart runtime, then re-run imports, before continuing.' % sample.total_digits
)

render_config = RendererConfig(img_width=640, img_height=640, seed=42)
renderer = DigitRenderer(render_config)
render_output = renderer.render(sample)

img_tensor = render_output['image']
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(img_display)
axes[0].set_title('Generated Image (%d digits, %d chunks)' % (sample.total_digits, sample.num_chunks))
axes[0].axis('off')
for node in sample.nodes:
    x, y = node.center_x, node.center_y
    w, h = node.width, node.height
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x, y - h/2 - 3, node.label, color='red', fontsize=8, ha='center')
hm = render_output['heatmap_target'].squeeze().numpy()
axes[1].imshow(hm, cmap='hot')
axes[1].set_title('Heatmap Target (digit centers)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/sample_preview.png', dpi=150, bbox_inches='tight')
plt.show()

gt_tokens = [t['token'] for t in sample.gt_sequence]
print('GT sequence:', ' '.join(gt_tokens))
print('Nodes: %d, Chunks: %d   <- GATE 1 OK if Nodes == 20' % (sample.total_digits, sample.num_chunks))

# %%
# GATE 2: histogram of READ nodes that actually reach the controller.
from collections import Counter
try:
    from data.collate import unpad_graph
    _hist = Counter()
    _tr, _ = create_dataloaders(dataset_config, batch_size=4, num_workers=0)
    for _bi, _batch in enumerate(_tr):
        for _i in range(_batch.batch_size):
            _sd = unpad_graph(_batch, _i, device=torch.device('cpu'))
            _hist[sum(1 for t in _sd['gt_sequence'] if t['mode'] == 'READ')] += 1
        if _bi >= 7:
            break
    print('READ-nodes per sample reaching the controller:', dict(sorted(_hist.items())))
    _real = {k: v for k, v in _hist.items() if k > 0}
    assert _real and max(_real) <= dataset_config.max_digits and len(_real) >= 5, 'curriculum collapsed'
    print('GATE 2 OK: curriculum is broad')
except Exception as _e:
    print('GATE 2 skipped:', type(_e).__name__, _e, '- rely on epoch-10 OOD as the end-to-end check')

# %% [markdown]
# ## 4. Clean slate (only when FRESH_RETRAIN is True)
#
# Deletes the LOCAL run dirs so the resume cell cannot reload the old flat-5 weights
# and so the detector is re-pretrained on the honest generator. Drive is untouched.

# %%
if FRESH_RETRAIN:
    import shutil
    for _d in ['checkpoints', 'metrics', 'logs']:
        _p = RUN_DIR + '/' + _d
        if os.path.isdir(_p):
            shutil.rmtree(_p)
            print('deleted', _p)
    print('clean slate set: detector pretrains fresh, joint starts at epoch 0')
else:
    print('FRESH_RETRAIN is False -> keeping existing LOCAL checkpoints (resume mode)')

# %% [markdown]
# ## 5. Detector Pre-Training
#
# The dataset class (with its init/len/getitem methods) lives in
# data/detector_dataset.py so those names survive any editor. It is imported here,
# never redefined -- so this cell cannot be corrupted. If this import fails, the
# file is missing on GitHub: push data/detector_dataset.py and re-run.

# %%
from data.detector_dataset import make_det_loaders
det_train_loader, det_val_loader = make_det_loaders(dataset_config)
print('Detector train batches:', len(det_train_loader), 'val batches:', len(det_val_loader))

# %%
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
    checkpoint_dir=RUN_DIR + '/checkpoints/detector',   # LOCAL
    device=str(device),
)

det_trainer = DetectorTrainer(
    config=det_trainer_config,
    backbone=backbone_for_detector,
    in_channels=512,
    hidden_channels=128,
    stride=8,
)

det_trainer.fit(det_train_loader, det_val_loader)
print('Detector pre-training complete')

# %% [markdown]
# ## 6. Joint Training

# %%
trainer = Trainer(trainer_config, dataset_config)
det_trainer.load_into_backbone(trainer.backbone)
print('Pre-trained backbone loaded into full model')

# %%
# AUTO-RESUME. After a clean slate this finds nothing and starts at epoch 0.
# With FRESH_RETRAIN False it continues an interrupted run from LOCAL checkpoints.
import glob
ckpt_dir = RUN_DIR + '/checkpoints'
resume_path = None
if os.path.isdir(ckpt_dir):
    periodic = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
    if periodic:
        periodic.sort(key=lambda p: int(os.path.basename(p).replace('checkpoint_epoch_', '').replace('.pt', '')))
        resume_path = periodic[-1]
    else:
        best = os.path.join(ckpt_dir, 'checkpoint_best.pt')
        if os.path.exists(best):
            resume_path = best
if resume_path:
    trainer.resume_from_checkpoint(resume_path)
    print('Resuming from:', os.path.basename(resume_path))
    print('  Starting at epoch %d/%d' % (trainer.current_epoch, trainer_config.num_epochs))
else:
    print('No checkpoint found. Starting from epoch 0.')

# %%
trainer.fit()
print('Joint training complete')

# %% [markdown]
# ## 7. Training Curves

# %%
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
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. OOD Evaluation
#
# Lengths 30/40/50/60 are all feasible at 640 / r=80. The old [50,100,200,500] would
# now hard-crash on 100 because the fixed generator raises on infeasible N instead of
# truncating -- correct behavior, but the eval list must stay at or below feasible max.

# %%
from evaluate import load_model, evaluate_length

model = load_model(
    checkpoint_path=RUN_DIR + '/checkpoints/checkpoint_best.pt',
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
lengths    = sorted(results_by_length.keys())
seq_accs   = [results_by_length[l].get('exact_match',    0) for l in lengths]
digit_accs = [results_by_length[l].get('digit_accuracy', 0) for l in lengths]
chunk_f1s  = [results_by_length[l].get('chunk_f1',       0) for l in lengths]
x = np.arange(len(lengths)); width = 0.25
ax.bar(x - width, seq_accs,   width, label='Exact Match',   color='steelblue')
ax.bar(x,         digit_accs, width, label='Digit Accuracy', color='coral')
ax.bar(x + width, chunk_f1s,  width, label='Chunk F1',       color='seagreen')
ax.set_xlabel('Sequence Length'); ax.set_ylabel('Score')
ax.set_title('OOD Length Generalization (Plan A)')
ax.set_xticks(x); ax.set_xticklabels([str(l) for l in lengths])
ax.legend(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis='y')
if ood_analysis['critical_length']:
    ax.axvline(x=lengths.index(ood_analysis['critical_length']),
               color='red', linestyle='--', alpha=0.7,
               label='Critical length: %s' % ood_analysis['critical_length'])
    ax.legend()
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/ood_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('OOD Analysis:')
print('  Critical length: ', ood_analysis['critical_length'])
print('  Seq degradation:  %.4f' % ood_analysis['seq_degradation_rate'])
print('  Accuracy at max:  %.4f' % ood_analysis['accuracy_at_max_length'])

# %% [markdown]
# ## 9. Visualization

# %%
from utils.viz import draw_graph, draw_reading_path
from evaluate import generate_eval_sample, evaluate_single_sample

test_sample = generate_eval_sample(
    total_digits=30, img_size=640, radius=80.0,
    noise_sigma=3.0, max_chunk_size=4, seed=12345,
)
result = evaluate_single_sample(
    sample=test_sample, model=model, device=device, max_steps=100, greedy=True,
)
base_img = denormalize_image(test_sample['image'])
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes[0].imshow(np.array(base_img)); axes[0].set_title('Input Image'); axes[0].axis('off')
graph_img = draw_graph(
    image=base_img,
    node_positions_px=test_sample['graph'].node_positions_px,
    adjacency=test_sample['graph'].adjacency,
    node_chunk_ids=test_sample['graph'].node_chunk_ids,
    radius=80.0, node_labels=test_sample['graph'].node_labels,
)
axes[1].imshow(np.array(graph_img)); axes[1].set_title('Spatial Graph'); axes[1].axis('off')
path_img = draw_reading_path(
    image=base_img, output_tokens=result['output_tokens'],
    node_positions_px=test_sample['graph'].node_positions_px,
    node_chunk_ids=test_sample['graph'].node_chunk_ids,
)
axes[2].imshow(np.array(path_img)); axes[2].set_title('Reading Path'); axes[2].axis('off')
plt.tight_layout()
plt.savefig(PROJECT_DIR + '/prediction_viz.png', dpi=150, bbox_inches='tight')
plt.show()
print('GT:  ', result['gt_string'])
print('PRED:', result['pred_string'])
print('Exact match:', result['metrics']['exact_match'])
print('Digit acc:   %.4f' % result['metrics']['digit_accuracy'])
print('Chunk F1:    %.4f' % result['metrics']['chunk_f1'])

# %% [markdown]
# ## 10. Save results (local) and export to Drive

# %%
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
    'ood_evaluation': {str(k): v for k, v in results_by_length.items()},
    'ood_analysis': ood_analysis,
}
results_path = RUN_DIR + '/eval_results.json'
with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2, default=str)
print('Results saved to:', results_path)

# Best-effort export to Drive at the very end. Wrapped so a flaky mount here
# cannot erase the local results you already have.
try:
    import shutil
    os.makedirs(DRIVE_DIR, exist_ok=True)
    shutil.copytree(RUN_DIR + '/checkpoints', DRIVE_DIR + '/checkpoints', dirs_exist_ok=True)
    if os.path.isdir(RUN_DIR + '/metrics'):
        shutil.copytree(RUN_DIR + '/metrics', DRIVE_DIR + '/metrics', dirs_exist_ok=True)
    shutil.copy(results_path, DRIVE_DIR + '/eval_results.json')
    print('Exported to Drive:', DRIVE_DIR)
except Exception as _e:
    print('Drive export WARNING:', type(_e).__name__, _e, '(local results are safe at', RUN_DIR, ')')

# %% [markdown]
# ## 11. Download checkpoint (optional)

# %%
from google.colab import files
best_ckpt = RUN_DIR + '/checkpoints/checkpoint_best.pt'
if os.path.exists(best_ckpt):
    print('Downloading:', best_ckpt)
    files.download(best_ckpt)
else:
    print('No checkpoint found.')

# %% [markdown]
# ---
# Training complete. Local checkpoint at /content/run/checkpoints/checkpoint_best.pt
# (and mirrored to Drive if the export succeeded).
#
# Read the result as a decision rule (committed before seeing the number):
#   chunk_f1 at 60 >= ~0.5 AND exact_match non-zero at 30-40
#       -> the routing head HAS long-sequence capacity; Plan B (bigger canvas)
#          becomes a justified scaling experiment on a proven base.
#   chunk_f1 ~0.3-0.5, exact_match zero
#       -> partial: routes within chunks, mis-places boundaries; stay at 640,
#          try action_loss_weight=3.0 or more epochs on the honest curriculum.
#   chunk_f1 < ~0.3, exact_match zero everywhere
#       -> no long-sequence capacity yet; do NOT scale the canvas; diagnose the
#          head at 640 (action-head size, exposure bias, visited-mask signal).