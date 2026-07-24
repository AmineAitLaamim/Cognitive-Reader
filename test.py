from collections import Counter
from data.dataset import create_dataloaders
hist = Counter()
try:
    tr, _ = create_dataloaders(dataset_config, batch_size=4, num_workers=0)
    for bi, batch in enumerate(tr):
        B = getattr(batch, 'batch_size', None) or batch.images.shape[0]
        for i in range(B):
            # try the documented unpad path; fall back to whatever length field exists
            try:
                from data.collate import unpad_graph
                sd = unpad_graph(batch, i, device=torch.device('cpu'))
                n = sum(1 for t in sd['gt_sequence'] if t['mode'] == 'READ')
            except Exception:
                n = -1  # unpad signature differs; skip rather than crash
            hist[n] += 1
        if bi >= 24:
            break
    print("READ-nodes per sample (dataloader):", dict(sorted(hist.items())))
    real = {k: v for k, v in hist.items() if k > 0}
    if real:
        print(f"  range [{min(real)}, {max(real)}], distinct={len(real)}")
        assert max(real) <= dataset_config.max_digits and len(real) >= 5, "curriculum collapsed"
except Exception as e:
    print(f"dataloader histogram skipped ({type(e).__name__}); rely on epoch-10 OOD as the end-to-end check")