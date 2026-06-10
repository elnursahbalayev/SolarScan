"""Train the IR fault classifier on InfraredSolarModules.

Tackles the 57:1 class imbalance with a ``WeightedRandomSampler`` (balanced
batches) + label smoothing + mild augmentation, and selects the best checkpoint
by **validation macro-F1** (not accuracy). Designed for the RTX A4000: AMP on,
batched data loading. Writes checkpoint, metrics, and a confusion matrix to the
run directory.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from solarscan.config import Config
from solarscan.data.infrared_modules import (
    InfraredSolarModules,
    class_counts,
    load_samples,
    stratified_split,
)
from solarscan.data.transforms import build_transforms
from solarscan.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
    save_confusion_png,
)
from solarscan.models.timm_classifier import TimmClassifier


def _balanced_sampler(samples) -> WeightedRandomSampler:
    counts = class_counts(samples)
    weights = [1.0 / counts[s.label] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


@torch.inference_mode()
def evaluate(model: TimmClassifier, loader: DataLoader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred


def train_classifier(
    cfg: Config,
    out_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[Path, ClassificationMetrics]:
    """Train and evaluate; returns (best_checkpoint_path, test_metrics).

    ``limit`` caps samples per split for fast smoke tests.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, ccfg = cfg.train, cfg.classifier

    out_dir = Path(out_dir or f"runs/classifier-{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(tcfg.seed)

    samples = load_samples(cfg.data.root)
    train_s, val_s, test_s = stratified_split(
        samples, tcfg.val_ratio, tcfg.test_ratio, tcfg.seed
    )
    if limit:
        train_s, val_s, test_s = train_s[:limit], val_s[:limit], test_s[:limit]

    train_tf = build_transforms(ccfg.input_size, train=True)
    eval_tf = build_transforms(ccfg.input_size, train=False)
    train_ds = InfraredSolarModules(train_s, train_tf)
    val_ds = InfraredSolarModules(val_s, eval_tf)
    test_ds = InfraredSolarModules(test_s, eval_tf)

    sampler = _balanced_sampler(train_s) if ccfg.class_balanced_sampling else None
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=tcfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers, pin_memory=pin
    )

    model = TimmClassifier(
        backbone=ccfg.backbone,
        num_classes=ccfg.num_classes,
        input_size=ccfg.input_size,
        pretrained=ccfg.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = tcfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device}  backbone={ccfg.backbone}  "
          f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    print(f"train class balance: {dict(Counter(s.label for s in train_s))}")

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tcfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()

        y_true, y_pred = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1),
                        "val_macro_f1": val_metrics.macro_f1, "val_acc": val_metrics.accuracy})
        print(f"epoch {epoch}: train_loss={running / max(len(train_loader), 1):.4f}  "
              f"val {val_metrics.summary_line()}")

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            model.save(best_path)

    # Final test eval with the best checkpoint.
    best_model = TimmClassifier.load(best_path, device=device)
    y_true, y_pred = evaluate(best_model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred)
    test_metrics.to_json(out_dir / "test_metrics.json")
    save_confusion_png(test_metrics.confusion, out_dir / "confusion_matrix.png")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nBEST val macro-F1={best_f1:.4f}")
    print(f"TEST {test_metrics.summary_line()}")
    print(f"artifacts -> {out_dir}")
    return best_path, test_metrics
