"""Train the Stage-1 module detector (Ultralytics YOLO) on the Zenodo UAV set.

Single class (module), 640x512 thermal frames. Fine-tunes a COCO-pretrained
YOLO; returns the best-weights path and the validation metrics.
"""

from __future__ import annotations

from pathlib import Path


def train_detector(
    data_yaml: str | Path = "configs/zenodo_uav.yaml",
    model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    out_dir: str | Path = "runs/detector",
    device: str | None = None,
) -> tuple[Path, dict]:
    from ultralytics import YOLO

    out_dir = Path(out_dir).resolve()
    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=True,
    )
    metrics = yolo.val(data=str(data_yaml), split="test", verbose=False)
    # Use the trainer's actual save_dir (Ultralytics can relocate the run).
    best = Path(yolo.trainer.save_dir) / "weights" / "best.pt"
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    print(f"\nBEST weights -> {best}")
    print(f"TEST mAP@50={summary['map50']:.3f}  mAP@50-95={summary['map50_95']:.3f}  "
          f"P={summary['precision']:.3f}  R={summary['recall']:.3f}")
    return best, summary
