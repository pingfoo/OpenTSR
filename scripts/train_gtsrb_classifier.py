#!/usr/bin/env python
"""Train a small CNN on the official GTSRB training split and score it on the official test split.

The run is fully described by a YAML config under ``configs/gtsrb/`` (model, image size,
epochs, optimiser, augmentation, seed). No model selection is done on the test set: the model
after the last epoch is the one scored. Per-epoch training loss and test accuracy are logged so
that the learning curve is visible, but only the final number is reported in ``docs/``.

Outputs a JSON file with the config, the final accuracy, per-class recall, the confusion matrix
and wall-clock time, plus the weights next to it (gitignored).

Usage:
    python scripts/train_gtsrb_classifier.py --config configs/gtsrb/lenet5_gray_32.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import SHORT_NAMES, load_gtsrb, to_arrays  # noqa: E402
from opentsr.signs.models.gtsrb_cnn import build_model, random_affine_batch, random_photometric_batch  # noqa: E402


def load_split_tensor(root: Path, split: str, size: int, cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    cache = cache_dir / f"gtsrb_{split}_{size}.npz"
    if cache.exists():
        z = np.load(cache)
        x, y = z["x"], z["y"]
    else:
        data = load_gtsrb(root, split)
        x = to_arrays(data, size)
        y = data.labels
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x, y=y)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> tuple[float, np.ndarray]:
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].float() / 255.0
        preds.append(model(xb).argmax(1))
    pred = torch.cat(preds).numpy()
    return float((pred == y.numpy()).mean()), pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--out", default=None, help="JSON output path (default runs/gtsrb/<config name>.json)")
    ap.add_argument("--epochs", type=int, default=None, help="override the config (smoke tests)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.threads:
        torch.set_num_threads(args.threads)
    name = Path(args.config).stem
    out = Path(args.out) if args.out else Path("runs/gtsrb") / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    size = int(cfg.get("image_size", 32))
    root = Path(args.root)
    cache_dir = root / "cache"
    x_train, y_train = load_split_tensor(root, "train", size, cache_dir)
    x_test, y_test = load_split_tensor(root, "test", size, cache_dir)
    print(f"train {tuple(x_train.shape)} test {tuple(x_test.shape)}", flush=True)

    model = build_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    opt_cfg = cfg.get("optimizer", {"name": "adam", "lr": 1e-3})
    if opt_cfg["name"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    elif opt_cfg["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg.get("momentum", 0.9), weight_decay=opt_cfg.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(opt_cfg["name"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg.get("batch_size", 64))
    steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
    sched_name = cfg.get("schedule", "cosine")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
        if sched_name == "cosine"
        else None
    )
    aug = cfg.get("augmentation") or {}

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), generator=gen)
        total_loss, seen = 0.0, 0
        te = time.time()
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx].float() / 255.0
            yb = y_train[idx]
            if aug.get("affine"):
                a = aug["affine"]
                xb = random_affine_batch(
                    xb,
                    degrees=a.get("degrees", 15),
                    translate=a.get("translate", 0.1),
                    scale=tuple(a.get("scale", (0.9, 1.1))),
                    shear_deg=a.get("shear", 2),
                    generator=gen,
                )
            if aug.get("photometric"):
                p = aug["photometric"]
                xb = random_photometric_batch(xb, p.get("brightness", 0.4), p.get("contrast", 0.4), generator=gen)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
        acc, _ = evaluate(model, x_test, y_test)
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "test_acc": acc, "epoch_seconds": time.time() - te})
        print(f"epoch {epoch:3d}  loss {total_loss / seen:.4f}  test acc {acc * 100:.2f}%  ({time.time() - te:.0f}s)", flush=True)

    acc, pred = evaluate(model, x_test, y_test)
    y_np = y_test.numpy()
    cm = np.zeros((43, 43), dtype=int)
    np.add.at(cm, (y_np, pred), 1)
    per_class_recall = (cm.diagonal() / cm.sum(1)).tolist()
    result = {
        "config_file": args.config,
        "config": cfg,
        "dataset": {"root": str(root), "train_images": int(len(x_train)), "test_images": int(len(x_test))},
        "model_parameters": int(n_params),
        "final_test_accuracy": acc,
        "final_test_errors": int((pred != y_np).sum()),
        "per_class_recall": {SHORT_NAMES[i]: r for i, r in enumerate(per_class_recall)},
        "confusion_matrix": cm.tolist(),
        "history": history,
        "wall_clock_seconds": time.time() - t0,
        "environment": {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }
    out.write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    print(f"final test accuracy {acc * 100:.2f}% ({result['final_test_errors']} errors of {len(y_np)}), {result['wall_clock_seconds'] / 60:.1f} min")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
