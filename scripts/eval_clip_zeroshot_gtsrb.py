#!/usr/bin/env python
"""Zero-shot CLIP-style classification of the official GTSRB test split.

Reproduces the protocol of the OpenAI CLIP paper (Radford et al. 2021, arXiv 2103.00020,
Table 11 "GTSRB" column): the 43 class descriptions and the three prompt templates from
https://github.com/openai/CLIP/blob/main/data/prompts.md, text embeddings averaged over the
templates (prompt ensembling), cosine similarity argmax, top-1 accuracy on the 12,630 test
images. Any open_clip model can be scored, so LAION / DataComp / SigLIP checkpoints are
compared under the identical protocol.

Usage:
    python scripts/eval_clip_zeroshot_gtsrb.py --model ViT-B-32 --pretrained openai
    python scripts/eval_clip_zeroshot_gtsrb.py --model ViT-B-16 --pretrained laion2b_s34b_b88k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import CLASS_NAMES, SHORT_NAMES, load_gtsrb  # noqa: E402

TEMPLATES = [
    'a zoomed in photo of a "{}" traffic sign.',
    'a centered photo of a "{}" traffic sign.',
    'a close up photo of a "{}" traffic sign.',
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N images (smoke test)")
    ap.add_argument("--single-template", action="store_true", help="use only the first template, no ensembling")
    ap.add_argument("--use-roi", action="store_true", help="crop to the official ROI before resizing")
    ap.add_argument(
        "--keep-gelu-mismatch",
        action="store_true",
        help="do not switch to the -quickgelu architecture for OpenAI weights (reproduces the pitfall described in docs)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import open_clip

    # Pitfall: open_clip 3.x loads pretrained="openai" into the plain "ViT-B-32" config with standard
    # GELU and only warns ("QuickGELU mismatch"); the OpenAI weights were trained with QuickGELU and
    # lose about 6 points of GTSRB zero-shot accuracy that way (26.6 vs 32.6, measured 2026-10-31).
    model_name = args.model
    if args.pretrained == "openai" and not model_name.endswith("-quickgelu") and not args.keep_gelu_mismatch:
        model_name = model_name + "-quickgelu"
        print(f"using {model_name} for the OpenAI weights (QuickGELU)")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=args.pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    templates = TEMPLATES[:1] if args.single_template else TEMPLATES
    with torch.no_grad():
        text_feats = []
        for name in CLASS_NAMES:
            toks = tokenizer([t.format(name) for t in templates])
            emb = model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(0)
            text_feats.append(emb / emb.norm())
        text_feats = torch.stack(text_feats)  # (43, D)

    data = load_gtsrb(args.root, args.split)
    n = len(data.images) if not args.limit else min(args.limit, len(data.images))
    preds = np.empty(n, dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            imgs = []
            for j in range(i, min(i + args.batch_size, n)):
                img = data.images[j]
                if args.use_roi:
                    img = img.crop(tuple(data.rois[j].tolist()))
                imgs.append(preprocess(img))
            x = torch.stack(imgs)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            preds[i : i + len(imgs)] = (f @ text_feats.T).argmax(1).numpy()
            if (i // args.batch_size) % 20 == 0:
                print(f"{i + len(imgs)}/{n}  {time.time() - t0:.0f}s", flush=True)
    labels = data.labels[:n]
    acc = float((preds == labels).mean())
    cm = np.zeros((43, 43), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    per_class = (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist()
    result = {
        "model": args.model,
        "model_config_used": model_name,
        "pretrained": args.pretrained,
        "protocol": {
            "templates": templates,
            "class_names": "openai/CLIP data/prompts.md GTSRB list",
            "ensembling": not args.single_template,
            "crop": "official ROI" if args.use_roi else "full image (as in the CLIP paper's torchvision loader)",
            "split": args.split,
            "images": n,
        },
        "top1_accuracy": acc,
        "per_class_recall": {SHORT_NAMES[i]: r for i, r in enumerate(per_class)},
        "confusion_matrix": cm.tolist(),
        "seconds": time.time() - t0,
    }
    tag = f"{model_name}_{args.pretrained}".replace("/", "-")
    out = Path(args.out) if args.out else Path("runs/gtsrb") / f"clip_zeroshot_{tag}{'_roi' if args.use_roi else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.model} {args.pretrained}: zero-shot top-1 {acc * 100:.2f}% on {n} images ({time.time() - t0:.0f}s)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
