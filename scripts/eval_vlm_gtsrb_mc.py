#!/usr/bin/env python
"""Multiple-choice traffic sign recognition with a small instruction-tuned VLM on GTSRB.

Protocol: a class-stratified subset of the official GTSRB test split (``--per-class`` images per
class, seed-fixed), each crop upscaled to ``--size`` pixels (GTSRB crops are 15 to 250 px and
VLMs are not meant for 30 px inputs), one chat turn containing the image, the numbered list of
the 43 GTSRB sign names and the instruction to answer with the number only. The first integer in
the reply is the prediction; replies without an integer count as errors. Greedy decoding.

This is the "zero-shot, closed vocabulary" setting that the LMM traffic sign papers use
(for example arXiv 2409.01534); results are not comparable with CNNs trained on GTSRB but show
where general-purpose VLMs stand on fine-grained sign classes.

Usage:
    python scripts/eval_vlm_gtsrb_mc.py --model HuggingFaceTB/SmolVLM-500M-Instruct --per-class 10
    python scripts/eval_vlm_gtsrb_mc.py --model Qwen/Qwen2-VL-2B-Instruct --per-class 5 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import SHORT_NAMES, load_gtsrb  # noqa: E402


def build_prompt() -> str:
    options = "\n".join(f"{i}: {name}" for i, name in enumerate(SHORT_NAMES))
    return (
        "This is a photo of a German traffic sign. Which of the following signs is it?\n"
        f"{options}\n"
        "Answer with the number of the correct option only."
    )


def stratified_subset(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(43):
        pool = np.where(labels == c)[0]
        idx.extend(rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist())
    return np.array(sorted(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224, help="upscale the crop to this side length")
    ap.add_argument("--use-roi", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-image-splitting", action="store_true", help="SmolVLM: feed one 512 px tile instead of a tiled grid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor_kwargs = {}
    if "qwen" in args.model.lower():
        # cap the visual tokens: a 224 px crop needs no more than a few hundred tokens
        processor_kwargs = {"min_pixels": 64 * 28 * 28, "max_pixels": 256 * 28 * 28}
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    if args.no_image_splitting and hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
        processor.image_processor.do_image_splitting = False  # SmolVLM / Idefics3: one image, not 4 to 13 tiles
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).eval()

    data = load_gtsrb(args.root, "test")
    idx = stratified_subset(data.labels, args.per_class, args.seed)
    if args.limit:
        idx = idx[: args.limit]
    prompt = build_prompt()

    records = []
    t0 = time.time()
    for k, i in enumerate(idx):
        img = data.images[i]
        if args.use_roi:
            img = img.crop(tuple(data.rois[i].tolist()))
        img = img.resize((args.size, args.size), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        if dtype == torch.bfloat16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        m = re.search(r"\d+", reply)
        pred = int(m.group()) if m else -1
        records.append({"index": int(i), "label": int(data.labels[i]), "pred": pred, "reply": reply})
        if k % 20 == 0:
            print(f"{k + 1}/{len(idx)} {time.time() - t0:.0f}s  label {data.labels[i]} pred {pred} reply {reply!r}", flush=True)

    labels = np.array([r["label"] for r in records])
    preds = np.array([r["pred"] for r in records])
    acc = float((preds == labels).mean())
    per_class = {SHORT_NAMES[c]: float((preds[labels == c] == c).mean()) for c in range(43) if (labels == c).any()}
    result = {
        "model": args.model,
        "dtype": args.dtype,
        "protocol": {
            "subset": f"stratified {args.per_class} per class, seed {args.seed}",
            "images": len(records),
            "input_size": args.size,
            "image_splitting": not args.no_image_splitting,
            "crop": "official ROI" if args.use_roi else "full image",
            "prompt": prompt,
            "decoding": "greedy",
        },
        "top1_accuracy": acc,
        "unparsable_replies": int((preds < 0).sum()),
        "per_class_accuracy": per_class,
        "records": records,
        "seconds": time.time() - t0,
    }
    out = Path(args.out) if args.out else Path("runs/gtsrb") / f"vlm_mc_{args.model.replace('/', '__')}_pc{args.per_class}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.model}: multiple-choice top-1 {acc * 100:.2f}% on {len(records)} images ({result['unparsable_replies']} unparsable), {time.time() - t0:.0f}s")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
