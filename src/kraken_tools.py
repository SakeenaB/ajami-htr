"""
kraken_tools.py
---------------
Kraken (Kiessling, 2019) as the third architecture in the ensemble.

Kraken is trained with its own command-line tool, `ketos`, not with this
project's training loop. This file does the two things around it:

  prep     write the training and validation lines in Kraken's "path"
           format: each line image with a matching .gt.txt transcription,
           plus list files naming which images are for training and which
           for validation (validation half A, as for every other model)

  predict  run a trained Kraken model over valA, valB and test and save the
           predictions with per-character confidences, in exactly the same
           CSV format as predict.py, so ensemble.py can use it unchanged

Right-to-left handling. Kraken does NOT need mirrored images: it applies the
Unicode bidirectional algorithm to the labels during training (ketos
--base-dir R) and reorders its output back to reading order. Images are
therefore given as they are on the page.

Whitespace. ketos collapses runs of spaces by default. This corpus uses
deliberate multiple spaces between verse halves, so training must be run
with --no-normalize-whitespace.

Usage:
  python kraken_tools.py prep --csv dataset.csv --images /tmp/lines --out /tmp/kraken
  python kraken_tools.py predict --model ajami_best.mlmodel --csv dataset.csv \
      --images /tmp/lines --name kraken --out preds_kraken.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def val_half(image_id):
    """Identical to dataset.val_half, repeated here so prep needs no torch."""
    return "A" if zlib.crc32(str(image_id).encode("utf-8")) % 2 == 0 else "B"


def in_part(row, part):
    if part == "test":
        return row["split"] == "test"
    if part in ("valA", "valB"):
        return row["split"] == "val" and val_half(row["ImageID"]) == part[-1]
    if part == "train":
        return row["split"] == "train"
    raise ValueError(part)


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

def prep(args):
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    df = df[df["text"].str.len() > 0]
    out = Path(args.out)
    images = Path(args.images)
    lists = {"train": [], "valA": []}
    for part in lists:
        d = out / part
        d.mkdir(parents=True, exist_ok=True)
        rows = df[df.apply(lambda r: in_part(r, part), axis=1)]
        for r in rows.itertuples(index=False):
            src = images / f"{r.ImageID}.png"
            if not src.exists():
                continue
            dst = d / f"{r.ImageID}.png"
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            # Written exactly as in the corpus: no stripping, no trailing
            # newline, so the multiple spaces between verse halves survive.
            (d / f"{r.ImageID}.gt.txt").write_text(r.text, encoding="utf-8")
            lists[part].append(str(dst))
        (out / f"{part}.txt").write_text("\n".join(lists[part]) + "\n",
                                         encoding="utf-8")
        print(f"{part}: {len(lists[part])} lines -> {out/part}")
    print(f"list files: {out/'train.txt'}  {out/'valA.txt'}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def recognise(net, image):
    """
    Recognise one pre-cut line image with Kraken and return (text, conf).

    Kraken's recognition API changed between versions: 5.x takes a
    Segmentation object, 4.x a plain dictionary. Both are tried.
    """
    from kraken import rpred
    w, h = image.size
    try:
        from kraken.containers import BBoxLine, Segmentation
        seg = Segmentation(type="bbox", imagename="line", text_direction="horizontal-lr",
                           script_detection=False,
                           lines=[BBoxLine(id="l0", bbox=(0, 0, w, h))])
        records = list(rpred.rpred(net, image, seg))
    except ImportError:
        records = list(rpred.rpred(net, image, {"boxes": [(0, 0, w, h)],
                                                "text_direction": "horizontal-lr"}))
    if not records:
        return "", []
    rec = records[0]
    text = rec.prediction
    conf = [float(c) for c in getattr(rec, "confidences", [])]
    conf = (conf + [0.5] * len(text))[: len(text)]
    return text, conf


def predict(args):
    from PIL import Image
    from kraken.lib import models

    device = args.device
    print("loading", args.model, "on", device)
    try:
        net = models.load_any(args.model, device=device)
    except TypeError:
        net = models.load_any(args.model)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    df = df[df["text"].str.len() > 0]
    rows = []
    for part in args.parts:
        sub = df[df.apply(lambda r: in_part(r, part), axis=1)]
        if args.limit:
            sub = sub.head(args.limit)
        t0 = time.time()
        for r in sub.itertuples(index=False):
            img = Image.open(Path(args.images) / f"{r.ImageID}.png").convert("L")
            text, conf = recognise(net, img)
            rows.append({"model": args.name, "part": part, "ImageID": r.ImageID,
                         "lang": r.lang, "reference": r.text, "prediction": text,
                         "confidence": json.dumps([round(c, 4) for c in conf])})
        print(f"  {part}: {len(sub)} lines in {time.time()-t0:.0f}s")
        if rows:
            print(f"    first ref : {rows[-len(sub)]['reference'][:60]}")
            print(f"    first pred: {rows[-len(sub)]['prediction'][:60]}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nwrote {len(out)} predictions to {args.out}")

    from predict import summarise
    summarise(out, show_test=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--csv", default="dataset.csv")
    p.add_argument("--images", default="lines")
    p.add_argument("--out", default="/tmp/kraken")
    q = sub.add_parser("predict")
    q.add_argument("--model", required=True)
    q.add_argument("--csv", default="dataset.csv")
    q.add_argument("--images", default="lines")
    q.add_argument("--name", default="kraken")
    q.add_argument("--parts", nargs="+", default=["valA", "valB", "test"])
    q.add_argument("--out", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--limit", type=int, default=None,
                   help="only the first N lines of each part (smoke test)")
    args = ap.parse_args()
    (prep if args.cmd == "prep" else predict)(args)


if __name__ == "__main__":
    main()
