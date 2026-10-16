"""
predict.py
----------
Run a trained base model over chosen parts of the data and save every
prediction with a confidence for each character.

These files are the hand-off between the base models and the meta-learner.
Nothing downstream needs a GPU: alignment, the learned scorer and every
Chapter 4 table work from the CSV files written here.

Parts:
  valA  validation half A - used to select the base models (sanity check:
        scoring it must reproduce the CER the training log reported)
  valB  validation half B - never seen during model selection; the
        meta-learner is trained on these predictions
  test  the test set - scored ONCE, at the end

Output columns:
  model, part, ImageID, lang, reference, prediction, confidence
where confidence is a JSON list with one value per character of the
prediction.

Examples:
  python predict.py --type crnn --checkpoint best.pt --alphabet alphabet.json \
      --images /tmp/lines --name crnn1 --out preds_crnn1.csv
  python predict.py --type trocr --checkpoint trocr_ckpt/best \
      --images /tmp/lines --name trocr --out preds_trocr.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# CTC decoding with confidences (pure numpy, so it can be tested anywhere)
# ---------------------------------------------------------------------------

def ctc_decode_with_confidence(probs, index_to_char, blank=0):
    """
    Best-path CTC decoding that also returns a confidence per character.

    probs: (T, C) array of per-frame probabilities for one line.

    A character usually spans several consecutive frames. Its confidence is
    the highest probability it reached in any of those frames - the frame
    where the model was most certain the character was there. Repeats
    separated by a blank are separate characters, exactly as in standard
    CTC decoding, so the text returned here is identical to
    Alphabet.decode_ctc on the same frames.
    """
    best = probs.argmax(axis=1)
    best_p = probs[np.arange(len(best)), best]

    chars, confs = [], []
    prev = None
    for idx, p in zip(best.tolist(), best_p.tolist()):
        if idx == blank:
            prev = idx
            continue
        if idx == prev:
            confs[-1] = max(confs[-1], p)       # same character, later frame
        else:
            chars.append(index_to_char[idx])
            confs.append(p)
        prev = idx
    return "".join(chars), confs


# ---------------------------------------------------------------------------
# selecting the parts
# ---------------------------------------------------------------------------

def part_mask(df, part):
    from dataset import val_half
    if part == "test":
        return df["split"] == "test"
    if part in ("valA", "valB"):
        return (df["split"] == "val") & (
            df["ImageID"].map(val_half) == part[-1])
    raise ValueError(f"unknown part {part!r}")


# ---------------------------------------------------------------------------
# CRNN
# ---------------------------------------------------------------------------

def predict_crnn(args, parts):
    import torch
    from alphabet import Alphabet
    from crnn import CRNN
    from dataset import AjamiLineDataset, make_loader
    from bidi_order import mirror_order_with_confidence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    alphabet = Alphabet.load(args.alphabet)
    ckpt = torch.load(args.checkpoint, map_location=device)
    n_classes = ckpt.get("alphabet_size", alphabet.size_with_blank)
    if n_classes != alphabet.size_with_blank:
        raise SystemExit(f"checkpoint expects {n_classes} classes but the "
                         f"alphabet has {alphabet.size_with_blank}; use the "
                         "alphabet.json saved with this model")
    model = CRNN(n_classes=n_classes, input_height=args.height).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"best val CER {ckpt.get('best_cer', float('nan'))*100:.2f}%)")

    rows = []
    for part in parts:
        split = "test" if part == "test" else "val"
        # Mirroring must match training: the model reads right-to-left text
        # from mirrored images.
        ds = AjamiLineDataset(args.csv, args.images, alphabet, split=split,
                              height=args.height, flip=not args.no_flip,
                              bidi_runs=args.bidi_runs)
        ds.df = ds.df[part_mask(ds.df, part)].reset_index(drop=True)
        loader = make_loader(ds, batch_size=32, shuffle=False,
                             num_workers=args.workers)
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                log_probs = model(batch["images"].to(device))     # (T, B, C)
                probs = log_probs.exp().cpu().numpy()
                for b in range(probs.shape[1]):
                    T = model.output_length(int(batch["image_widths"][b]))
                    text, conf = ctc_decode_with_confidence(
                        probs[:T, b, :], alphabet.index_to_char)
                    if args.bidi_runs and not args.no_flip:
                        text, conf = mirror_order_with_confidence(text, conf)
                    rows.append({"part": part,
                                 "ImageID": batch["image_ids"][b],
                                 "lang": batch["langs"][b],
                                 "reference": batch["texts"][b],
                                 "prediction": text,
                                 "confidence": json.dumps(
                                     [round(c, 4) for c in conf])})
        print(f"  {part}: {len(ds)} lines in {time.time()-t0:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# TrOCR
# ---------------------------------------------------------------------------

def predict_trocr(args, parts):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoImageProcessor, AutoTokenizer,
                              VisionEncoderDecoderModel)
    from train_trocr import (TrOCRLineDataset, collate,
                             configure_special_tokens, predict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    proc = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).to(device)
    configure_special_tokens(model, tok, args.max_len)
    model.eval()

    rows = []
    for part in parts:
        split = "test" if part == "test" else "val"
        ds = TrOCRLineDataset(args.csv, args.images, proc, tok, split,
                              args.max_len, transform=None)
        ds.df = ds.df[part_mask(ds.df, part)].reset_index(drop=True)
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)
        t0 = time.time()
        for r in predict(model, tok, loader, device, args.max_len):
            r = dict(r)
            r["part"] = part
            # predict() already stores confidence as JSON; keep it as is.
            rows.append(r)
        print(f"  {part}: {len(ds)} lines in {time.time()-t0:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def summarise(df, show_test=False):
    from metrics import cer, wer, diacritic_cer
    print()
    for part, g in df.groupby("part", sort=False):
        if part == "test" and not show_test:
            print(f"test  n={len(g):<4} (score withheld - the test set is "
                  f"scored once, at the end; use --show-test then)")
            continue
        refs, hyps = g["reference"].tolist(), g["prediction"].tolist()
        line = (f"{part:<5} n={len(g):<4} CER {cer(refs, hyps)*100:6.2f}%  "
                f"WER {wer(refs, hyps)*100:6.2f}%  "
                f"diacritic {diacritic_cer(refs, hyps)*100:6.2f}%")
        for lang, gl in g.groupby("lang"):
            lang_cer = cer(gl["reference"].tolist(), gl["prediction"].tolist())
            line += f"   {lang} {lang_cer*100:.2f}%"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=["crnn", "trocr"])
    ap.add_argument("--checkpoint", required=True,
                    help="crnn: path to best.pt; trocr: the best/ folder")
    ap.add_argument("--alphabet", default=None,
                    help="crnn only: the alphabet.json saved with the model")
    ap.add_argument("--name", required=True,
                    help="label for this model in the output, e.g. crnn1")
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--images", default="lines")
    ap.add_argument("--parts", nargs="+", default=["valA", "valB", "test"],
                    choices=["valA", "valB", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--bidi-runs", action="store_true",
                    help="CRNN trained with --bidi-runs: put each prediction "
                         "back into logical order")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--show-test", action="store_true",
                    help="print test-set scores. Leave off until the final "
                         "evaluation, so no decision is influenced by them.")
    args = ap.parse_args()

    if args.type == "crnn" and not args.alphabet:
        sys.exit("--alphabet is required for a CRNN")

    rows = (predict_crnn if args.type == "crnn" else predict_trocr)(args, args.parts)
    df = pd.DataFrame(rows)
    df.insert(0, "model", args.name)
    df = df[["model", "part", "ImageID", "lang", "reference",
             "prediction", "confidence"]]

    # Every prediction must carry exactly one confidence per character.
    bad = sum(len(json.loads(c)) != len(p)
              for c, p in zip(df["confidence"], df["prediction"].fillna("")))
    if bad:
        print(f"WARNING: {bad} rows have a confidence list that does not "
              f"match the prediction length")

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nwrote {len(df)} predictions to {args.out}")
    summarise(df, args.show_test)
    if "valA" in args.parts:
        print("\nsanity check: valA CER above should match the best "
              "validation CER in this model's training log")


if __name__ == "__main__":
    main()
