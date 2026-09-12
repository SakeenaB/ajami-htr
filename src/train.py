"""
train.py
--------
Trains the CRNN baseline with CTC loss.

Run it in two modes.

  Smoke test - a few hundred lines, a handful of epochs, on purpose too small
  to learn anything:

      python train.py --limit 200 --epochs 5 --batch-size 8

  The point is not accuracy. It is to prove that images load, the alphabet
  encodes, CTC accepts the shapes, decoding returns strings and CER computes -
  before any of that is buried under a six-hour run. Nearly every
  catastrophic failure in a project like this is a plumbing bug found late.

  Full run - the configuration of Al-azzawi, Barney & Liwicki (2026), whose
  reported figure on this corpus is 10.66% CER:

      python train.py --epochs 800 --batch-size 16 --lr 5e-4

Checkpoints are written after every epoch. Kaggle sessions end without
warning, and an interrupted run that resumes costs minutes; one that does not
costs hours.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from alphabet import Alphabet, check_round_trip
from crnn import CRNN, greedy_decode
from dataset import AjamiLineDataset, make_loader
from metrics import cer, wer, diacritic_cer


def build_alphabet(csv_path, out_path):
    """
    Build the alphabet from the whole corpus and verify it before training.

    From the whole corpus, not just the training split: a character that
    appears only in validation must still be representable, or evaluation
    would crash partway through.
    """
    import pandas as pd

    df = pd.read_csv(csv_path).fillna({"text": ""})
    texts = df["text"].astype(str).tolist()

    alphabet = Alphabet.from_texts(texts)
    print(alphabet.report())

    failures = check_round_trip(alphabet, texts)
    if failures:
        raise SystemExit(
            "alphabet round trip failed - characters would be lost before the "
            "model ever sees them. Fix this before training."
        )

    alphabet.save(out_path)
    return alphabet


def check_ctc_feasible(dataset, model, sample=None):
    """
    Confirm every line has more output time steps than label characters.

    CTC cannot align a label to a shorter output sequence: the loss for such
    an item is infinite, and with zero_infinity=True it silently becomes
    zero, so the line contributes nothing and nothing warns you. On a corpus
    where lines reach 106 characters, a narrow crop scaled to 64 pixels high
    can fall below its own label length.

    Run once before training. If lines fail, raise the input height so that
    widths scale up proportionally.
    """
    n = len(dataset) if sample is None else min(sample, len(dataset))
    bad = []
    for i in range(n):
        item = dataset[i]
        steps = model.output_length(item["width"])
        needed = len(item["label"])
        if steps < needed:
            bad.append((item["image_id"], item["width"], steps, needed))

    if bad:
        print(f"WARNING: {len(bad)} of {n} lines are too narrow for their "
              f"label - CTC will skip them silently.")
        for image_id, width, steps, needed in bad[:5]:
            print(f"    {image_id}: width {width} gives {steps} steps "
                  f"for {needed} characters")
        print("    Increase --height so images scale wider.")
    else:
        print(f"CTC length check OK on {n} lines "
              f"(every output sequence is longer than its label)")
    return bad


def evaluate(model, loader, alphabet, device, max_batches=None):
    """Decode a whole split and return the three metrics plus sample output."""
    model.eval()
    refs, hyps, langs = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = batch["images"].to(device)
            log_probs = model(images)
            lengths = [model.output_length(int(w)) for w in batch["image_widths"]]
            hyps.extend(greedy_decode(log_probs, alphabet, lengths))
            refs.extend(batch["texts"])
            langs.extend(batch["langs"])

    result = {
        "cer": cer(refs, hyps),
        "wer": wer(refs, hyps),
        "diacritic_cer": diacritic_cer(refs, hyps),
        "n": len(refs),
    }

    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        result[f"cer_{lang.lower()}"] = cer([refs[i] for i in idx],
                                            [hyps[i] for i in idx])

    result["samples"] = list(zip(refs[:3], hyps[:3]))
    return result


def main():
    p = argparse.ArgumentParser(description="Train the CRNN baseline.")
    p.add_argument("--csv", default="dataset.csv")
    p.add_argument("--images", default="lines")
    p.add_argument("--alphabet", default="alphabet.json")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="use only this many training lines (smoke test)")
    p.add_argument("--patience", type=int, default=20,
                   help="stop after this many epochs without a better val CER")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--weighted", action="store_true",
                   help="sample Fulfulde and Hausa equally in expectation")
    p.add_argument("--overfit", action="store_true",
                   help="diagnostic: validate on the training lines themselves. "
                        "A healthy model can drive a few dozen lines to near "
                        "zero CER. If it cannot, the fault is in the model or "
                        "the loss, not in the amount of data.")
    p.add_argument("--no-schedule", action="store_true",
                   help="hold the learning rate constant (use for short runs, "
                        "where the milestones would otherwise fire immediately)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU. Fine for a smoke test, far too slow for a "
              "full run - use Kaggle with the accelerator enabled.")

    # ---- alphabet -------------------------------------------------------
    if Path(args.alphabet).exists():
        alphabet = Alphabet.load(args.alphabet)
        print(f"loaded alphabet: {len(alphabet)} characters")
    else:
        alphabet = build_alphabet(args.csv, args.alphabet)
    print()

    # ---- data -----------------------------------------------------------
    train_ds = AjamiLineDataset(args.csv, args.images, alphabet,
                                split="train", height=args.height)
    val_ds = AjamiLineDataset(args.csv, args.images, alphabet,
                              split="val", height=args.height)

    if args.limit:
        train_ds.df = train_ds.df.head(args.limit).reset_index(drop=True)
        val_ds.df = val_ds.df.head(max(20, args.limit // 4)).reset_index(drop=True)
        print(f"SMOKE TEST: {len(train_ds)} train, {len(val_ds)} val lines")

    if args.overfit:
        # Validate on the training lines. This is deliberately the one thing
        # you must never do when measuring performance - and exactly the right
        # thing when asking whether the model can learn at all.
        val_ds.df = train_ds.df.copy()
        print("OVERFIT DIAGNOSTIC: validating on the training lines. "
              "CER should approach zero. If it plateaus high, the fault is "
              "in the model or the loss, not the data.")

    print("train split:"); print(train_ds.report())
    print("val split:");   print(val_ds.report())
    print()

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.workers,
                               weighted_by_language=args.weighted)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False,
                             num_workers=args.workers)

    # ---- model ----------------------------------------------------------
    model = CRNN(n_classes=alphabet.size_with_blank,
                 input_height=args.height).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    check_ctc_feasible(train_ds, model, sample=500)
    print()

    # zero_infinity guards against a label longer than the output sequence,
    # which produces an infinite loss and would otherwise poison training.
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Step the learning rate down at 50% and 75% of training, as in the
    # reference configuration. On a short run the milestones fire almost
    # immediately and strangle learning, so --no-schedule disables them.
    if args.no_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimiser, milestones=[], gamma=1.0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimiser,
            milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
            gamma=0.1,
        )

    start_epoch, best_cer, bad_epochs = 1, float("inf"), 0
    history = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        history = ckpt.get("history", [])
        print(f"resumed from epoch {ckpt['epoch']}, best val CER {best_cer:.4f}")

    print("\n" + "-" * 62)

    # ---- training loop --------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            log_probs = model(images)          # (time, batch, classes)

            # True output length per item. Using the padded width here would
            # tell CTC there is more room than the image really occupies.
            input_lengths = torch.tensor(
                [model.output_length(int(w)) for w in batch["image_widths"]],
                dtype=torch.long, device=device,
            )

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimiser.zero_grad()
            loss.backward()
            # Recurrent networks are prone to exploding gradients; clipping
            # keeps a single bad batch from destabilising the whole run.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0

        line = (f"epoch {epoch:>4}  loss {train_loss:7.4f}  "
                f"{elapsed:5.1f}s  lr {optimiser.param_groups[0]['lr']:.1e}")

        # ---- validation --------------------------------------------------
        if epoch % args.eval_every == 0:
            val = evaluate(model, val_loader, alphabet, device)
            line += (f"  |  val CER {val['cer']*100:6.2f}%  "
                     f"WER {val['wer']*100:6.2f}%  "
                     f"diacritic {val['diacritic_cer']*100:6.2f}%")

            history.append({"epoch": epoch, "loss": train_loss, **{
                k: v for k, v in val.items() if k != "samples"}})

            improved = val["cer"] < best_cer
            if improved:
                best_cer = val["cer"]
                bad_epochs = 0
                torch.save({
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_cer": best_cer,
                    "history": history,
                    "alphabet_size": alphabet.size_with_blank,
                }, out_dir / "best.pt")
                line += "  *"
            else:
                bad_epochs += 1

        print(line)

        # Checkpoint every epoch regardless: a free-tier session can end at
        # any moment and --resume needs somewhere to resume from.
        torch.save({
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_cer": best_cer,
            "history": history,
            "alphabet_size": alphabet.size_with_blank,
        }, out_dir / "last.pt")

        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

        if bad_epochs >= args.patience:
            print(f"\nno improvement for {args.patience} evaluations - stopping")
            break

    # ---- finish ----------------------------------------------------------
    print("-" * 62)
    print(f"best validation CER: {best_cer*100:.2f}%   (benchmark 10.66%)")

    val = evaluate(model, val_loader, alphabet, device)
    print("\nsample predictions")
    for ref, hyp in val["samples"]:
        mark = "match" if ref == hyp else "differs"
        print(f"  [{mark}]")
        print(f"    reference : {ref}")
        print(f"    prediction: {hyp}")

    if args.limit:
        print("\nSmoke test complete. A high CER here is expected and means "
              "nothing about the method - what it shows is that the pipeline "
              "runs end to end. Remove --limit for the real run.")


if __name__ == "__main__":
    main()
