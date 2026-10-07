"""
train_trocr.py
--------------
Stage 2 of the TrOCR adaptation: fine-tune the vocabulary-extended model on
the Ajami line images.

TrOCR is the ensemble's contextual member. Where the CRNN reads a short
window at a time, the vision transformer attends over the whole line and the
decoder generates text autoregressively, so it can infer damaged characters
from their surroundings. That is also why it cannot share a frame axis with
the CTC models and why reconciliation happens at string level (Section 3.6).

Two risks drive the settings here:

  * Overfitting. The model has roughly 0.5 billion parameters against 4,344
    training lines. The learning rate is kept very low, early stopping is on
    validation CER, and photometric augmentation is applied.

  * The new embeddings. The twelve Ajami tokens were added with random
    embeddings. They must be learned from scratch, so they get a higher
    learning rate than the pretrained weights.

Run a smoke test first:
    python train_trocr.py --model trocr_ajami --limit 64 --epochs 1

Then the real run:
    python train_trocr.py --model trocr_ajami --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
import augment                                   # noqa: E402
from alphabet import AJAMI_CHARS                 # noqa: E402
from metrics import cer, wer, diacritic_cer      # noqa: E402


class TrOCRLineDataset(Dataset):
    """
    Line image -> pixel tensor via the model's own image processor, text ->
    token ids. The image processor resizes to the fixed square the vision
    transformer expects; aspect ratio is NOT preserved, which is one of the
    architectural differences from the CRNN and part of why the two fail
    differently.
    """

    def __init__(self, csv, images_dir, processor, tokenizer, split,
                 max_len=160, transform=None, limit=None):
        df = pd.read_csv(csv).fillna({"text": ""})
        df = df[(df["split"] == split) & (df["text"].str.len() > 0)]
        if limit:
            df = df.head(limit)
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.tok = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.images_dir / f"{row.ImageID}.png").convert("L")
        if self.transform is not None:
            arr = self.transform(image=np.array(img))["image"]
            img = Image.fromarray(arr)
        pixels = self.processor(img.convert("RGB"),
                                return_tensors="pt").pixel_values[0]
        ids = self.tok(row.text, max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt").input_ids[0]
        # Padding positions must not contribute to the loss.
        labels = ids.clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {"pixel_values": pixels, "labels": labels,
                "text": row.text, "image_id": row.ImageID, "lang": row.lang}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "langs": [b["lang"] for b in batch],
    }


@torch.no_grad()
def predict(model, tok, loader, device, max_len=160, beams=1):
    """
    Generate transcriptions and a per-character confidence.

    generate() returns a probability per TOKEN. A WordPiece token may span
    several characters, so each character inherits the probability of the
    token that produced it. That is the confidence the meta-learner consumes
    (Section 3.6.3): coarser than the CRNN's per-column softmax, but on the
    same 0-1 scale.
    """
    model.eval()
    rows = []
    for batch in loader:
        out = model.generate(
            batch["pixel_values"].to(device),
            max_length=max_len,
            num_beams=beams,
            output_scores=True,
            return_dict_in_generate=True,
        )
        seqs = out.sequences
        # Per-step probability of the chosen token.
        probs = []
        if out.scores:
            for step, logits in enumerate(out.scores):
                p = torch.softmax(logits.float(), dim=-1)
                chosen = seqs[:, step + 1]            # step 0 is the start token
                probs.append(p.gather(1, chosen[:, None]).squeeze(1))
            probs = torch.stack(probs, dim=1).cpu().numpy()  # (B, steps)
        for b in range(seqs.shape[0]):
            ids = seqs[b].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            # Map token probs onto characters.
            conf = []
            if len(probs):
                for step, tid in enumerate(ids[1:]):
                    if tid in tok.all_special_ids:
                        continue
                    piece = tok.decode([tid], skip_special_tokens=True)
                    conf.extend([float(probs[b, step])] * max(1, len(piece)))
            conf = conf[:len(text)] + [0.0] * max(0, len(text) - len(conf))
            rows.append({"ImageID": batch["image_ids"][b],
                         "reference": batch["texts"][b],
                         "prediction": text,
                         "lang": batch["langs"][b],
                         "confidence": json.dumps(conf)})
    return rows


def score(rows):
    refs = [r["reference"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    res = {"cer": cer(refs, hyps), "wer": wer(refs, hyps),
           "diacritic_cer": diacritic_cer(refs, hyps), "n": len(rows)}
    for lang in sorted({r["lang"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        res[f"cer_{lang.lower()}"] = cer([refs[i] for i in idx],
                                         [hyps[i] for i in idx])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trocr_ajami",
                    help="output of trocr_vocab.py")
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--images", default="lines")
    ap.add_argument("--out", default="trocr_ckpt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation steps; effective batch = "
                         "batch-size x accum")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--new-token-lr", type=float, default=1e-4,
                    help="higher rate for the freshly added embeddings")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train the decoder only (less memory, less risk, "
                         "less capacity to adapt to degraded images)")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()

    from transformers import (AutoImageProcessor, AutoTokenizer,
                              VisionEncoderDecoderModel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.config.decoder_start_token_id = (
        model.config.decoder_start_token_id or tok.cls_token_id)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = model.config.eos_token_id or tok.sep_token_id
    print(f"vocab {len(tok)}   params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("encoder frozen")

    # ---- data ----------------------------------------------------------
    train_ds = TrOCRLineDataset(args.csv, args.images, proc, tok, "train",
                                args.max_len, augment.get("photometric", 0),
                                args.limit)
    val_ds = TrOCRLineDataset(args.csv, args.images, proc, tok, "val",
                              args.max_len, None,
                              max(16, args.limit // 4) if args.limit else None)
    print(f"train {len(train_ds)}   val {len(val_ds)}")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)

    # ---- optimiser: new embeddings learn faster ------------------------
    emb = model.decoder.get_input_embeddings().weight
    new_ids = torch.tensor(sorted(tok.get_added_vocab().values()),
                           device=device)
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and p is not emb]
    optim = torch.optim.AdamW([
        {"params": other, "lr": args.lr},
        {"params": [emb], "lr": args.lr},
    ], weight_decay=0.01)
    print(f"{len(new_ids)} new token embeddings get lr {args.new_token_lr}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, bad, history = float("inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(pixel_values=batch["pixel_values"].to(device),
                             labels=batch["labels"].to(device)).loss
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Boost the gradient on the new rows so they learn faster
                # than the pretrained vocabulary at the same base lr.
                if emb.grad is not None:
                    emb.grad[new_ids] *= args.new_token_lr / args.lr
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            total += loss.item()
            n += 1

        rows = predict(model, tok, val_loader, device, args.max_len)
        res = score(rows)
        line = (f"epoch {epoch:>3}  loss {total/max(1,n):.4f}  "
                f"{time.time()-t0:6.0f}s  |  val CER {res['cer']*100:6.2f}%  "
                f"WER {res['wer']*100:6.2f}%  diacritic "
                f"{res['diacritic_cer']*100:6.2f}%")
        history.append({"epoch": epoch, "loss": total / max(1, n), **res})
        if res["cer"] < best:
            best, bad = res["cer"], 0
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")
            proc.save_pretrained(out_dir / "best")
            pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv",
                                      index=False)
            line += "  *"
        else:
            bad += 1
        print(line, flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        if bad >= args.patience:
            print(f"no improvement for {args.patience} epochs - stopping")
            break

    print(f"\nbest val CER {best*100:.2f}%")
    for r in rows[:3]:
        print(f"  ref : {r['reference']}\n  pred: {r['prediction']}\n")


if __name__ == "__main__":
    main()
