"""
train_trocr.py
--------------
Stage 2 of the TrOCR adaptation: fine-tune the vocabulary-extended model.

TrOCR is the ensemble's contextual member. The vision transformer attends
across the whole line and the decoder generates text autoregressively, so it
can infer damaged characters from their surroundings - and, for the same
reason, has no frame axis to share with the CTC models (Section 3.6).

Training runs in two phases.

  Warm-up (default 1 epoch). Only the decoder's token embeddings and output
  head are trained, at a higher learning rate. The tokens added for this
  corpus start at the mean of the existing embeddings and know nothing yet;
  letting them settle first stops their early, noisy gradients from
  disturbing the pretrained vision encoder and decoder.

  Fine-tune. Everything trains at a low learning rate, with early stopping
  on validation CER. The model has roughly half a billion parameters against
  4,344 lines, so overfitting is the main risk.

All text passes through trocr_codec, never through tok.decode(): the plain
decoder inserts spaces between tokens, which would corrupt every prediction.

Smoke test:
    python train_trocr.py --model trocr_ajami --limit 64 --epochs 2
Full run:
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
import trocr_codec as codec                      # noqa: E402
from metrics import cer, wer, diacritic_cer      # noqa: E402


class TrOCRLineDataset(Dataset):
    """
    Line image -> pixel tensor via the model's image processor; text -> token
    ids via the codec. The processor resizes to the fixed square the vision
    transformer expects, so aspect ratio is not preserved - one of the
    architectural differences from the CRNN, and part of why they fail
    differently.
    """

    def __init__(self, csv, images_dir, processor, tokenizer, split,
                 max_len=192, transform=None, limit=None):
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
            img = Image.fromarray(self.transform(image=np.array(img))["image"])
        pixels = self.processor(img.convert("RGB"),
                                return_tensors="pt").pixel_values[0]
        ids = codec.encode(self.tok, row.text, max_length=self.max_len,
                           truncation=True, padding="max_length",
                           return_tensors="pt")["input_ids"][0]
        labels = ids.clone()
        labels[labels == self.tok.pad_token_id] = -100   # ignored by the loss
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
def predict(model, tok, loader, device, max_len=192, beams=1):
    """
    Generate transcriptions with a per-character confidence.

    generate() gives a probability per TOKEN; each character inherits the
    probability of the token that produced it. Coarser than the CRNN's
    per-column softmax, but on the same 0-1 scale, which is what the
    meta-learner needs (Section 3.6.3).
    """
    model.eval()
    special = set(tok.all_special_ids)
    rows = []
    for batch in loader:
        out = model.generate(batch["pixel_values"].to(device),
                             max_length=max_len, num_beams=beams,
                             output_scores=True, return_dict_in_generate=True)
        seqs = out.sequences
        step_probs = None
        if out.scores:
            ps = []
            for step, logits in enumerate(out.scores):
                p = torch.softmax(logits.float(), dim=-1)
                chosen = seqs[:, step + 1]            # position 0 is the start token
                ps.append(p.gather(1, chosen[:, None]).squeeze(1))
            step_probs = torch.stack(ps, dim=1).cpu().numpy()

        for b in range(seqs.shape[0]):
            ids = seqs[b].tolist()
            text, conf = [], []
            for step, tid in enumerate(ids[1:]):
                if tid in special:
                    continue
                piece = codec.token_text(tok.convert_ids_to_tokens(tid))
                text.append(piece)
                p = float(step_probs[b, step]) if step_probs is not None else 0.0
                conf.extend([p] * len(piece))
            text = "".join(text)
            rows.append({"ImageID": batch["image_ids"][b],
                         "reference": batch["texts"][b],
                         "prediction": text,
                         "lang": batch["langs"][b],
                         "confidence": json.dumps(conf[:len(text)])})
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


def set_phase(model, warmup, freeze_encoder):
    """Choose which parameters train in the current phase."""
    for p in model.parameters():
        p.requires_grad = not warmup
    if warmup:
        model.decoder.get_input_embeddings().weight.requires_grad = True
        head = model.decoder.get_output_embeddings()
        if head is not None:
            head.weight.requires_grad = True
    elif freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False


def make_optimiser(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"  optimiser: {n/1e6:.1f}M trainable parameters at lr {lr}")
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trocr_ajami")
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--images", default="lines")
    ap.add_argument("--out", default="trocr_ckpt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation; effective batch = "
                         "batch-size x accum")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--warmup-lr", type=float, default=5e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=192)
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

    # Sanity: the codec must reproduce a real line exactly before training.
    sample = pd.read_csv(args.csv).fillna({"text": ""}).text.iloc[0]
    back = codec.round_trip(tok, sample)
    print("codec check:", "exact" if back == sample
          else f"MISMATCH - {codec.first_difference(sample, back)}")

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

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, bad, history = float("inf"), 0, []
    optim = None

    for epoch in range(1, args.epochs + 1):
        warm = epoch <= args.warmup_epochs
        if epoch == 1 or epoch == args.warmup_epochs + 1:
            print(f"\n--- {'warm-up: embeddings and head only' if warm else 'fine-tune'} ---")
            set_phase(model, warm, args.freeze_encoder)
            optim = make_optimiser(model, args.warmup_lr if warm else args.lr)

        model.train()
        t0, total, n = time.time(), 0.0, 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(pixel_values=batch["pixel_values"].to(device),
                             labels=batch["labels"].to(device)).loss
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train_loader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            total += loss.item()
            n += 1

        rows = predict(model, tok, val_loader, device, args.max_len)
        res = score(rows)
        line = (f"epoch {epoch:>3}{' (warm-up)' if warm else ''}  "
                f"loss {total/max(1, n):.4f}  {time.time()-t0:6.0f}s  |  "
                f"val CER {res['cer']*100:6.2f}%  WER {res['wer']*100:6.2f}%  "
                f"diacritic {res['diacritic_cer']*100:6.2f}%")
        history.append({"epoch": epoch, "warmup": warm,
                        "loss": total / max(1, n), **res})

        # Early stopping only counts fine-tuning epochs.
        if res["cer"] < best:
            best, bad = res["cer"], 0
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")
            proc.save_pretrained(out_dir / "best")
            pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv", index=False)
            line += "  *"
        elif not warm:
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
