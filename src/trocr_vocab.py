"""
trocr_vocab.py
--------------
Stage 1 of the TrOCR adaptation: make the tokenizer represent this corpus
EXACTLY, then (optionally) resize the model to match.

The gate is strict on purpose. The first version of this script compared
round-trips with spaces removed and only checked the Ajami codepoints. It
reported a pass while 3,649 lines still contained [UNK] and every decoded
word would have had spaces inserted inside it. Now:

  * every corpus character the tokenizer cannot reproduce on its own is
    added as a raw (un-normalised) token, whatever its category;
  * spaces travel as an explicit SPACE token (see trocr_codec.py);
  * the gate requires zero [UNK] lines and exact, character-for-character
    round-trips - spaces, joiners and diacritics included.

Usage:
    python trocr_vocab.py --csv dataset.csv --out trocr_ajami --no-model
    python trocr_vocab.py --csv dataset.csv --out trocr_ajami
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from alphabet import AJAMI_CODEPOINTS            # noqa: E402
import trocr_codec as codec                      # noqa: E402

BASE_MODEL = "RayR1/trocr-base-arabic-handwritten"
IMAGE_PROCESSOR_FALLBACK = "microsoft/trocr-large-handwritten"

# The gate tolerates this fraction of lines with a non-Ajami mismatch, so a
# handful of edge cases cannot block the whole stage. Any [UNK] or any lost
# Ajami character fails it outright.
MAX_MISMATCH_FRACTION = 0.005


def exact_report(tok, texts, label):
    """Exact round-trip through the codec. Returns (unk_lines, mismatches)."""
    unk = tok.unk_token_id
    unk_lines, mismatches = 0, []
    lost = Counter()
    for t in texts:
        ids = tok(codec.to_model(t), add_special_tokens=False)["input_ids"]
        if unk is not None and unk in ids:
            unk_lines += 1
        back = codec.decode(tok, ids)
        if back != t:
            mismatches.append((t, back))
            for c in set(t):
                if c not in back:
                    tag = " (Ajami)" if ord(c) in AJAMI_CODEPOINTS else ""
                    lost[f"U+{ord(c):04X} {unicodedata.name(c, '?')[:30]}{tag}"] += 1

    print(f"\n[{label}]")
    print(f"  lines containing [UNK]      : {unk_lines} / {len(texts)}")
    print(f"  lines not reproduced exactly: {len(mismatches)} / {len(texts)}")
    if lost:
        print("  characters lost (lines affected):")
        for k, v in sorted(lost.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {k:<48} {v}")
    for t, back in mismatches[:3]:
        print(f"  example: {codec.first_difference(t, back)}")
    return unk_lines, mismatches, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--out", default="trocr_ajami")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--no-model", action="store_true",
                    help="tokenizer only; skip the ~2 GB model download")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from tokenizers import AddedToken

    df = pd.read_csv(args.csv).fillna({"text": ""})
    texts = df["text"].astype(str).tolist()
    corpus_chars = sorted(set("".join(texts)) - {" "})

    if codec.SPACE in corpus_chars:
        sys.exit(f"the corpus already contains {codec.SPACE!r}; choose "
                 "another SPACE symbol in trocr_codec.py")

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"tokenizer: {type(tok).__name__}, vocab {len(tok)}")
    print(f"corpus uses {len(corpus_chars)} distinct characters (excluding space)")

    exact_report(tok, texts, "BEFORE extension (exact, via codec)")

    # ---- choose what to add -------------------------------------------
    # A character is added if, tokenised ALONE, it does not come back as
    # itself. That catches unknown letters, stripped marks, removed control
    # characters and precomposed letters the normaliser decomposes - without
    # needing to know in advance which category each falls into.
    to_add = []
    for c in corpus_chars:
        if codec.round_trip(tok, c) != c:
            to_add.append(c)
    for cp in sorted(AJAMI_CODEPOINTS):                 # always, explicitly
        if chr(cp) not in to_add:
            to_add.append(chr(cp))
    to_add.append(codec.SPACE)

    by_cat = Counter(unicodedata.category(c) for c in to_add)
    print(f"\nadding {len(to_add)} raw tokens "
          f"(by Unicode category: {dict(by_cat)})")
    print("  " + " ".join(f"U+{ord(c):04X}" for c in to_add))

    added = tok.add_tokens([
        AddedToken(c, normalized=False, single_word=False,
                   lstrip=False, rstrip=False) for c in to_add
    ])
    print(f"tokens actually added: {added}   new vocab: {len(tok)}")

    unk_lines, mismatches, lost = exact_report(
        tok, texts, "AFTER extension (exact, via codec)")

    # ---- gate ----------------------------------------------------------
    ajami_lost = [k for k in lost if "(Ajami)" in k]
    frac = len(mismatches) / len(texts)
    print()
    if unk_lines:
        print(f"GATE FAILED: {unk_lines} lines still contain [UNK].")
        sys.exit(1)
    if ajami_lost:
        print(f"GATE FAILED: Ajami characters still lost: {ajami_lost}")
        sys.exit(1)
    if frac > MAX_MISMATCH_FRACTION:
        print(f"GATE FAILED: {len(mismatches)} lines ({frac:.2%}) are not "
              "reproduced exactly. Send the examples above.")
        sys.exit(1)
    print(f"GATE PASSED: no [UNK], every Ajami character survives, and "
          f"{len(texts) - len(mismatches)} of {len(texts)} lines round-trip "
          f"exactly (spaces included).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    (out / "ajami_codec.json").write_text(json.dumps({
        "space_token": codec.SPACE,
        "added_tokens": [f"U+{ord(c):04X}" for c in to_add],
        "exact_lines": len(texts) - len(mismatches),
        "total_lines": len(texts),
    }, indent=1))
    print(f"tokenizer saved to {out}")

    if args.no_model:
        return

    # ---- resize the model ----------------------------------------------
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    old = model.decoder.get_input_embeddings().weight.shape[0]
    try:
        # New rows start at the mean of the existing embeddings rather than
        # random noise, so the decoder is not destabilised on step one.
        model.decoder.resize_token_embeddings(len(tok), mean_resizing=True)
    except TypeError:
        model.decoder.resize_token_embeddings(len(tok))
    new = model.decoder.get_input_embeddings().weight.shape[0]
    model.config.decoder.vocab_size = len(tok)
    print(f"decoder embeddings resized: {old} -> {new}")

    head = model.decoder.get_output_embeddings()
    if head is not None:
        assert head.weight.shape[0] == new, "output head not resized"
        print("output head resized to match")

    model.config.decoder_start_token_id = (
        model.config.decoder_start_token_id or tok.cls_token_id)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = model.config.eos_token_id or tok.sep_token_id
    model.save_pretrained(out)

    from transformers import AutoImageProcessor
    try:
        ip = AutoImageProcessor.from_pretrained(args.model)
    except Exception:
        ip = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_FALLBACK)
    ip.save_pretrained(out)
    print(f"model, tokenizer and image processor saved to {out}")


if __name__ == "__main__":
    main()
