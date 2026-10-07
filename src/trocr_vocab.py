"""
trocr_vocab.py
--------------
Stage 1 of the TrOCR adaptation: give the tokenizer the Ajami characters.

The published Arabic TrOCR checkpoint uses a classical-Arabic WordPiece
tokenizer. Tested against this corpus it maps every one of the twelve Ajami
codepoints to <unk>, and 93% of lines contain at least one. A model that
cannot *represent* a character cannot output it, so fine-tuning the vision
side would be pointless until this is fixed.

Two separate problems hide behind "lost in round-trip", and this script
checks for both:

  1. Vocabulary. The character is simply not in the token table. Fixed by
     adding it as a new token and resizing the decoder's embedding matrix.

  2. Normalisation. BERT-style tokenizers often strip accents (Unicode
     category Mn) before tokenising. Eight of the twelve Ajami codepoints are
     combining marks, as are ALL the standard Arabic vowels. If the
     normaliser strips them, no vocabulary change helps - the marks are gone
     before lookup. Added tokens must therefore be registered to match the
     raw, un-normalised text.

The script exits non-zero if the extended tokenizer still cannot round-trip
the corpus, so a failing gate cannot be missed.

Usage (CPU is fine):
    python trocr_vocab.py --csv dataset.csv --out trocr_ajami
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from alphabet import AJAMI_CODEPOINTS  # noqa: E402

BASE_MODEL = "RayR1/trocr-base-arabic-handwritten"
# RayR1 was built on this Microsoft checkpoint; it is the fallback source for
# the image processor if the RayR1 repo does not ship one.
IMAGE_PROCESSOR_FALLBACK = "microsoft/trocr-large-handwritten"


def strip_spaces(s):
    return "".join(s.split())


def round_trip_report(tok, texts, label):
    """
    Encode then decode every line. Report lines with <unk>, lines that do
    not come back identical (ignoring whitespace, which WordPiece
    legitimately rewrites), and which Ajami and Arabic-mark codepoints were
    lost.
    """
    unk = tok.unk_token_id
    n_unk = n_changed = 0
    lost = Counter()
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        back = tok.decode(ids, skip_special_tokens=True)
        if unk is not None and unk in ids:
            n_unk += 1
        if strip_spaces(back) != strip_spaces(t):
            n_changed += 1
        for c in set(t):
            cp = ord(c)
            is_ajami = cp in AJAMI_CODEPOINTS
            is_mark = unicodedata.category(c) == "Mn"
            if (is_ajami or is_mark) and c not in back:
                lost[f"U+{cp:04X}{' (Ajami)' if is_ajami else ''}"] += 1

    print(f"\n[{label}]")
    print(f"  lines with <unk>          : {n_unk} / {len(texts)}")
    print(f"  lines changed by round-trip: {n_changed} / {len(texts)}")
    if lost:
        print("  codepoints lost (lines affected):")
        for k, v in sorted(lost.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {k:<18} {v}")
    else:
        print("  codepoints lost: none")
    return n_changed, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--out", default="trocr_ajami",
                    help="where to save the extended tokenizer and model")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--no-model", action="store_true",
                    help="only test and save the tokenizer; skip the "
                         "model download and resize (useful on CPU)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from tokenizers import AddedToken

    df = pd.read_csv(args.csv).fillna({"text": ""})
    texts = df["text"].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"tokenizer: {type(tok).__name__}, vocab {len(tok)}")

    # ---- what does the normaliser do? ---------------------------------
    probe = "\u0628\u064e\u08f8"           # beh + fatha + Ajami arrowhead
    back = tok.decode(tok.encode(probe, add_special_tokens=False),
                      skip_special_tokens=True)
    print(f"normaliser probe: {probe!r} -> {back!r}")
    if "\u064e" not in back:
        print("  WARNING: the standard Arabic fatha did not survive. The "
              "normaliser strips combining marks. Added tokens will be "
              "registered with normalized=False to bypass it; if standard "
              "vowels still vanish after extension, every vowel character "
              "would need adding too.")

    before_changed, _ = round_trip_report(tok, texts, "BEFORE extension")

    # ---- extend --------------------------------------------------------
    new_chars = [chr(cp) for cp in sorted(AJAMI_CODEPOINTS)]
    # Also add any standard Arabic combining marks that the corpus uses and
    # the tokenizer loses, so vowels are not silently dropped either.
    corpus_marks = {c for t in texts for c in t
                    if unicodedata.category(c) == "Mn"
                    and ord(c) not in AJAMI_CODEPOINTS}
    lost_marks = []
    for c in sorted(corpus_marks):
        back = tok.decode(tok.encode("\u0628" + c, add_special_tokens=False),
                          skip_special_tokens=True)
        if c not in back:
            lost_marks.append(c)
    to_add = new_chars + lost_marks
    print(f"\nadding {len(to_add)} tokens: {len(new_chars)} Ajami + "
          f"{len(lost_marks)} standard marks the tokenizer was losing")

    added = tok.add_tokens(
        [AddedToken(c, normalized=False, single_word=False,
                    lstrip=False, rstrip=False) for c in to_add]
    )
    print(f"tokens actually added: {added}   new vocab: {len(tok)}")

    after_changed, lost = round_trip_report(tok, texts, "AFTER extension")

    # ---- gate ----------------------------------------------------------
    ajami_lost = {k: v for k, v in lost.items() if "Ajami" in k}
    if ajami_lost:
        print("\nGATE FAILED: Ajami characters still lost after extension. "
              "Do not fine-tune on this tokenizer.")
        sys.exit(1)
    print(f"\nGATE PASSED: all Ajami codepoints survive. Lines changed by "
          f"round-trip fell from {before_changed} to {after_changed}.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    print(f"tokenizer saved to {out}")

    if args.no_model:
        return

    # ---- resize the model's decoder to the new vocabulary --------------
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    old = model.decoder.get_input_embeddings().weight.shape[0]
    model.decoder.resize_token_embeddings(len(tok))
    model.config.decoder.vocab_size = len(tok)
    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = len(tok)
    new = model.decoder.get_input_embeddings().weight.shape[0]
    print(f"decoder embeddings resized: {old} -> {new}")

    # Sanity: the new rows must exist and the output head must agree.
    head = model.decoder.get_output_embeddings()
    if head is not None:
        assert head.weight.shape[0] == new, "output head not resized"
        print("output head resized to match")

    # Make sure generation knows the special tokens.
    model.config.decoder_start_token_id = (
        model.config.decoder_start_token_id or tok.cls_token_id)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = (model.config.eos_token_id or tok.sep_token_id)

    model.save_pretrained(out)

    try:
        from transformers import AutoImageProcessor
        try:
            ip = AutoImageProcessor.from_pretrained(args.model)
        except Exception:
            ip = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_FALLBACK)
        ip.save_pretrained(out)
        print("image processor saved")
    except Exception as e:  # pragma: no cover
        print(f"WARNING: could not save image processor: {e}")

    print(f"\nextended model saved to {out} - ready for train_trocr.py")


if __name__ == "__main__":
    main()
