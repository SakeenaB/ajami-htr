"""
trocr_codec.py
--------------
Exact, reversible conversion between Ajami text and TrOCR token ids.

The Arabic TrOCR checkpoint uses a BERT-style WordPiece tokenizer. Out of the
box it cannot reproduce this corpus, for three separate reasons:

  1. Missing characters. The twelve Ajami codepoints, letters such as keheh
     (U+06A9), and the invisible joiners all map to [UNK].

  2. Normalisation. The tokenizer strips combining marks and control
     characters before lookup, so standard vowels and the zero-width
     joiners vanish even when they are in the vocabulary.

  3. Spacing. WordPiece does not store where spaces were. It decodes by
     joining tokens WITH spaces and gluing "##" continuations. Once a mark
     like U+08F8 is its own token, a decoded word comes back with spaces
     inside it, which would wreck every CER figure.

The fixes, in the same order:

  1-2. Every corpus character the tokenizer cannot reproduce is added as a
       token with normalized=False, so it is matched on the RAW text before
       any normalisation can touch it.

  3.   Real spaces are replaced by a dedicated SPACE token (U+2581) before
       tokenising, and decoding concatenates tokens with NO separator,
       turning SPACE back into a space. Spacing becomes explicit in the
       token stream, so runs of several spaces between hemistichs survive
       too - something CTC models struggle with.

Use these functions everywhere the model meets text. Never call tok.decode()
on model output directly; it reinserts the spurious spaces.
"""

from __future__ import annotations

SPACE = "\u2581"          # LOWER ONE EIGHTH BLOCK, the conventional word marker


def to_model(text: str) -> str:
    """Text as stored in the corpus -> text as the tokenizer must see it."""
    return text.replace(" ", SPACE)


def token_text(token: str) -> str:
    """One WordPiece token string -> the characters it contributes."""
    if token.startswith("##"):
        token = token[2:]
    return token.replace(SPACE, " ")


def decode(tok, ids) -> str:
    """Token ids -> corpus text, with no spurious spaces."""
    tokens = tok.convert_ids_to_tokens(list(ids), skip_special_tokens=True)
    return "".join(token_text(t) for t in tokens)


def encode(tok, text: str, **kwargs):
    """Corpus text -> token ids. kwargs pass through to the tokenizer."""
    return tok(to_model(text), **kwargs)


def round_trip(tok, text: str) -> str:
    ids = tok(to_model(text), add_special_tokens=False)["input_ids"]
    return decode(tok, ids)


def first_difference(a: str, b: str) -> str:
    """Human-readable description of where two strings first differ."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"at char {i}: expected U+{ord(x):04X} {x!r}, "
                    f"got U+{ord(y):04X} {y!r}")
    if len(a) != len(b):
        return f"length differs: expected {len(a)}, got {len(b)}"
    return "identical"
