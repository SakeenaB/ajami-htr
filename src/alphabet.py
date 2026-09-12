"""
alphabet.py
-----------
The character set for the recognition models, and the mapping between text and
the integer indices a neural network works with.

Every model in the ensemble outputs a probability distribution over this
alphabet at each step, so the alphabet must be:
  * built from the ground truth, never guessed;
  * identical for every model, or their outputs cannot be aligned;
  * verified to preserve the Ajami-specific characters through a full
    encode -> decode round trip.

That last point is the reason this file exists as a module rather than a few
lines inside a notebook. If an Ajami diacritic is silently dropped during
encoding, every downstream number in the project is measuring the wrong thing,
and nothing else in the pipeline would reveal it.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# The twelve Ajami-specific codepoints
# ---------------------------------------------------------------------------
# These are the characters in the Arabic Extended-A Unicode block that appear
# in the corpus ground truth and do not exist in standard Arabic. Four are
# letters (consonants that Hausa and Fulfulde have and Arabic does not); eight
# are combining marks (chiefly vowels, since Arabic writes only a/i/u while
# Hausa also has e and o).
#
# Section 3.7.1 defines the diacritic-restricted error rate over exactly this
# set, so it is declared once here and imported everywhere else.

AJAMI_LETTERS = {
    0x08A8,  # yeh with two dots below and hamza above
    0x08A9,  # yeh with two dots below and dot above
    0x08B3,  # ain with three dots below
    0x08BD,  # African Noon (dotless noon)
}

AJAMI_MARKS = {
    0x08F5,  # fatha with dot above
    0x08F6,  # kasra with dot below
    0x08F7,  # left arrowhead above
    0x08F8,  # right arrowhead above
    0x08F9,  # left arrowhead below
    0x08FB,  # double right arrowhead above
    0x08FC,  # double right arrowhead above with dot
    0x08FD,  # right arrowhead above with dot
}

AJAMI_CODEPOINTS = AJAMI_LETTERS | AJAMI_MARKS
AJAMI_CHARS = {chr(cp) for cp in AJAMI_CODEPOINTS}

# Invisible control characters present in the corpus. They are kept, not
# stripped: the published 10.66% benchmark was computed with them included, so
# removing them would make our character error rate incomparable.
INVISIBLES = {
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
}

# Index 0 is reserved for the CTC blank symbol and never maps to a real
# character. Real characters therefore start at index 1.
BLANK_INDEX = 0


class Alphabet:
    """Bidirectional mapping between characters and integer indices."""

    def __init__(self, chars):
        # Sorted for determinism: the same corpus must always produce the same
        # index for the same character, or a checkpoint trained today will be
        # meaningless tomorrow.
        self.chars = sorted(set(chars))
        self.char_to_index = {c: i + 1 for i, c in enumerate(self.chars)}
        self.index_to_char = {i + 1: c for i, c in enumerate(self.chars)}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_texts(cls, texts):
        """Build the alphabet from every character appearing in the ground truth."""
        chars = set()
        for t in texts:
            if t is None:
                continue
            chars.update(str(t))
        return cls(chars)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])

    def save(self, path):
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # -- encoding -----------------------------------------------------------

    def encode(self, text):
        """Text -> list of indices. Unknown characters raise rather than vanish."""
        out = []
        for c in str(text):
            if c not in self.char_to_index:
                raise KeyError(
                    f"character {c!r} (U+{ord(c):04X}) is not in the alphabet; "
                    "rebuild the alphabet from the full corpus"
                )
            out.append(self.char_to_index[c])
        return out

    def decode(self, indices):
        """Indices -> text. The CTC blank is skipped."""
        return "".join(
            self.index_to_char[i] for i in indices if i != BLANK_INDEX
        )

    def decode_ctc(self, indices):
        """
        Collapse a frame-level CTC output into a string.

        CTC emits one index per frame, so the same character repeats across the
        frames it occupies. The standard decoding rule is: drop repeats, then
        drop blanks. A blank between two identical characters is what allows a
        genuine double letter to survive.
        """
        out, previous = [], None
        for i in indices:
            if i != previous and i != BLANK_INDEX:
                out.append(self.index_to_char[i])
            previous = i
        return "".join(out)

    # -- properties ---------------------------------------------------------

    def __len__(self):
        return len(self.chars)

    @property
    def size_with_blank(self):
        """Output dimension of the final layer: every character plus the blank."""
        return len(self.chars) + 1

    def ajami_present(self):
        """The Ajami-specific characters that made it into this alphabet."""
        return sorted(c for c in self.chars if c in AJAMI_CHARS)

    def report(self):
        lines = [
            f"alphabet size        : {len(self)} characters (+1 CTC blank)",
            f"Ajami-specific found : {len(self.ajami_present())} of {len(AJAMI_CHARS)}",
        ]
        missing = AJAMI_CHARS - set(self.chars)
        if missing:
            codes = ", ".join(f"U+{ord(c):04X}" for c in sorted(missing))
            lines.append(f"not present in corpus: {codes}")
        invis = [c for c in self.chars if c in INVISIBLES]
        if invis:
            codes = ", ".join(f"U+{ord(c):04X}" for c in invis)
            lines.append(f"invisible controls   : {codes} (kept, see docstring)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def check_round_trip(alphabet, texts, verbose=True):
    """
    Confirm that encoding and decoding returns every line unchanged.

    This is the single most important check in the data pipeline. A mismatch
    means characters are being lost or altered before the model ever sees them,
    which no amount of training can recover and no other test would expose.

    Returns the list of lines that failed; an empty list means all passed.
    """
    failures = []
    for t in texts:
        t = "" if t is None else str(t)
        try:
            if alphabet.decode(alphabet.encode(t)) != t:
                failures.append(t)
        except KeyError:
            failures.append(t)

    if verbose:
        total = len(list(texts))
        if failures:
            print(f"ROUND TRIP FAILED on {len(failures)} of {total} lines")
            for t in failures[:3]:
                print("  ", repr(t[:60]))
        else:
            print(f"round trip OK on {total} lines")
    return failures


def check_normalisation(texts, verbose=True):
    """
    Warn if Unicode normalisation would alter the text.

    Some libraries normalise strings without being asked. On Arabic script that
    can reorder or recompose combining marks, which would change the character
    sequence and therefore the error rate. We normalise nothing; this check
    exists so the assumption is tested rather than hoped for.
    """
    affected = 0
    for t in texts:
        t = "" if t is None else str(t)
        if unicodedata.normalize("NFC", t) != t:
            affected += 1
    if verbose:
        if affected:
            print(f"WARNING: NFC normalisation would change {affected} lines. "
                  "Ensure no library normalises silently.")
        else:
            print("normalisation check OK (NFC leaves all lines unchanged)")
    return affected


# ---------------------------------------------------------------------------
# Run directly to build and verify the alphabet from dataset.csv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build and verify the alphabet.")
    parser.add_argument("--csv", default="dataset.csv",
                        help="CSV with a 'text' column of ground-truth lines")
    parser.add_argument("--out", default="alphabet.json",
                        help="where to save the alphabet")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).fillna({"text": ""})
    texts = df["text"].astype(str).tolist()

    alphabet = Alphabet.from_texts(texts)
    print(alphabet.report())
    print()

    check_normalisation(texts)
    failures = check_round_trip(alphabet, texts)

    if failures:
        raise SystemExit("alphabet verification failed - do not train on this")

    alphabet.save(args.out)
    print(f"\nsaved to {args.out}")
    print("Ajami characters retained:",
          " ".join(f"U+{ord(c):04X}" for c in alphabet.ajami_present()))
