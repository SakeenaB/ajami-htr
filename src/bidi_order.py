"""
bidi_order.py
-------------
Reading order for mirrored lines that mix right-to-left and left-to-right text.

The CRNN reads Arabic from MIRRORED images, so that the first character of
the transcription is the first thing the network scans (Section 3.4). For
pure Arabic that is enough: the label in logical (typing) order matches the
scan order of the mirrored image.

Digits and Latin words break this. Inside a right-to-left line they are
written left to right, so "سنة 1957" shows 1957 as a block whose own digits
run left to right, and scanning the mirrored image meets them as 7, 5, 9, 1.
Training against the logical label would ask the network to read every
number and every English word backwards.

mirror_order() converts a logical transcription into the order in which its
characters appear when the line is read from right to left, following the
Unicode Bidirectional Algorithm in simplified form:

  * an ATOM is a maximal group of left-to-right characters (Latin letters,
    European or Arabic-Indic digits) that touch, or digits of one kind joined
    by a single separator such as "10/12" or "21-9-1966" (rule W4);
  * consecutive atoms that both contain Latin letters form one run, with the
    spaces and punctuation between them ("Joseph Khouri");
  * separate numbers stay separate and keep right-to-left order between them,
    as in the written date "٢١ -٩- ١٩٦٦" (rule N1);
  * a line with Latin letters and no Arabic is left to right as a whole.

Each run is reversed in place. The transform is its own inverse on all but
pathological strings, so the same function turns a prediction made on a
mirrored image back into logical order; involution_failures() measures the
exceptions on a corpus.

The Ajami corpus is trained without this option, so its results are unchanged.
"""

from __future__ import annotations

import unicodedata


def _cls(c):
    b = unicodedata.bidirectional(c)
    if b in ("R", "AL"):
        return "R"
    if b == "L":
        return "LAT"
    if b in ("EN", "AN"):
        return b
    if b in ("ES", "CS"):
        return b
    return "N"


def _atoms(text, cls):
    """Maximal left-to-right atoms as (start, end) pairs, end exclusive."""
    ltr = [k in ("LAT", "EN", "AN") for k in cls]
    atoms, i, n = [], 0, len(text)
    while i < n:
        if not ltr[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            if ltr[j]:
                j += 1
                continue
            # digits joined by one separator: EN (ES|CS) EN, or AN CS AN
            if (j + 1 < n and cls[j - 1] == cls[j + 1]
                    and ((cls[j - 1] == "EN" and cls[j] in ("ES", "CS"))
                         or (cls[j - 1] == "AN" and cls[j] == "CS"))):
                j += 2
                continue
            break
        atoms.append((i, j))
        i = j
    return atoms


def ltr_runs(text):
    """(start, end) pairs of the runs that are written left to right."""
    cls = [_cls(c) for c in text]
    if "R" not in cls and "LAT" in cls:
        return [(0, len(text))] if text else []
    atoms = _atoms(text, cls)
    runs = []
    for s, e in atoms:
        latin = any(cls[k] == "LAT" for k in range(s, e))
        if runs and latin and runs[-1][2]:
            ps, pe, _ = runs[-1]
            if "R" not in cls[pe:s]:
                runs[-1] = (ps, e, True)
                continue
        runs.append((s, e, latin))
    return [(s, e) for s, e, _ in runs]


def mirror_order(text):
    """Logical order <-> right-to-left scan order of the line."""
    chars = list(text)
    for s, e in ltr_runs(text):
        chars[s:e] = chars[s:e][::-1]
    return "".join(chars)


def mirror_order_with_confidence(text, confidence):
    """mirror_order(), moving each character's confidence with it."""
    chars, conf = list(text), list(confidence)
    for s, e in ltr_runs(text):
        chars[s:e] = chars[s:e][::-1]
        conf[s:e] = conf[s:e][::-1]
    return "".join(chars), conf


def involution_failures(texts):
    """Lines on which applying the transform twice does not restore the text."""
    return [t for t in texts if mirror_order(mirror_order(t)) != t]


def _self_test():
    cases = [
        ("الجرد بعض الاشقياء", "الجرد بعض الاشقياء"),            # Arabic only
        ("سنة 1957 م", "سنة 7591 م"),                             # a number
        ("بتاريخ ٢١ -٩- ١٩٦٦ حضر", "بتاريخ ١٢ -٩- ٦٦٩١ حضر"),     # a written date
        ("في 21-9-1966 حضر", "في 6691-9-12 حضر"),                 # joined by '-'
        ("صفحة 10/12", "صفحة 21/01"),                             # joined by '/'
        ("سنة 1957 1958", "سنة 7591 8591"),                       # separate numbers
        ("MG26-43 10/12", "21/01 34-62GM"),                       # no Arabic: whole line
        ("اسم Joseph Khouri هنا", "اسم iruohK hpesoJ هنا"),       # English words
        ("", ""),
    ]
    for logical, scan in cases:
        assert mirror_order(logical) == scan, (logical, mirror_order(logical))
        assert mirror_order(scan) == logical, (scan, mirror_order(scan))
    t, c = mirror_order_with_confidence("سنة 1957", [1, 2, 3, 4, 5, 6, 7, 8])
    assert t == "سنة 7591" and c == [1, 2, 3, 4, 8, 7, 6, 5], (t, c)
    import random
    rng = random.Random(0)
    pool = list("ابتثجحخدذرسشصضطظعغفقكلمنهوي      ٠١٢٣٤٥٦٧٨٩0123456789abcXYZ.,:-/()")
    texts = ["".join(rng.choice(pool) for _ in range(rng.randint(0, 40))) for _ in range(20000)]
    fails = involution_failures(texts)
    assert all(len(mirror_order(s)) == len(s) for s in texts)
    print(f"bidi_order: {len(cases)} reading-order cases passed; "
          f"{len(fails)} of {len(texts)} random strings not restored by a double transform")
    return fails


if __name__ == "__main__":
    for f in _self_test()[:5]:
        print("  example:", repr(f), "->", repr(mirror_order(f)), "->", repr(mirror_order(mirror_order(f))))
