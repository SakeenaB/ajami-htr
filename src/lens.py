"""
lens.py
-------
Iʿjām (إعجام, "the dots that tell letters apart"): point it at a manuscript
page and it reads the page. Formerly the Ajami Lens. Iʿjām shares its Arabic
root, ʿ-j-m, with "Ajami".

  page image -> find the lines -> every model of the chosen script reads every
             line -> the confidence vote combines them (the final system of
             the evaluation) -> the page with each line boxed, beside the
             text, with the characters the models disputed marked by blue dots.

Two scripts, each with exactly the members and member order of its evaluated
final system:
  Ajami  : CRNN-1, CRNN-2, TrOCR            (test CER 10.42%, benchmark 10.66%)
  Arabic : CRNN-1, CRNN-3, CRNN-2, TrOCR    (test CER 10.17%; 10.12% against
           12.36% for the published system on the lines both read)

Line finding is not part of the evaluated system: both corpora came with line
boxes. Iʿjām uses Kraken's pretrained page segmenter when available (run
from its own environment, because Kraken and TrOCR need incompatible library
versions), and falls back to a simple projection-profile segmenter that suits
clean single-column scans. Recognition uses exactly the preprocessing, models
and ensemble rule that produced the reported test results.

Usage
  python lens.py --self-test                      # no models needed
  python lens.py --models /content/models --image page.png --script arabic --out read.html
  python lens.py --models /content/models --kraken-bin /content/kenv/bin/kraken --share

--models must contain crnn1/, crnn2/ and trocr/ (Ajami) and may contain
arabic/crnn1/, arabic/crnn2/, arabic/crnn3/ and arabic/trocr/ (Arabic). Each
CRNN folder holds best.pt and alphabet.json; each TrOCR folder a saved model.
"""

from __future__ import annotations

import argparse
import os
import base64
import re
import difflib
import html
import io
import json
import subprocess
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from alignment import GAP, Hypothesis, apply_constraints, build_network, candidates  # noqa: E402

HEIGHT = 64            # CRNN input height, as trained
MAX_WIDTH = 1600       # CRNN width cap, as trained (dataset.MAX_WIDTH)
# Each script's evaluated final system. "order" is the member order that
# ensemble.py used when it scored the test set; it decides vote ties, so it is
# kept exactly. "bidi" marks CRNNs trained with --bidi-runs, whose output is in
# mirrored-scan order and is put back into logical order (bidi_order.py).
PROFILES = {
    "ajami": {"label": "Ajami (Hausa, Fulfulde)", "dir": "", "crnns": ["crnn1", "crnn2"],
              "order": ["crnn1", "crnn2", "trocr"], "max_len": 128, "bidi": False,
              "lang": "ha-Arab", "lm": {"order": 8, "lam": 0.15, "gap_bias": 0.0}},
    "arabic": {"label": "Arabic", "dir": "arabic", "crnns": ["crnn1", "crnn2", "crnn3"],
               "order": ["crnn1", "crnn3", "crnn2", "trocr"], "max_len": 256, "bidi": True,
               "lang": "ar", "lm": {"order": 8, "lam": 0.35, "gap_bias": -0.8},
               "tta": {"variants": ["light", "dark", "thick", "thin"], "variant_weight": 1.0,
                       "trocr_weight": 1.75}},
}
# "tta" (Arabic): each CRNN also reads every line four more ways (ink lighter,
# darker, thicker, thinner), giving the vote 16 readings; members are ordered
# as in the tuning: crnn1, crnn3, crnn2, trocr, then each CRNN's variants in
# that CRNN order. With the language model: half B 10.83% -> 9.16% CER.
# "lm": the character language model in the vote (lm_vote.py), built at start-up
# from <models>/lm_ajami.txt or <models>/lm_arabic.txt: the training
# transcriptions, one line each. Settings were tuned on validation
# half A (Arabic) or by cross-validation on half B (Ajami); the vote without
# it is the evaluated system, and the switch in the app restores it.
MAX_LINES = 60         # a public server must not be tied up by one huge image


# ---------------------------------------------------------------------------
# finding lines
# ---------------------------------------------------------------------------

def otsu_threshold(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def _runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    return runs


def segment_projection(img):
    """
    Lines from the horizontal ink profile. Suits clean, single-column scans.
    Vowel marks above and below a line often form thin bands of their own;
    those are merged into the nearest full line.
    """
    g = np.asarray(ImageOps.grayscale(img), dtype=np.uint8)
    H, W = g.shape
    ink = g <= otsu_threshold(g)            # Otsu's dark class is <= t
    # Dark scan borders and ruler edges are almost solid ink; handwriting,
    # even a heavy line, never is.
    ink[ink.mean(1) > 0.85] = False
    ink[:, ink.mean(0) > 0.85] = False
    prof = ink.sum(1).astype(float)
    if prof.max() == 0:
        return []
    k = max(3, H // 200)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    bands = _runs(prof > 0.10 * prof.max())
    if not bands:
        return []
    # Typical line height: thin bands of vowel marks must not drag this down,
    # so only bands at least 40% as tall as the tallest are counted.
    heights = [b - a for a, b in bands]
    med = float(np.median([h for h in heights if h >= 0.4 * max(heights)]))

    merged = True                             # merge thin bands into neighbours
    while merged and len(bands) > 1:
        merged = False
        for i, (a, b) in enumerate(bands):
            if b - a >= 0.45 * med:
                continue
            gap_up = a - bands[i - 1][1] if i > 0 else 1e9
            gap_dn = bands[i + 1][0] - b if i + 1 < len(bands) else 1e9
            j = i - 1 if gap_up <= gap_dn else i + 1
            if min(gap_up, gap_dn) > med:
                continue
            lo, hi = min(bands[i][0], bands[j][0]), max(bands[i][1], bands[j][1])
            bands[min(i, j)] = [lo, hi]
            del bands[max(i, j)]
            merged = True
            break
    bands = [b for b in bands if b[1] - b[0] >= 0.3 * med]

    boxes = []
    for a, b in bands:
        h = b - a
        y0, y1 = max(0, int(a - 0.30 * h)), min(H, int(b + 0.30 * h))
        cols = np.where(ink[y0:y1].sum(0) > 0)[0]
        if len(cols) == 0:
            continue
        pad = max(4, W // 200)
        boxes.append((max(0, cols[0] - pad), y0, min(W, cols[-1] + pad), y1))
    return boxes


def _bbox(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_segmentation(path):
    """Line boxes from Kraken's output: native JSON, ALTO or PageXML."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    boxes = []
    try:
        data = json.loads(raw)

        # Kraken's JSON describes text REGIONS (whole blocks) as well as
        # lines, and both carry a "boundary". Only the lines are wanted:
        # a region box would swallow every line inside it.
        def find_lines(o):
            if isinstance(o, dict):
                if isinstance(o.get("lines"), list):
                    return o["lines"]
                for k, v in o.items():
                    if k != "regions":
                        found = find_lines(v)
                        if found is not None:
                            return found
            elif isinstance(o, list):
                for v in o:
                    found = find_lines(v)
                    if found is not None:
                        return found
            return None

        for ln in find_lines(data) or []:
            if not isinstance(ln, dict):
                continue
            if isinstance(ln.get("boundary"), list) and ln["boundary"]:
                boxes.append(_bbox(ln["boundary"]))
            elif isinstance(ln.get("bbox"), (list, tuple)) and len(ln["bbox"]) == 4:
                boxes.append(tuple(float(v) for v in ln["bbox"]))
        return boxes
    except json.JSONDecodeError:
        pass
    root = ET.fromstring(raw)
    for el in root.iter():
        if not el.tag.endswith("TextLine"):
            continue
        poly = None
        for sub in el.iter():
            if sub.tag.endswith("Polygon") and sub.get("POINTS"):
                v = [float(x) for x in sub.get("POINTS").replace(",", " ").split()]
                poly = list(zip(v[0::2], v[1::2]))
                break
            if sub.tag.endswith("Coords") and sub.get("points"):
                poly = [tuple(map(float, p.split(","))) for p in sub.get("points").split()]
                break
        if poly:
            boxes.append(_bbox(poly))
        elif el.get("HPOS"):
            x, y = float(el.get("HPOS")), float(el.get("VPOS"))
            boxes.append((x, y, x + float(el.get("WIDTH")), y + float(el.get("HEIGHT"))))
    return boxes


def segment_kraken(img, kraken_bin, log=print):
    """Kraken's pretrained baseline segmenter, run as a separate program."""
    if not kraken_bin or not Path(kraken_bin).exists():
        return None
    try:
        import torch
        device = ["-d", "cuda:0"] if torch.cuda.is_available() else []
    except Exception:                                       # noqa: BLE001
        device = []
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "page.png"
        img.convert("RGB").save(page)
        t0 = time.time()
        for flag, ext in (("-n", "json"), ("-a", "xml"), ("-x", "xml")):
            out = Path(d) / f"seg.{ext}"
            cmd = [str(kraken_bin)] + device + ["-i", str(page), str(out), flag, "segment", "-bl"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except Exception as e:                          # noqa: BLE001
                log(f"kraken failed to run: {e!r}")
                return None
            if r.returncode == 0 and out.exists():
                try:
                    boxes = _parse_segmentation(out)
                except Exception as e:                      # noqa: BLE001
                    log(f"could not read kraken output ({flag}): {e!r}")
                    continue
                if boxes:
                    log(f"timings: line finding {time.time() - t0:.1f}s ({'GPU' if device else 'CPU'})")
                    return boxes
            else:
                log(f"kraken {flag}: exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    return None


def group_rows(boxes, overlap=0.5):
    """
    Merge boxes that share a row (e.g. the two halves of a verse found as
    separate lines) into one box per row, as in the corpus, and sort rows
    top to bottom.

    Safeguards: a box far taller than a typical line is a block, not a line,
    and is dropped; a box much narrower than a line is tall (0.6 of it or less)
    is a speck of dirt or a stray mark, and is dropped; and only boxes of similar height are ever
    merged, so no line can be swallowed by a larger box.
    """
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    med = heights[len(heights) // 2]
    boxes = [b for b in boxes
             if (b[3] - b[1]) <= 2.5 * med and (b[2] - b[0]) > 0.6 * med]    # keeps page numbers, short words
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        for r in rows:
            inter = min(r[3], b[3]) - max(r[1], b[1])
            hr, hb = r[3] - r[1], b[3] - b[1]
            if inter > overlap * min(hr, hb) and max(hr, hb) <= 2 * min(hr, hb):
                r[0], r[1] = min(r[0], b[0]), min(r[1], b[1])
                r[2], r[3] = max(r[2], b[2]), max(r[3], b[3])
                break
        else:
            rows.append(list(b))
    return [tuple(int(round(v)) for v in r) for r in sorted(rows, key=lambda r: r[1])]




def split_white_bands(img, min_strips=3):
    """
    A sheet of line images stacked with pure-white gaps (as made for the demos):
    each strip is one line. Returns the strips' boxes, trimmed to their content,
    or None for an ordinary page (real paper is never pure white edge to edge).
    """
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    white_row = (g >= 250).all(axis=1)
    boxes, y = [], 0
    H = len(white_row)
    while y < H:
        if white_row[y]:
            y += 1
            continue
        y0 = y
        while y < H and not white_row[y]:
            y += 1
        band = g[y0:y]
        cols = np.where((band < 250).any(axis=0))[0]
        if len(cols) and (y - y0) >= 8:
            boxes.append((int(cols[0]), y0, int(cols[-1]) + 1, y))
    gaps = white_row.sum()
    if len(boxes) >= min_strips and gaps >= 4 * (len(boxes) - 1):
        return boxes
    return None

def refine_boxes(img, boxes):
    """
    Clean the segmenter's lines before reading them (display and reading only;
    the evaluated system was given its lines).

      1. A box much shorter than a typical line that sits right against a line
         is a dot, a stroke or a sliver of it: drop it. Standing alone (a page
         number, a signature) it is kept.
      2. A box lying mostly inside another box is a fragment of that line:
         drop it.
      3. A box that stops short of the ink on its row is widened, left and
         right, to the ink, so no line is cut off before its end. Widening
         stops at a gap wider than about one and a half line heights, so it
         never runs into a second column.
    """
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    med = max(8, heights[len(heights) // 2])

    def hugs_a_line(b):
        """A short box within half a line height of a full-height line is a dot or a sliver of it."""
        for o in boxes:
            if o is b or (o[3] - o[1]) < 0.45 * med:
                continue
            gap = max(o[1] - b[3], b[1] - o[3], 0)          # vertical distance, 0 if they overlap
            side = min(b[2], o[2] + med) - max(b[0], o[0] - med)   # horizontal overlap, with a margin
            if gap <= 0.5 * med and side > 0:
                return True
        return False

    # short boxes go only when they hug a line; standing alone (a page number, a signature) they stay
    boxes = [b for b in boxes if (b[3] - b[1]) >= 0.45 * med or not hugs_a_line(b)]

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    kept = []
    for i, b in enumerate(boxes):
        inside = False
        for j, o in enumerate(boxes):
            if i == j or area(o) <= area(b):
                continue
            ix = max(0, min(b[2], o[2]) - max(b[0], o[0]))
            iy = max(0, min(b[3], o[3]) - max(b[1], o[1]))
            if ix * iy >= 0.6 * area(b):
                inside = True
                break
        if not inside:
            kept.append(b)

    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    ink = gray <= otsu_threshold(gray)       # <=: the threshold value itself belongs to the ink class
    W = ink.shape[1]
    max_gap = int(1.5 * med)
    out = []
    for x0, y0, x1, y1 in kept:
        h = y1 - y0
        band = ink[y0 + h // 5: max(y0 + h // 5 + 1, y1 - h // 5), :]   # the core of the line
        cols = band.sum(axis=0) > 0
        gap, x = 0, x1
        while x < W and gap <= max_gap:                    # widen to the right
            gap = 0 if cols[x] else gap + 1
            x += 1
        nx1 = min(W, x - gap)
        gap, x = 0, x0 - 1
        while x >= 0 and gap <= max_gap:                   # widen to the left
            gap = 0 if cols[x] else gap + 1
            x -= 1
        nx0 = max(0, x + 1 + gap)
        out.append((min(x0, nx0), y0, max(x1, nx1), y1))
    return sorted(out, key=lambda b: (b[1], -b[2]))


_LATIN = re.compile(r"[A-Za-z]")


def tidy_line(text, flags):
    """
    Make a reading look like text (display only; the evaluated system is not
    tidied): remove Latin letters, tokens made only of dots or other symbols,
    and bare single letters standing alone (except و, a word in its own
    right). Returns the tidied text and its flags, kept aligned.
    """
    pairs = [(c, f) for c, f in zip(text, flags) if not _LATIN.match(c)]
    tokens, cur = [], []
    for c, f in pairs:
        if c == " ":
            if cur:
                tokens.append(cur)
            cur = []
        else:
            cur.append((c, f))
    if cur:
        tokens.append(cur)

    def keep(tok):
        chars = "".join(c for c, _ in tok)
        letters = [c for c in chars if unicodedata.category(c).startswith("L")]
        marks = [c for c in chars if unicodedata.category(c).startswith("M")]
        digits = [c for c in chars if unicodedata.category(c) == "Nd"]
        if not letters and not digits:
            return False                                   # dots, dashes, stray symbols
        if len(letters) == 1 and not marks and not digits and letters[0] != "و":
            return False                                   # a lone bare letter
        return True

    tokens = [t for t in tokens if keep(t)]
    out_text, out_flags = [], []
    for k, t in enumerate(tokens):
        if k:
            out_text.append(" ")
            out_flags.append(False)
        out_text += [c for c, _ in t]
        out_flags += [f for _, f in t]
    return "".join(out_text), out_flags


def tidy_results(boxes, crops, results):
    """Tidy every line; drop lines left with fewer than two letters, unless they are a number (a page number)."""
    kept = []
    for b, c, r in zip(boxes, crops, results):
        text, flags = tidy_line(r["text"], r["unsure"])
        letters = sum(1 for ch in text if unicodedata.category(ch).startswith("L"))
        digits = sum(1 for ch in text if unicodedata.category(ch) == "Nd")
        if letters < 2 and digits < 2:
            continue
        kept.append((b, c, dict(r, text=text, unsure=flags)))
    return [k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept]


def looks_like_single_line(img):
    return img.height <= 400 and img.width >= 3 * img.height


# ---------------------------------------------------------------------------
# the ensemble, with per-character certainty
# ---------------------------------------------------------------------------


def _variant_thick(a):
    from scipy.ndimage import grey_erosion
    return grey_erosion(a, size=(2, 2))


def _variant_thin(a):
    from scipy.ndimage import grey_dilation
    return grey_dilation(a, size=(2, 2))


# Test-time variants of a 64-pixel greyscale line (applied before mirroring),
# exactly as used to tune the enhanced Arabic reading.
VARIANTS = {
    "thick": _variant_thick,
    "thin": _variant_thin,
    "dark": lambda a: (255.0 * (a / 255.0) ** 1.6).astype(np.uint8),
    "light": lambda a: (255.0 * (a / 255.0) ** 0.6).astype(np.uint8),
}


def ensemble_line(hyps, gap_conf=0.5):
    """
    The final system's rule (alignment.vote, confidence mode, then
    apply_constraints), also reporting which output characters were
    uncertain: proposed by only one model, or winning less than half the
    column's confidence.
    """
    net = build_network(hyps)
    chars, unsure = [], []
    for column in net:
        scores, support = {}, {}
        for ch, c in column:
            s = gap_conf if ch is GAP else (c if c is not None else 0.0)
            scores[ch] = scores.get(ch, 0.0) + s
            support[ch] = support.get(ch, 0) + 1
        best = max(candidates(column), key=lambda ch: scores[ch])
        if best is GAP:
            continue
        total = sum(scores.values()) or 1.0
        chars.append(best)
        unsure.append((support[best] == 1 and len(column) > 1)
                      or scores[best] / total < 0.5)
    raw = "".join(chars)
    text = apply_constraints(raw)
    flags = [True] * len(text)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=raw, b=text,
                                                       autojunk=False).get_opcodes():
        if tag == "equal":
            flags[j1:j2] = unsure[i1:i2]
    return text, flags


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

def _find(root, name):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise SystemExit(f"no {name} under {root}")
    return hits[0]


class _Script:
    """One script's members: its CRNNs and its TrOCR, loaded once."""

    def __init__(self, key, root, device, torch, models_root=None):
        from alphabet import Alphabet
        from crnn import CRNN
        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
        from train_trocr import configure_special_tokens
        self.key, self.profile, self.device, self.torch = key, PROFILES[key], device, torch
        self.crnns = []
        for name in self.profile["crnns"]:
            alphabet = Alphabet.load(_find(root / name, "alphabet.json"))
            ckpt = torch.load(_find(root / name, "best.pt"), map_location=device)
            model = CRNN(n_classes=alphabet.size_with_blank, input_height=HEIGHT).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self.crnns.append((name, model, alphabet))
            print(f"{key}: loaded {name} (epoch {ckpt.get('epoch')})")
        best = next((q.parent for q in sorted((root / "trocr").rglob("config.json"))
                     if q.parent.name == "best"), None) or _find(root / "trocr", "config.json").parent
        self.tok = AutoTokenizer.from_pretrained(best)
        self.proc = AutoImageProcessor.from_pretrained(best)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(best).to(device)
        configure_special_tokens(self.trocr, self.tok, self.profile["max_len"])
        self.trocr.eval()
        print(f"{key}: loaded trocr from {best} on {device}")
        self.lm = None
        # lm_<script>.txt at the top of the models folder (lm_ajami.txt, lm_arabic.txt);
        # lm_text.txt inside the script's own folder also works
        lm_text = Path(models_root or root) / f"lm_{key}.txt"
        if not lm_text.exists():
            lm_text = root / "lm_text.txt"
        if lm_text.exists():
            from lm_vote import KNLM
            texts = [t for t in lm_text.read_text(encoding="utf-8").splitlines() if t.strip()]
            self.lm = KNLM.from_texts(texts, order=self.profile["lm"]["order"])
            print(f"{key}: language model built from {len(texts)} training lines")
        else:
            print(f"{key}: no lm_{key}.txt, the language model option is off")

    def _read_crnn(self, model, alphabet, images, variant=None):
        from predict import ctc_decode_with_confidence
        from bidi_order import mirror_order_with_confidence
        torch = self.torch
        arrays, widths = [], []
        for im in images:
            g = im.convert("L")
            w = max(1, min(MAX_WIDTH, int(round(g.width * HEIGHT / g.height))))
            a = np.array(g.resize((w, HEIGHT), Image.LANCZOS), dtype=np.uint8)
            if variant is not None:
                a = VARIANTS[variant](a)
            arrays.append(np.ascontiguousarray(a[:, ::-1]))   # mirrored, as trained
            widths.append(w)
        batch = torch.ones(len(images), 1, HEIGHT, max(widths))   # white padding
        for i, a in enumerate(arrays):
            batch[i, 0, :, : a.shape[1]] = torch.from_numpy(a).float().div(255.0)
        with torch.no_grad():
            probs = model(batch.to(self.device)).exp().cpu().numpy()     # (T, B, C)
        out = []
        for i, w in enumerate(widths):
            T = model.output_length(w)
            text, conf = ctc_decode_with_confidence(probs[:T, i, :], alphabet.index_to_char)
            if self.profile["bidi"]:
                text, conf = mirror_order_with_confidence(text, conf)
            out.append((text, conf))
        return out

    def _read_trocr(self, images, chunk=16):
        from train_trocr import predict
        out = []
        for s in range(0, len(images), chunk):
            part = [im.convert("L").convert("RGB") for im in images[s:s + chunk]]
            pix = self.proc(part, return_tensors="pt").pixel_values
            batch = {"pixel_values": pix, "image_ids": [""] * len(part),
                     "texts": [""] * len(part), "langs": [""] * len(part)}
            for r in predict(self.trocr, self.tok, [batch], self.device, self.profile["max_len"]):
                out.append((r["prediction"], json.loads(r["confidence"])))
        return out

    def learn(self, lines):
        """Add corrected lines to this script's language model, for the rest of the session."""
        if self.lm is None or not lines:
            return 0
        self.lm.add_texts(lines)
        return len(lines)

    def read(self, images, use_lm=True):
        per_model = {name: self._read_crnn(model, alphabet, images)
                     for name, model, alphabet in self.crnns}
        per_model["trocr"] = self._read_trocr(images)
        order, weights = list(self.profile["order"]), None
        lm = self.lm if use_lm else None
        tta = self.profile.get("tta") if use_lm else None
        if tta:
            for name, model, alphabet in sorted(self.crnns, key=lambda c: order.index(c[0])):
                for v in tta["variants"]:
                    per_model[f"{name}_{v}"] = self._read_crnn(model, alphabet, images, variant=v)
                    order.append(f"{name}_{v}")
            weights = [tta["trocr_weight"] if m == "trocr" else 1.0 if "_" not in m
                       else tta["variant_weight"] for m in order]
        return combine(per_model, order, len(images), lm, self.profile["lm"], weights)


def combine(per_model, order, n, lm=None, lm_settings=None, weights=None):
    """
    The vote over each line's readings, members in evaluation order. With a
    language model, the vote is searched with it (lm_vote.lm_decode);
    without one, it is exactly the evaluated confidence vote.
    """
    results = []
    for i in range(n):
        hyps = []
        for name in order:
            text, conf = per_model[name][i]
            conf = (list(conf) + [0.5] * len(text))[: len(text)]
            hyps.append(Hypothesis(name, text, conf))
        if lm is not None:
            from lm_vote import lm_decode
            text, flags = lm_decode(hyps, lm, lam=lm_settings["lam"], gap_bias=lm_settings["gap_bias"],
                                    weights=weights)
        else:
            text, flags = ensemble_line(hyps)
        results.append({"text": text, "unsure": flags,           # shown: the four main readers
                        "models": {h.model: h.text for h in hyps if "_" not in h.model}})
    return results


class Reader:
    """Loads every script whose models are present; reads lists of line images."""

    def __init__(self, models_dir, device=None):
        import torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        root = Path(models_dir)
        self.scripts = {}
        for key, profile in PROFILES.items():
            sub = root / profile["dir"] if profile["dir"] else root
            if all((sub / name).exists() for name in profile["crnns"] + ["trocr"]):
                self.scripts[key] = _Script(key, sub, self.device, torch, models_root=root)
        if not self.scripts:
            raise SystemExit(f"no complete set of models under {root}")

    def read(self, images, script="ajami", use_lm=True):
        if not images:
            return []
        return self.scripts[script].read(images, use_lm)

    def learn(self, lines, script="ajami"):
        return self.scripts[script].learn(lines)


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

GOLD = (36, 87, 197)       # the box colour: lapis blue
INK = (16, 32, 63)         # ink navy


def draw_boxes(img, boxes):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 400)
    size = max(14, im.width // 45)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for n, (x0, y0, x1, y1) in enumerate(boxes, 1):
        d.rectangle([x0, y0, x1, y1], outline=GOLD, width=lw)
        label = str(n)
        tw = size * (0.6 * len(label) + 0.6)
        d.rectangle([x1 - tw, y0, x1, y0 + size * 1.3], fill=GOLD)
        d.text((x1 - tw + size * 0.3, y0 + size * 0.1), label, fill="white", font=font)
    return im


def _image_uri(img):
    """A line image as a lossless PNG at its own resolution (no downscaling)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def highlight(text, flags):
    """Disputed characters get blue dots beneath; a letter and its marks stay together."""
    out, k = [], 0
    while k < len(text):
        s = k
        k += 1
        while k < len(text) and unicodedata.category(text[k]).startswith("M"):
            k += 1
        piece = html.escape(text[s:k])
        out.append(f'<span class="nq-unsure">{piece}</span>' if any(flags[s:k]) else piece)
    return "".join(out) or '<span class="nq-faint">Nothing was read on this line.</span>'


FONTS = ("https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@400;700"
         "&family=Gentium+Book+Plus:ital,wght@0,400;0,700;1,400"
         "&family=Scheherazade+New:wght@400;500;600&family=Amiri&family=Noto+Naskh+Arabic:wght@400;600"
         "&family=Noto+Sans:wght@400;600&display=swap")

CSS = """
@import url('""" + FONTS + """');
:root, .light {
  --nq-folio: #FFFFFF;   --nq-leaf: #F5F8FD;   --nq-ink: #10203F;
  --nq-faded: #5A6B8C;   --nq-rubric: #2457C5; --nq-gold: #2457C5;
  --nq-gold-wash: rgba(36, 87, 197, .14);     --nq-rule: #D6E0F0;
}
.dark {
  --nq-folio: #0E1730;   --nq-leaf: #16223F;   --nq-ink: #EAF0FA;
  --nq-faded: #A9B8D6;   --nq-rubric: #8FB3FF; --nq-gold: #8FB3FF;
  --nq-gold-wash: rgba(143, 179, 255, .22);   --nq-rule: #2B3A5E;
}
.gradio-container { background: var(--nq-folio) !important; color: var(--nq-ink);
  font-family: 'Gentium Book Plus', Georgia, serif; max-width: 1180px !important; }
.gradio-container button, .gradio-container input, .gradio-container label,
.gradio-container .block-label, .gradio-container span.svelte-1gfkn6j {
  font-family: 'Noto Sans', system-ui, sans-serif; }

/* the masthead: the wordmark is the one bold thing on the page */
#nq-head { padding: 28px 4px 6px; }
#nq-head .nq-mark { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
#nq-head .nq-ar { font-family: 'Aref Ruqaa', 'Noto Naskh Arabic', serif; font-size: 4.6rem;
  line-height: 1; color: var(--nq-gold); direction: rtl; }
#nq-head .nq-lat { font-size: 2.6rem; line-height: 1.05; color: var(--nq-ink); letter-spacing: .01em; }
#nq-head .nq-lat small { display: block; font-size: 1.02rem; font-style: italic; color: var(--nq-faded);
  letter-spacing: 0; margin-top: 6px; }
#nq-head .nq-lede { max-width: 62ch; font-size: 1.12rem; line-height: 1.6; margin: 14px 0 0; }

/* the reading */
.nq-read { font-family: 'Gentium Book Plus', Georgia, serif; color: var(--nq-ink); }
.nq-read .nq-summary { font-size: 1rem; line-height: 1.55; margin: 0 0 18px; color: var(--nq-faded);
  max-width: 64ch; }
.nq-line { display: grid; grid-template-columns: 2.6rem minmax(0, 1fr); gap: 4px 14px; padding: 16px 0;
  border-top: 1px solid var(--nq-rule); }
.nq-body { min-width: 0; }
.nq-line:first-of-type { border-top: 0; }
.nq-num { font-size: 1.35rem; color: var(--nq-rubric); text-align: right;
  padding-top: 2px; font-variant-numeric: oldstyle-nums; }
.nq-line img { max-width: 100%; height: auto; display: block; border-radius: 2px;
  box-shadow: 0 0 0 1px var(--nq-rule); background: #fff; }
.nq-text { font-family: 'Scheherazade New', 'Amiri', 'Noto Naskh Arabic', serif; font-size: 1.95rem; line-height: 2.1;
  direction: rtl; text-align: right; margin: 6px 0 0; overflow: visible; white-space: normal;
  overflow-wrap: break-word; }
.nq-unsure { color: var(--nq-gold); text-decoration: underline dotted var(--nq-gold);
  text-decoration-thickness: 3px; text-underline-offset: .45em; text-decoration-skip-ink: none; }
.nq-models { display: grid; grid-template-columns: 4.5rem 1fr; gap: 0 10px; margin-top: 4px;
  align-items: baseline; font-size: .9rem; color: var(--nq-faded); }
.nq-models .nq-text { font-size: 1.4rem; line-height: 1.9; margin: 0; color: var(--nq-faded); }
.nq-faint { color: var(--nq-faded); font-style: italic; }
.nq-enh { font-family: 'Scheherazade New', 'Amiri', 'Noto Naskh Arabic', serif; font-size: 1.75rem;
  line-height: 2; direction: rtl; text-align: right; margin: 6px 0 0; color: var(--nq-ink);
  border-right: 3px solid var(--nq-gold); padding-right: 12px; background: var(--nq-gold-wash);
  border-radius: 6px 0 0 6px; white-space: normal; overflow-wrap: break-word; }
.nq-enh-tag { font-family: 'Noto Sans', sans-serif; font-size: .72rem; letter-spacing: .03em;
  color: #fff; background: var(--nq-gold); border-radius: 4px; padding: 1px 7px; margin-left: 10px;
  vertical-align: middle; }
.dark .nq-enh-tag { color: #0E1730; }
.nq-boko { font-family: 'Gentium Book Plus', Georgia, serif; font-size: 1.15rem; line-height: 1.6;
  margin: 4px 0 0; direction: ltr; text-align: left; unicode-bidi: isolate; white-space: normal;
  color: var(--nq-ink); }
.nq-en { font-family: 'Gentium Book Plus', Georgia, serif; font-size: 1.08rem; line-height: 1.6;
  margin: 2px 0 0; direction: ltr; text-align: left; unicode-bidi: isolate; white-space: normal;
  font-style: italic; color: var(--nq-faded); }

/* the reading in progress: the dots that tell ب, ت and ث apart, in turn */
.nq-loading { display: flex; flex-direction: column; align-items: center; gap: 16px; padding: 56px 0 48px;
  color: var(--nq-faded); font-size: 1.05rem; }
.nq-letter { width: 112px; height: 75px; overflow: visible; }
.nq-letter path { fill: none; stroke: var(--nq-ink); stroke-width: 4; stroke-linecap: round; opacity: .85; }
.nq-letter circle { fill: var(--nq-gold); opacity: 0; transform-box: fill-box; transform-origin: center; }
.nq-letter .b1 { animation: nq-b 2.4s infinite; }
.nq-letter .t1, .nq-letter .t2 { animation: nq-t 2.4s infinite; }
.nq-letter .th1, .nq-letter .th2, .nq-letter .th3 { animation: nq-th 2.4s infinite; }
.nq-letter .t2, .nq-letter .th2 { animation-delay: .06s; }
.nq-letter .th3 { animation-delay: .12s; }
@keyframes nq-b  { 0%, 5% { opacity: 0; transform: translateY(-3px) } 10%, 30% { opacity: 1; transform: none }
                   36%, 100% { opacity: 0 } }
@keyframes nq-t  { 0%, 36% { opacity: 0; transform: translateY(-3px) } 42%, 62% { opacity: 1; transform: none }
                   68%, 100% { opacity: 0 } }
@keyframes nq-th { 0%, 68% { opacity: 0; transform: translateY(-3px) } 74%, 94% { opacity: 1; transform: none }
                   100% { opacity: 0 } }
.nq-stage { margin: 0; letter-spacing: .01em; animation: nq-fade .5s ease-out; }
@keyframes nq-fade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
  .nq-letter circle { animation: none !important; } .nq-letter .b1 { opacity: 1; }
  .nq-stage { animation: none; } }

/* calmer surfaces */
.gradio-container .block { border-radius: 12px !important; box-shadow: 0 1px 3px rgba(16, 32, 63, .06); }
#nq-read-btn { border-radius: 10px !important; min-height: 48px; font-weight: 600 !important; }
#nq-teach { border-radius: 10px !important; }
#nq-stop { border-radius: 10px !important; min-height: 48px; background: transparent !important;
  border: 1px solid var(--nq-rule) !important; color: var(--nq-faded) !important; }
#nq-stop:hover { border-color: var(--nq-gold) !important; color: var(--nq-gold) !important; }
#nq-head .nq-lede { color: var(--nq-faded); }
.nq-empty { padding: 28px 4px; color: var(--nq-faded); font-size: 1.08rem; line-height: 1.6; max-width: 46ch; }

/* the desk: quiet controls, one inked action */
#nq-read-btn { background: var(--nq-gold) !important; color: #FFFFFF !important;
  border: 0 !important; font-size: 1.02rem !important; letter-spacing: .01em; }
.dark #nq-read-btn { color: #0E1730 !important; }
#nq-read-btn::before { content: ""; display: inline-block; width: .55em; height: .55em;
  border-radius: 50%; background: currentColor; margin-right: .6em; vertical-align: .08em; }
#nq-read-btn:focus-visible { outline: 3px solid var(--nq-gold) !important; outline-offset: 2px; }
#nq-copy-en textarea { font-family: 'Gentium Book Plus', Georgia, serif !important;
  font-size: 1.1rem !important; line-height: 1.6 !important; direction: ltr !important; text-align: left !important; }
#nq-copy textarea { font-family: 'Scheherazade New', 'Amiri', 'Noto Naskh Arabic', serif !important;
  font-size: 1.55rem !important; line-height: 2 !important; }
#nq-foot { font-size: .98rem; line-height: 1.6; color: var(--nq-faded); max-width: 70ch; }
#nq-foot ol { padding-left: 1.3em; }
#nq-foot strong { color: var(--nq-ink); font-weight: 700; }
@media (max-width: 640px) {
  #nq-head .nq-ar { font-size: 3.4rem; }  #nq-head .nq-lat { font-size: 2rem; }
  .nq-text { font-size: 1.6rem; }
}
"""


def render_html(crops, results, seconds, method, show_models, script="ajami", use_lm=True):
    lang = PROFILES.get(script, PROFILES["ajami"])["lang"]
    n = len(results)
    refined = sum(1 for r in results if r.get("enhanced") not in (None, r["text"]))
    parts = [f'<div class="nq-read"><p class="nq-summary">Read {n} line{"s" if n != 1 else ""} '
             f'in {seconds:.1f} seconds; lines found by {html.escape(method)}'
             f'{"" if use_lm else ", using the tested system only"}. A letter in blue with '
             f'dots beneath it means the recognisers disagreed about it: check those letters against '
             f'the manuscript.'
             + (f' <strong>The final check could not run this time</strong> '
                f'({html.escape(results[0]["check_note"])}).' if results and results[0].get("check_note") else '')
             + (f' Beneath each line, <strong>Enhanced</strong> is the final, checked version ({refined} '
                f'line{"s" if refined != 1 else ""} refined), with the letters it changed marked; the copy box '
                f'holds the Enhanced lines.' if any(r.get("enhanced") for r in results) else '')
             + (' The English (and, for Ajami, the Latin script) is experimental and still being improved: '
                'it is a machine translation of the reading, best read as a gist.'
                if any(r.get("boko") or r.get("english") for r in results) else '')
             + '</p>']
    for k, (crop, r) in enumerate(zip(crops, results), 1):
        models = ""
        if show_models:
            models = '<div class="nq-models">' + "".join(
                f'<span>{html.escape(name)}</span>'
                f'<p class="nq-text" lang="{lang}">{html.escape(t).replace(" ", "&nbsp;") or "—"}</p>'
                for name, t in r["models"].items()) + "</div>"
        extra = ""
        if r.get("enhanced"):
            changed = carry_flags(r["text"], [False] * len(r["text"]), r["enhanced"])
            extra += (f'<p class="nq-enh" lang="{lang}" dir="rtl"><span class="nq-enh-tag">Enhanced</span>'
                      f'{highlight(r["enhanced"], changed)}</p>')
        if r.get("boko"):
            extra += f'<p class="nq-boko" lang="ha" dir="ltr">{html.escape(r["boko"])}</p>'
        if r.get("english"):
            extra += f'<p class="nq-en" lang="en" dir="ltr">{html.escape(r["english"])}</p>'
        parts.append(f'<div class="nq-line"><div class="nq-num">{k}</div><div class="nq-body">'
                     f'<img src="{_image_uri(crop)}" alt="Line {k} of the page">'
                     f'<p class="nq-text" lang="{lang}" dir="rtl">{highlight(r["text"], r["unsure"])}</p>'
                     f'{extra}{models}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def loading(stage="Reading the page…"):
    """The reading in progress: a letter's bowl whose dots become ب, then ت, then ث."""
    return ('<div class="nq-loading" role="status" aria-live="polite">'
            '<svg class="nq-letter" viewBox="0 0 96 64" aria-hidden="true">'
            '<path d="M10 26 Q 12 44 48 44 Q 84 44 86 26"/>'
            '<circle class="b1" cx="48" cy="56" r="4.5"/>'
            '<circle class="t1" cx="41" cy="16" r="4.5"/><circle class="t2" cx="55" cy="16" r="4.5"/>'
            '<circle class="th1" cx="48" cy="8" r="4.5"/><circle class="th2" cx="41" cy="20" r="4.5"/>'
            '<circle class="th3" cx="55" cy="20" r="4.5"/></svg>'
            f'<p class="nq-stage">{html.escape(stage)}</p></div>')


LOADING = loading()

STOPPED = ('<div class="nq-empty">Stopped. Change what you need (the script, how to find the lines, or '
           'the image) and read the page again.</div>')


class Stopped(Exception):
    """Raised at the next step boundary when the reader presses Stop."""

EMPTY = ('<div class="nq-empty">Upload a photo or scan of a manuscript page, or of a single line, '
         'choose its script, and read the page. The transcription appears here, line by line.</div>')


def standalone_page(body):
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Iʿjām</title><style>{CSS} body{{max-width:1000px;margin:24px auto;"
            "padding:0 16px;background:var(--nq-folio)}}</style></head><body>"
            "<h1 style=\"font-family:'Gentium Book Plus',serif\">Iʿjām</h1>"
            f"{body}</body></html>")



# ---------------------------------------------------------------------------
# Latin script and English (Ajami, experimental)
# ---------------------------------------------------------------------------
_NLLB = {}
_CANCELS = {}                       # one Stop switch per browser session


def cancel_flag(request=None):
    import threading
    key = getattr(request, "session_hash", None) or "default"
    return _CANCELS.setdefault(key, threading.Event())


def translate_lines(lines, src_lang="hau_Latn", device="cpu"):
    """
    English with NLLB-200 (600M), loaded on first use. src_lang: hau_Latn for
    Latin-script Hausa; apc_Arab (North Levantine Arabic) for Muharaf, whose
    letters are mostly in Lebanese Arabic.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if "model" not in _NLLB:
        name = "facebook/nllb-200-distilled-600M"
        _NLLB["tok"] = AutoTokenizer.from_pretrained(name, src_lang=src_lang)
        _NLLB["model"] = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    tok, model = _NLLB["tok"], _NLLB["model"]
    tok.src_lang = src_lang
    out = []
    for i in range(0, len(lines), 16):
        batch = tok(lines[i:i + 16], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                                 max_new_tokens=64)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def add_latin_and_english(results, device="cpu", log=print, english=True, script="ajami"):
    """
    Ajami: Latin script (Boko) and English. With an API key, Claude writes the
    standard Latin spelling from the Ajami reading and a rough rule-based
    transliteration, then translates (a Hausa speaker scored this 4.8/5 for
    the Latin and 4.8/5 for the English on 30 held-out lines); without a key,
    the rules and NLLB-200 are used.
    Arabic: English from the Enhanced line (or the reading); Claude when a key
    is present, otherwise NLLB-200.
    """
    if script == "ajami":
        from ajami2boko import to_boko
        for r in results:
            r["boko"] = to_boko(r["text"])
        source, src_lang = "boko", "hau_Latn"
    else:
        source, src_lang = "text", "apc_Arab"
    if not english:
        return results
    if ai_available():
        try:
            if script == "ajami":
                latin_english_claude(results, log)
            else:
                english_claude(results, log)
            if any(r.get("english") for r in results):
                return results
        except Exception as e:                            # noqa: BLE001
            log(f"Claude translation unavailable, using NLLB: {e!r}")
    pick = (lambda r: r.get("enhanced") or r.get("text")) if source == "text" else (lambda r: r.get(source))
    todo = [k for k, r in enumerate(results) if pick(results[k])]
    try:
        en = translate_lines([pick(results[k]) for k in todo], src_lang, device)
        for k, e in zip(todo, en):
            results[k]["english"] = e
    except Exception as e:                                # noqa: BLE001
        log(f"translation unavailable: {e!r}")
    return results


# Claude for the Latin script and English (Ajami) and for English (Arabic).
# The Ajami instructions are the ones a Hausa speaker reviewed (Latin 4.84/5,
# English 4.78/5 on 30 held-out Infiraji lines).
AJ_LATIN_SYSTEM = (
    "You are an expert in Hausa and Fulfulde and in Ajami (these languages written in Arabic script). The lines "
    "come, in order, from one page of a West African Ajami manuscript, in Hausa or Fulfulde; decide which from the "
    "text. For each line you receive the machine's Ajami reading (it may contain recognition errors) and a rough, "
    "letter-by-letter Latin transliteration made by rules, which often joins or splits words wrongly and may contain "
    "spelling slips. For each line: (1) write it in the standard Latin spelling of that language (Boko for Hausa; "
    "standard Fulfulde spelling), with correct word division and the letters ɓ, ɗ, ƙ, 'y, ƴ, ŋ or ñ where the "
    "language needs them, writing Arabic phrases and names as that language writes them; stay faithful to the "
    "Ajami and add no words or meaning that are not there; if a word is unclear, choose the closest real word to "
    "the reading and mark it with (?). (2) Translate the line into natural English, faithful to the original; mark "
    "uncertain parts with (?). Return exactly {n} lines, numbered '1.' to '{n}.', each in the form: Latin ||| English")

AR_ENGLISH_SYSTEM = (
    "You are an expert translator of handwritten Arabic letters and documents from Lebanon (19th to 21st century, "
    "often in Lebanese colloquial Arabic). The lines come, in order, from one page. Translate each line into "
    "natural English, faithful to the Arabic: keep names as names, do not add content, and mark unclear parts "
    "with (?). Return exactly {n} lines, numbered '1.' to '{n}.', each containing only the English.")


def _claude_text(system, body, max_tokens=4000):
    import anthropic
    msg = anthropic.Anthropic().messages.create(model=AI_MODEL, max_tokens=max_tokens, system=system,
                                                messages=[{"role": "user", "content": body}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def parse_latin_english(text, n):
    got = {}
    for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*\|\|\|\s*(.*)$", text, flags=re.M):
        got[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return [got.get(i) for i in range(1, n + 1)]


def latin_english_claude(results, log=print):
    body = "\n".join(f"{i}. Ajami: {r['text']}\n   Rough Latin: {r.get('boko', '')}" for i, r in enumerate(results, 1))
    out = parse_latin_english(_claude_text(AJ_LATIN_SYSTEM.format(n=len(results)), body), len(results))
    got = 0
    for r, o in zip(results, out):
        if o and o[0]:
            r["boko"], r["english"] = o[0], o[1]
            got += 1
    log(f"Latin and English by Claude: {got} of {len(results)} lines")


def english_claude(results, log=print):
    lines = [r.get("enhanced") or r["text"] for r in results]
    body = "\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1))
    text = _claude_text(AR_ENGLISH_SYSTEM.format(n=len(lines)), body)
    got = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.*)$", text, flags=re.M)}
    for i, r in enumerate(results, 1):
        if got.get(i):
            r["english"] = got[i]
    log(f"English by Claude: {sum(1 for i in range(1, len(results) + 1) if got.get(i))} of {len(results)} lines")


# ---------------------------------------------------------------------------
# AI-corrected reading (Arabic, experimental): Claude, faithful + evidence
# ---------------------------------------------------------------------------
# Exactly the instructions measured on validation half B (544 lines): with
# Claude Opus 5.5, CER 9.16% -> 7.65% on top of the enhanced reading (202
# lines better, 15 worse); a free "readable" rewrite was far worse, so it is
# not offered. Needs ANTHROPIC_API_KEY in the environment; shown beside the
# reading, never replacing it.
AI_MODEL = "claude-opus-5-5"
AI_SYSTEM = (
    "You are helping digitise handwritten Arabic letters and documents from Lebanon, written between the "
    "early 19th and early 21st centuries, mostly in Ruq'ah script and often in Lebanese colloquial Arabic. "
    "You receive a machine's reading of each line of one page. Correct only clear recognition errors: wrong "
    "letters (especially dots), wrongly split or joined words, misread names when the evidence supports it. "
    "Preserve the writer's own spelling and dialect (for example كتير، شو، الي، انشالله): do not convert to "
    "Modern Standard Arabic, and do not add hamzas, vowel marks or punctuation that the reading lacks. If "
    "unsure, keep the machine's reading. For each line you also get the readings of four separate "
    "recognisers; prefer letters and words that the evidence supports. Return exactly {n} lines, numbered "
    "'1.' to '{n}.', each containing only the Arabic text, nothing else.")


def ai_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ai_prompt(results):
    names = ["crnn1", "crnn3", "crnn2", "trocr"]
    body = []
    for i, r in enumerate(results, 1):
        m = r.get("models", {})
        body.append(f"{i}. reading: {r['text']}\n" + "\n".join(
            f"   recogniser {chr(65 + k)}: {m.get(n, '')}" for k, n in enumerate(names)))
    return AI_SYSTEM.format(n=len(results)), "\n".join(body)


def parse_numbered(text, fallback):
    got = {int(m.group(1)): m.group(2).strip()
           for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.*)$", text, flags=re.M)}
    return [got.get(i, f) or f for i, f in enumerate(fallback, 1)]



def carry_flags(old, old_flags, new):
    """
    Flags for a corrected line: letters kept from the reading keep their flag;
    letters the final check changed or added are flagged, so the reader sees
    where to look.
    """
    import difflib
    flags = [True] * len(new)
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                flags[j1 + k] = bool(old_flags[i1 + k]) if i1 + k < len(old_flags) else False
    for k, ch in enumerate(new):
        if ch == " ":
            flags[k] = False
    return flags

def _check_once(system, body, n, fallback):
    """One final-check request; returns the corrected lines (the reading where it gave none)."""
    import anthropic
    client = anthropic.Anthropic()
    request = dict(model=AI_MODEL, max_tokens=4000, system=system,
                   messages=[{"role": "user", "content": body}])
    try:                                   # ask for temperature 0 where the model allows it
        msg = client.messages.create(temperature=0, **request)
    except (TypeError, anthropic.BadRequestError):   # this model and library do not take one
        msg = client.messages.create(**request)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return parse_numbered(text, fallback)


def combine_checks(versions):
    """
    Several independent checks of one line, combined like the recognisers:
    identical majority wins outright; otherwise a letter-by-letter vote over
    the aligned versions, so a slip made by a single check is outvoted.
    """
    from collections import Counter
    versions = [v for v in versions if v]
    if not versions:
        return ""
    best, count = Counter(versions).most_common(1)[0]
    if count * 2 > len(versions) or len(versions) == 1:
        return best
    hyps = [Hypothesis(f"check{k}", v, [0.9] * len(v)) for k, v in enumerate(versions)]
    from alignment import build_network, vote
    return vote(build_network(hyps))


def add_ai_correction(results, log=print, checks=3):
    """
    The final check: `checks` independent requests to Claude run in parallel,
    combined per line by vote (self-consistency), so one run's random slip
    does not reach the reading. Sets r['ai'] on each line.
    """
    if not results:
        return results
    from concurrent.futures import ThreadPoolExecutor
    system, body = ai_prompt(results)
    fallback = [r["text"] for r in results]
    runs, errors = [], []
    with ThreadPoolExecutor(checks) as ex:
        futures = [ex.submit(_check_once, system, body, len(results), fallback) for _ in range(checks)]
        for f in futures:
            try:
                runs.append(f.result())
            except Exception as e:                        # noqa: BLE001
                errors.append(e)
    if not runs:
        e = errors[0]
        log(f"AI correction unavailable: {e!r}")
        results[0]["check_note"] = f"{type(e).__name__}: {str(e)[:160]}"
        return results
    if errors:
        log(f"final check: {len(errors)} of {checks} requests failed; combining the other {len(runs)}")
    for k, r in enumerate(results):
        r["ai"] = combine_checks([run[k] for run in runs]) or r["text"]
    return results


# ---------------------------------------------------------------------------
# one page, end to end
# ---------------------------------------------------------------------------

def read_page(reader, img, mode="auto", kraken_bin=None, log=print, script="ajami", use_lm=True,
              tidy=True, translate=False, ai=False, progress=None):
    say = progress or (lambda msg: None)
    say("Preparing the image…")
    img = ImageOps.exif_transpose(img).convert("RGB")
    W, H = img.size
    if mode == "single" or (mode == "auto" and looks_like_single_line(img)):
        boxes, method = [(0, 0, W, H)], "treating the image as one line"
    else:
        say("Finding the lines…")
        boxes, method = None, None
        if mode == "auto":
            strips = split_white_bands(img)
            if strips:
                boxes, method = strips, "the white gaps between line strips"
        if not boxes and mode in ("auto", "kraken"):
            found = segment_kraken(img, kraken_bin, log=log)
            if found:
                boxes, method = found, "Kraken's page segmenter"
        if not boxes:
            boxes, method = segment_projection(img), "the projection segmenter"
        if method != "the white gaps between line strips":
            boxes = group_rows(boxes)
            if tidy:
                boxes = refine_boxes(img, boxes)
    boxes = [(max(0, x0), max(0, y0), min(W, x1), min(H, y1)) for x0, y0, x1, y1 in boxes
             if x1 - x0 > 10 and y1 - y0 > 8][:MAX_LINES]
    crops = [img.crop(b) for b in boxes]
    t_start = time.time()
    timings = {}
    t0 = time.time()
    say(f"Reading {len(crops)} line{'s' if len(crops) != 1 else ''}…")
    results = reader.read(crops, script, use_lm)
    timings["recognition"] = time.time() - t0
    if tidy:
        boxes, crops, results = tidy_results(boxes, crops, results)
    if ai and use_lm and script == "arabic" and not ai_available():
        log("final check skipped: no ANTHROPIC_API_KEY in the environment")
        if results:
            results[0]["check_note"] = "no API key was found when the app started"
    if ai and use_lm and script == "arabic" and ai_available():
        t0 = time.time()
        say("Checking the reading…")
        add_ai_correction(results, log)
        for r in results:                    # the checked line is shown as "Enhanced", beside the reading
            if r.get("ai"):
                r["enhanced"] = r.pop("ai")
        changed = sum(1 for r in results if r.get("enhanced") not in (None, r["text"]))
        log(f"final check: {changed} of {len(results)} lines corrected")
        for k, r in enumerate(results[:3], 1):                 # the first lines, before and after
            if r.get("enhanced") not in (None, r["text"]):
                log(f"  line {k}: {r['text']}  ->  {r['enhanced']}")
        timings["final check"] = time.time() - t0
    if translate:
        t0 = time.time()
        say("Translating…")
        add_latin_and_english(results, getattr(reader, "device", "cpu"), log, script=script)
        timings["translation"] = time.time() - t0
    seconds = time.time() - t_start
    say("Laying out the page…")
    log("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    text = "\n".join(r.get("enhanced") or r["text"] for r in results)   # the copy box: Enhanced lines
    return boxes, crops, results, seconds, method, text


# ---------------------------------------------------------------------------
# web interface
# ---------------------------------------------------------------------------

def _make(cls, *args, **kwargs):
    """Create a Gradio component, dropping options this Gradio version lacks."""
    while True:
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            bad = next((k for k in kwargs if f"'{k}'" in str(e)), None)
            if bad is None:
                raise
            kwargs.pop(bad)


def ijam_theme():
    """Gradio theme in lapis blue and white; None if this Gradio lacks theming."""
    try:
        import gradio as gr
        theme = gr.themes.Base(
            font=[gr.themes.GoogleFont("Gentium Book Plus"), "Georgia", "serif"],
            font_mono=["ui-monospace", "monospace"],
            radius_size=gr.themes.sizes.radius_sm,
        )
        settings = dict(
            body_background_fill="#FFFFFF", body_background_fill_dark="#0E1730",
            body_text_color="#10203F", body_text_color_dark="#EAF0FA",
            block_background_fill="#F5F8FD", block_background_fill_dark="#16223F",
            block_border_color="#D6E0F0", block_border_color_dark="#2B3A5E",
            block_label_text_color="#5A6B8C", block_label_text_color_dark="#A9B8D6",
            input_background_fill="#FFFFFF", input_background_fill_dark="#121C36",
            border_color_accent="#2457C5", border_color_accent_dark="#8FB3FF",
            color_accent_soft="rgba(36,87,197,.14)", color_accent_soft_dark="rgba(143,179,255,.2)",
            button_primary_background_fill="#2457C5", button_primary_background_fill_dark="#8FB3FF",
            button_primary_background_fill_hover="#1B46A3",
            button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0E1730",
        )
        for key, value in settings.items():          # skip any this version lacks
            try:
                theme.set(**{key: value})
            except Exception:                         # noqa: BLE001
                pass
        return theme
    except Exception:                                 # noqa: BLE001
        return None


def favicon_path():
    """A blue dot on white: the app's icon, also used when it is installed."""
    path = Path(tempfile.gettempdir()) / "ijam_icon.png"
    if not path.exists():
        im = Image.new("RGB", (512, 512), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.ellipse([136, 136, 376, 376], fill=GOLD)
        im.save(path)
    return str(path)


def launch(demo, **kwargs):
    """
    Start the app. Gradio 6 takes styling in launch(), older versions in
    Blocks() (where build_app already passed it). pwa=True lets visitors install
    Iʿjām as an app; options this Gradio lacks are dropped one by one.
    """
    extras = {"css": CSS, "theme": ijam_theme(), "pwa": True, "favicon_path": favicon_path()}
    if extras["theme"] is None:
        extras.pop("theme")
    while True:
        try:
            return demo.launch(**kwargs, **extras)
        except TypeError as e:
            bad = next((k for k in extras if f"'{k}'" in str(e)), None)
            if bad is None:
                raise
            extras.pop(bad)


HEAD = """
<div id="nq-head-inner">
  <div class="nq-mark">
    <span class="nq-ar" lang="ar">إعجام</span>
    <span class="nq-lat">Iʿjām<small>the dots that tell letters apart</small></span>
  </div>
  <p class="nq-lede">Reads handwritten Ajami and Arabic manuscripts line by line, and marks every
  letter it is unsure of. In these scripts a single dot changes the letter, so Iʿjām shows you which
  dots to check.</p>
</div>
"""

FOOT = """
<p><strong>How Iʿjām reads a page</strong></p>
<ol>
  <li>It finds the lines of text on the page.</li>
  <li>Several recognisers of different kinds read every line: three for Ajami, four for Arabic.</li>
  <li>A vote combines their readings letter by letter, trusting each by its confidence.</li>
  <li>For Arabic, each recogniser also reads every line a few more ways (ink lighter, darker, thicker,
  thinner), so the vote has more readings to choose from.</li>
  <li>A language model learned from the training transcriptions helps the vote prefer real words.
  It only chooses among letters the recognisers proposed; it never invents text.</li>
  <li>Letters the recognisers disputed are shown in blue, with dots beneath them.</li>
  <li>You can correct the text and press <em>Teach Iʿjām my corrections</em>: the language model learns
  your corrected words at once, so later readings prefer them.</li>
  <li>For Arabic, a final check passes the combined reading, together with all four recognisers' readings,
  to a large language model (Claude, by Anthropic), which repairs clear recognition errors from that
  evidence while keeping the writer's own spelling and dialect. On held-out lines this lowered the
  character error rate from 9.16% to 7.65%.</li>
  <li>Optionally, each line is given in Latin script (for Ajami) and translated to English. With an API key,
  a large language model (Claude) writes the standard Latin spelling from the reading and translates it; a Hausa
  speaker scored this 4.8 out of 5 for both on held-out lines. Without a key, rules and the open NLLB-200
  translation model are used. Both remain experimental: uncertain parts are marked with (?).</li>
</ol>
<p><strong>How well it reads.</strong> On manuscript lines held out from training it made fewer
errors than the published systems: a character error rate of 10.42% on Ajami, where the published
benchmark is 10.66%, and 10.12% on historical Arabic, where the dataset authors' system reached
12.36% on the same lines. With the enhanced reading, validation error fell further, from 10.83% to
9.16% on Arabic and from 9.80% to 9.57% on Ajami.</p>
<p>A research prototype from a final-year project in the Department of Computer Science, University
of Ibadan, by Bashir Sekinat Omotoke, supervised by Dr O. A. Abiola. It reads clean scans similar to
its training manuscripts best; check important readings against the original.</p>
"""


def build_app(reader, kraken_bin):
    import gradio as gr
    import warnings

    # Offer only what this server can actually do.
    have_kraken = bool(kraken_bin) and Path(kraken_bin).exists()
    modes = {"Automatic": "auto"}
    if have_kraken:
        modes["Kraken segmenter"] = "kraken"
    modes["Simple segmenter"] = "projection"
    modes["The image is one line"] = "single"
    available = getattr(reader, "scripts", {"ajami": None})
    scripts = {PROFILES[k]["label"]: k for k in PROFILES if k in available}

    def run(image, script_label, mode_label, trans_on, lm_on, tidy_on, show_models,
            request: gr.Request = None):
        """Streams the loader with the current stage, then the page."""
        import queue
        import threading
        if image is None:
            yield None, EMPTY, "", gr.update(value="", visible=False), None, ""
            return
        stages, box = queue.Queue(), {}
        cancel = cancel_flag(request)
        cancel.clear()

        def report(stage):
            if cancel.is_set():
                raise Stopped()
            stages.put(stage)

        def work():
            try:
                box["out"] = _run(image, script_label, mode_label, trans_on, lm_on, tidy_on, show_models,
                                  progress=report)
            except Stopped:
                box["out"] = (None, STOPPED, "", "")
            except Exception as e:                       # noqa: BLE001
                box["out"] = (None, f'<div class="nq-empty">Reading stopped with an error: '
                                    f'{html.escape(repr(e))}</div>', "")
            finally:
                stages.put(None)

        threading.Thread(target=work, daemon=True).start()
        yield None, loading("Preparing the image…"), "", gr.update(value="", visible=False), None, ""
        while True:
            stage = stages.get()
            if stage is None:
                break
            yield None, loading(stage), "", gr.update(value="", visible=False), None, ""
        image_out, body, text, english = box["out"]
        yield (image_out, body, text, gr.update(value=english, visible=bool(english)),
               {"script": scripts.get(script_label, "ajami"), "lines": text.split("\n")}, "")

    def correction_lines(edited, state):
        """Corrected lines: the edited reading's lines that differ from the machine's."""
        if not state or not edited:
            return []
        edited = edited.strip()
        original = set(l.strip() for l in state["lines"])
        return [l.strip() for l in edited.split("\n") if l.strip() and l.strip() not in original]

    def learn(edited, state):
        lines = correction_lines(edited, state)
        if not lines:
            return ("No corrections found: edit the text above (fix the misread letters or words), "
                    "then press the button.")
        try:
            n = reader.learn(lines, state["script"])
        except Exception as e:                              # noqa: BLE001
            return f"Could not learn from the corrections: {e!r}"
        if not n:
            return "Learning needs the language model, which is not loaded for this script."
        return (f"Learned from {n} corrected line{'s' if n != 1 else ''}. From now on, readings in this "
                f"session prefer these words and spellings. Corrections are kept until the app restarts.")

    def stop_now(request: gr.Request = None):
        cancel_flag(request).set()                           # stops this browser's reading only
        return None, STOPPED, "", gr.update(value="", visible=False), ""

    def _run(image, script_label, mode_label, trans_on, lm_on, tidy_on, show_models, progress=None):
        if image is None:
            return None, EMPTY, "", ""
        script = scripts.get(script_label, "ajami")
        notes = []

        def note(msg):                                       # kept, and shown in the server's output
            notes.append(msg)
            print(msg, flush=True)
        try:
            boxes, crops, results, secs, method, text = read_page(
                reader, image, modes[mode_label], kraken_bin, log=note, script=script,
                use_lm=bool(lm_on), tidy=bool(tidy_on), translate=bool(trans_on), ai=True,
                progress=progress)
        except Exception as e:                              # noqa: BLE001
            return None, (f'<div class="nq-empty">Reading stopped with an error: '
                          f'{html.escape(repr(e))}. Try "The image is one line" for a single '
                          f'line, or crop the page to its text.</div>'), "", ""
        if not results:
            return image, ('<div class="nq-empty">No lines of text were found on this image. '
                           'Crop it to the text, or choose "The image is one line".</div>'), "", ""
        body = render_html(crops, results, secs, method, show_models, script, bool(lm_on))
        english = "\n".join(r.get("english", "") for r in results) if any(r.get("english") for r in results) else ""
        return draw_boxes(image, boxes), body, text, english

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # Gradio 6: css and theme belong to launch()
        blocks = _make(gr.Blocks, title="Iʿjām", css=CSS, theme=ijam_theme())
    with blocks as demo:
        gr.HTML(HEAD, elem_id="nq-head")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                inp = _make(gr.Image, type="pil", label="Manuscript page or line",
                            sources=["upload", "webcam", "clipboard"], height=440)
            with gr.Column(scale=2):
                script = _make(gr.Radio, choices=list(scripts), value=next(iter(scripts)),
                               label="Script")
                mode = _make(gr.Radio, choices=list(modes), value="Automatic",
                             label="How to find the lines")
                trans = _make(gr.Checkbox, value=False,
                              label="Also show an English translation, and for Ajami the Latin script "
                                    "(experimental, still being improved)")
                with gr.Row():
                    go = _make(gr.Button, "Read the page", variant="primary", elem_id="nq-read-btn", scale=4)
                    stop = _make(gr.Button, "Stop", variant="secondary", elem_id="nq-stop", scale=1)
                with _make(gr.Accordion, "Advanced options", open=False):
                    lm_on = _make(gr.Checkbox, value=True,
                                  label="Enhanced reading (recommended). Untick to use the system exactly as it "
                                        "was tested.")
                    tidy = _make(gr.Checkbox, value=True,
                                 label="Tidy the output: skip dots and specks, widen lines to their full "
                                       "length, hide stray Latin letters and lone letters")
                    show = _make(gr.Checkbox, value=False, label="Show each recogniser's reading")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                boxed = _make(gr.Image, type="pil", label="Lines found", interactive=False)
            with gr.Column(scale=4):
                out = gr.HTML(EMPTY)
        txt = _make(gr.Textbox, label="The text (Enhanced where available): copy it, or correct it and teach Iʿjām",
                    lines=8,
                    rtl=True, interactive=True, show_copy_button=True, buttons=["copy"], elem_id="nq-copy")
        txt_en = _make(gr.Textbox, label="English translation, to copy (experimental)", lines=6,
                       interactive=False, visible=False, show_copy_button=True, buttons=["copy"],
                       elem_id="nq-copy-en")
        with gr.Row():
            teach = _make(gr.Button, "Teach Iʿjām my corrections", variant="secondary", elem_id="nq-teach")
        status = gr.Markdown("")
        last = gr.State(None)
        gr.HTML(FOOT, elem_id="nq-foot")
        inputs = [inp, script, mode, trans, lm_on, tidy, show]
        outputs = [boxed, out, txt, txt_en, last, status]
        try:                                                 # up to three readers at once (phone + laptop)
            reading = go.click(run, inputs=inputs, outputs=outputs, show_progress="hidden",
                               concurrency_limit=3)
        except TypeError:                                    # an older Gradio
            reading = go.click(run, inputs=inputs, outputs=outputs)
        try:                                                 # Stop: cancel the reading and free the queue
            stop.click(stop_now, outputs=[boxed, out, txt, txt_en, status], cancels=[reading], queue=False)
        except TypeError:
            stop.click(stop_now, outputs=[boxed, out, txt, txt_en, status])
        teach.click(learn, inputs=[txt, last], outputs=[status])
    try:
        demo.queue(default_concurrency_limit=3)
    except TypeError:
        demo.queue()
    return demo


# ---------------------------------------------------------------------------
# self-test (no models, no GPU)
# ---------------------------------------------------------------------------

def _self_test():
    import random
    from alignment import reconcile

    rng = random.Random(0)
    alphabet = list("ابتدرسکلمنهوي") + ["\u064e", "\u0650", "\u0652", "\u08f8"]
    for _ in range(300):
        base = "".join(rng.choice(alphabet + [" "]) for _ in range(rng.randint(5, 40)))
        hyps = []
        for name in PROFILES["ajami"]["order"]:
            t = "".join(c for c in base if rng.random() > 0.08)
            hyps.append(Hypothesis(name, t, [round(rng.random(), 3) for _ in t]))
        text, flags = ensemble_line(hyps)
        assert text == reconcile(hyps), "ensemble_line differs from the evaluated rule"
        assert len(flags) == len(text)
    print("ensemble_line reproduces the evaluated confidence vote on 300 random cases")

    page = Image.new("L", (1200, 900), 235)
    d = ImageDraw.Draw(page)
    truth = []
    for i in range(6):
        y = 90 + i * 130
        d.rectangle([150, y, 1050, y + 40], fill=40)                  # the letters
        for x in range(170, 1030, 60):
            d.rectangle([x, y - 22, x + 18, y - 14], fill=60)         # marks above
            d.rectangle([x + 25, y + 52, x + 40, y + 60], fill=60)    # marks below
        truth.append((y - 22, y + 60))
    boxes = group_rows(segment_projection(page))
    assert len(boxes) == 6, f"found {len(boxes)} lines, expected 6: {boxes}"
    for (x0, y0, x1, y1), (a, b) in zip(boxes, truth):
        assert y0 <= a + 2 and y1 >= b - 2, "a line's marks were cut off"
    print("projection segmenter: 6 of 6 lines found, marks above and below kept")

    halves = [(600, 100, 1000, 150), (100, 105, 520, 148), (100, 300, 1000, 350)]
    assert group_rows(halves) == [(100, 100, 1000, 150), (100, 300, 1000, 350)]
    print("two halves of a verse are merged into one line")

    lines = [(100, 100 + 60 * i, 1000, 150 + 60 * i) for i in range(8)]
    region = (90, 90, 1010, 640)                 # a whole text block
    speck = (500, 700, 530, 740)                 # a smudge
    got = group_rows(lines + [region, speck])
    assert got == [tuple(b) for b in lines], got
    print("a text-block box and a speck are ignored; all 8 lines kept separate")

    alto = ('<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page><PrintSpace>'
            '<TextBlock><TextLine HPOS="10" VPOS="20" WIDTH="300" HEIGHT="40"><Shape>'
            '<Polygon POINTS="10,20 310,20 310,60 10,60"/></Shape></TextLine></TextBlock>'
            '</PrintSpace></Page></Layout></alto>')
    native = json.dumps({"type": "baselines",
        "regions": {"text": [{"boundary": [[0, 0], [500, 0], [500, 900], [0, 900]]}]},
        "lines": [{"baseline": [[5, 50], [200, 52]],
                   "boundary": [[5, 30], [200, 30], [200, 70], [5, 70]]}]})
    with tempfile.TemporaryDirectory() as dd:
        (Path(dd) / "a.xml").write_text(alto)
        (Path(dd) / "n.json").write_text(native)
        assert _parse_segmentation(Path(dd) / "a.xml") == [(10.0, 20.0, 310.0, 60.0)]
        assert _parse_segmentation(Path(dd) / "n.json") == [(5.0, 30.0, 200.0, 70.0)]
    print("Kraken output parsed from ALTO and native JSON")

    class FakeReader:
        scripts = {"ajami": None, "arabic": None}

        def read(self, images, script="ajami", use_lm=True):
            return [{"text": "بَرࣸ دࣸࢽْ", "unsure": [False, True, False, False, False, False, False],
                     "models": {m: "بَرࣸ" for m in PROFILES[script]["order"]}} for _ in images]

    boxes, crops, results, secs, method, text = read_page(FakeReader(), page.convert("RGB"),
                                                          mode="projection")
    html_out = standalone_page(render_html(crops, results, secs, method, True))
    assert html_out.count('class="nq-line"') == 6 and "nq-unsure" in html_out
    draw_boxes(page, boxes)
    seen = []
    read_page(FakeReader(), page, "projection", None, log=lambda *a: None, progress=seen.append)
    assert seen[0] == "Preparing the image…" and seen[1] == "Finding the lines…" and \
        seen[2].startswith("Reading ") and seen[-1] == "Laying out the page…", seen
    assert "Preparing the image…" in loading("Preparing the image…")

    class _Req:
        def __init__(self, h): self.session_hash = h
    a, b = cancel_flag(_Req("laptop")), cancel_flag(_Req("phone"))
    a.set()
    assert a.is_set() and not b.is_set(), "Stop on one device must not stop another"
    a.clear()

    def stop_at_reading(msg):
        if msg.startswith("Reading "):
            raise Stopped()
    try:
        read_page(FakeReader(), page, "projection", None, log=lambda *a: None, progress=stop_at_reading)
        raise AssertionError("Stop was ignored")
    except Stopped:
        pass
    print("page read end to end with a stand-in reader; HTML and boxed image rendered; stages reported in order; Stop ends a reading")

    # Arabic: four members in evaluation order, and the mirrored-scan output of
    # a --bidi-runs CRNN returned to logical order before the vote.
    rng = random.Random(1)
    order = PROFILES["arabic"]["order"]
    assert order == ["crnn1", "crnn3", "crnn2", "trocr"]
    for _ in range(300):
        base = "".join(rng.choice(list("ابتثجحخدرسشصعفقكلمنهوي ١٢٣") + ["\u064e"])
                       for _ in range(rng.randint(5, 40)))
        per_model = {}
        for name in order:
            t = "".join(c for c in base if rng.random() > 0.08)
            per_model[name] = [(t, [round(rng.random(), 3) for _ in t])]
        got = combine(per_model, order, 1)[0]["text"]
        want = reconcile([Hypothesis(nm, *per_model[nm][0]) for nm in order])
        assert got == want, "the Arabic vote differs from the evaluated rule"
    from bidi_order import mirror_order_with_confidence
    text, conf = mirror_order_with_confidence("سنة 7591 م", list(range(10)))
    assert text == "سنة 1957 م" and conf == [0, 1, 2, 3, 7, 6, 5, 4, 8, 9]
    print("Arabic: four-member vote matches the evaluated rule on 300 cases; "
          "mirrored CRNN output returned to logical order")
    html_ar = render_html(crops, FakeReader().read(crops, "arabic"), 1.0, "a test", True, "arabic")
    assert 'lang="ar"' in html_ar and html_ar.count('class="nq-line"') == 6
    # the language-model vote: lam 0 reproduces the evaluated vote; proposals only
    from lm_vote import KNLM, lm_decode
    lm = KNLM.from_texts(["سلام كتير حار وقلب مشتاق"] * 20 + ["كلنا بالصحة الحمدلله"] * 20, order=5)
    per = {"crnn1": [("سلام كتير", [0.9] * 9)], "crnn3": [("سلام كشر", [0.6] * 8)],
           "crnn2": [("سلام كتير", [0.5] * 9)], "trocr": [("سلام كثير", [0.7] * 9)]}
    plain = combine(per, PROFILES["arabic"]["order"], 1)[0]
    with_lm = combine(per, PROFILES["arabic"]["order"], 1, lm, PROFILES["arabic"]["lm"])[0]
    assert with_lm["text"] == "سلام كتير" and len(with_lm["unsure"]) == len(with_lm["text"])
    assert lm_decode([Hypothesis(n, t, c) for n, (t, c) in [(k, v[0]) for k, v in per.items()]], lm, lam=0.0, gap_bias=0.0, beam=1)[0] == plain["text"]
    print("language-model vote: picks the word the training lines support; lam 0 equals the evaluated vote")
    per16 = dict(per)
    order16 = list(PROFILES["arabic"]["order"])
    for name in ("crnn1", "crnn3", "crnn2"):
        for v in PROFILES["arabic"]["tta"]["variants"]:
            per16[f"{name}_{v}"] = per[name]
            order16.append(f"{name}_{v}")
    w16 = [1.75 if m == "trocr" else 1.0 for m in order16]
    r16 = combine(per16, order16, 1, lm, PROFILES["arabic"]["lm"], w16)[0]
    assert r16["text"] == "سلام كتير" and set(r16["models"]) == {"crnn1", "crnn3", "crnn2", "trocr"}
    a = np.full((64, 120), 255, np.uint8); a[20:40, 10:110] = 0
    assert all(VARIANTS[v](a).shape == a.shape for v in VARIANTS)
    print("enhanced reading: 16 weighted readings combine; only the four main readers are shown")

    # tidying: dots, Latin letters and lone letters go; real words and flags stay aligned
    kept_b, _, kept_r = tidy_results([(0, 0, 9, 9)] * 3, [None] * 3,
                                     [{"text": t, "unsure": [False] * len(t)} for t in ("١٠", "ن", "1 M .")])
    assert [r["text"] for r in kept_r] == ["١٠"], [r["text"] for r in kept_r]     # a page number stays
    t, f = tidy_line("1 M . .. ..11", [False] * 13)
    assert sum(1 for ch in t if unicodedata.category(ch).startswith("L")) < 2
    sample = "عبود الى ن الخوري BL و طوبيا . . ."
    t, f = tidy_line(sample, [False] * 5 + [True] * (len(sample) - 5))
    assert t == "عبود الى الخوري و طوبيا" and len(f) == len(t), (t, len(f))
    t, f = tidy_line("دَ بَرࣸ", [True] + [False] * (len("دَ بَرࣸ") - 1))
    assert t == "دَ بَرࣸ", t                                 # a letter with its mark is a word
    # line clean-up: a box stopping halfway is widened to the ink; a dot box is dropped
    page = Image.new("L", (800, 200), 255)
    d = ImageDraw.Draw(page)
    d.rectangle([40, 60, 760, 90], fill=0)                  # one long line of "ink"
    d.ellipse([400, 150, 406, 156], fill=0)                 # a dot below it
    got = refine_boxes(page.convert("RGB"), [(420, 50, 760, 100), (395, 105, 411, 120)])
    assert len(got) == 1 and got[0][0] <= 45 and got[0][2] >= 755, got        # a dot hugging the line goes
    page2 = Image.new("L", (800, 420), 255)
    d2 = ImageDraw.Draw(page2)
    for y in (40, 110, 180):
        d2.rectangle([40, y + 10, 760, y + 40], fill=0)                      # three lines of "ink"
    d2.ellipse([400, 232, 408, 240], fill=0)                                   # a dot just under the last
    d2.rectangle([700, 330, 730, 350], fill=0)                                 # a page number, on its own
    lines3 = [(40, 40, 760, 90), (40, 110, 760, 160), (40, 180, 760, 230)]
    got = refine_boxes(page2.convert("RGB"), lines3 + [(395, 230, 411, 244), (695, 325, 735, 355)])
    assert len(got) == 4 and got[-1][1] >= 320, got                           # dot gone, page number kept
    sheet = Image.new("RGB", (600, 300), (255, 255, 255))
    for k, y in enumerate((10, 90, 170, 250)):
        sheet.paste(Image.new("RGB", (500 - 60 * k, 40), (225, 214, 190)), (80 + 60 * k, y))
    got = split_white_bands(sheet)
    assert got is not None and len(got) == 4 and got[3][0] == 260, got
    assert split_white_bands(Image.new("RGB", (600, 300), (232, 220, 191))) is None   # a real page: no
    print("tidying: dots, Latin and lone letters removed; a short box widened to its line; a dot box dropped; "
          "a page number kept; a sheet of strips cut at its white gaps")
    got = parse_latin_english("1. Mai sonsa ba zai baƙin ciki ba ||| Whoever loves him will not grieve\n2) Gobe ||| Tomorrow", 3)
    assert got[0] == ("Mai sonsa ba zai baƙin ciki ba", "Whoever loves him will not grieve") and got[1][1] == "Tomorrow" and got[2] is None
    assert "exactly 2 lines" in AJ_LATIN_SYSTEM.format(n=2) and "exactly 3 lines" in AR_ENGLISH_SYSTEM.format(n=3)
    res = add_latin_and_english([{"text": "وَاڧَرْ إِࢽْڢِرَاجِ تَڢَرْکࣸو", "unsure": [False] * 28, "models": {}}],
                                english=False)
    assert res[0]["boko"] == "waƙar infiraji ta farko", res[0]["boko"]
    res[0]["english"] = "The first song of Infiraji"
    page_html = render_html([crops[0]], res, 1.0, "a test", False, "ajami")
    assert 'class="nq-boko"' in page_html and 'class="nq-en"' in page_html and "experimental" in page_html
    print("Latin script and English: transliterated, shown under the line, marked experimental")
    s_ar = _Script.__new__(_Script)
    s_ar.lm = KNLM.from_texts(["الى بيت الشيخ موسى"] * 30, order=5)
    assert s_ar.learn(["بيت الشيخ يزبك"]) == 1 and s_ar.lm.prob(tuple("الشيخ ي")[-4:], "ز") > 0
    print("learning from corrections: a corrected line is added to the script's language model")
    demo = [{"text": "الحديب لان للابنة", "unsure": [False] * 17,
             "models": {"crnn1": "الحديت لان", "crnn3": "الحديب", "crnn2": "الحديب", "trocr": "الحديث لان"}},
            {"text": "واكثروا من", "unsure": [False] * 10, "models": {}}]
    system, body = ai_prompt(demo)
    assert "exactly 2 lines" in system and "recogniser D: الحديث لان" in body
    got = parse_numbered("1. الحديث لان للابنة\n2) واكثروا من", [d["text"] for d in demo])
    assert got == ["الحديث لان للابنة", "واكثروا من"]
    assert parse_numbered("nothing useful", ["a", "b"]) == ["a", "b"]       # falls back to the reading
    f = carry_flags("الحديب لان", [False] * 6 + [True] * 4, "الحديث لان")
    assert f[5] is True and f[0] is False and f[6] is False and f[7:] == [True] * 3, f
    demo[0]["enhanced"] = "الحديث لان للابنة"
    page_html0 = render_html([crops[0]] * 2, demo, 1.0, "a test", False, "arabic")
    assert 'class="nq-enh"' in page_html0 and ">Enhanced<" in page_html0 and "nq-unsure" in page_html0
    del demo[0]["enhanced"]
    demo[0]["check_note"] = "AuthenticationError: invalid x-api-key"
    assert "could not run this time" in render_html([crops[0]] * 2, demo, 1.0, "t", False, "arabic")
    del demo[0]["check_note"]
    demo[0]["english"] = "the talk because the daughter"
    page_html = render_html([crops[0]] * 2, demo, 1.0, "a test", False, "arabic")
    assert 'dir="ltr"' in page_html and "nq-body" in page_html and "nq-ai" not in page_html
    assert combine_checks(["اسمعيني من", "السمعيني من", "اسمعيني من"]) == "اسمعيني من"
    assert combine_checks(["اسمعيني تسحر", "السمعيني تسحر", "اسمعيني سحر"]) == "اسمعيني تسحر"
    print("final check: prompt as tested, answer parsed, changed letters flagged; English laid out left to right; "
          "three checks combined by vote")
    print("all lens self-tests passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="folder with crnn1/, crnn2/, trocr/ and optionally arabic/")
    ap.add_argument("--kraken-bin", default=None,
                    help="the kraken program in its own environment, e.g. /content/kenv/bin/kraken")
    ap.add_argument("--image", help="read one image and write an HTML page instead of the app")
    ap.add_argument("--mode", default="auto", choices=["auto", "kraken", "projection", "single"])
    ap.add_argument("--script", default="ajami", choices=list(PROFILES))
    ap.add_argument("--out", default="lens_result.html")
    ap.add_argument("--share", action="store_true", help="public link for the web app")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.models:
        sys.exit("--models is required")
    reader = Reader(args.models)
    if args.image:
        img = Image.open(args.image)
        boxes, crops, results, secs, method, text = read_page(reader, img, args.mode,
                                                              args.kraken_bin, script=args.script)
        Path(args.out).write_text(standalone_page(render_html(crops, results, secs, method, True,
                                                              args.script)), encoding="utf-8")
        draw_boxes(img, boxes).save(Path(args.out).with_suffix(".boxes.jpg"))
        print(f"{len(results)} lines via {method} in {secs:.1f}s -> {args.out}")
        print(text)
        return
    launch(build_app(reader, args.kraken_bin), share=args.share)


if __name__ == "__main__":
    main()
