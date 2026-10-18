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
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "page.png"
        img.convert("RGB").save(page)
        for flag, ext in (("-n", "json"), ("-a", "xml"), ("-x", "xml")):
            out = Path(d) / f"seg.{ext}"
            cmd = [str(kraken_bin), "-i", str(page), str(out), flag, "segment", "-bl"]
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
    and is dropped; a box narrower than a line is tall is a speck of dirt or
    a stray mark, and is dropped; and only boxes of similar height are ever
    merged, so no line can be swallowed by a larger box.
    """
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    med = heights[len(heights) // 2]
    boxes = [b for b in boxes
             if (b[3] - b[1]) <= 2.5 * med and (b[2] - b[0]) >= 0.8 * med]
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



def refine_boxes(img, boxes):
    """
    Clean the segmenter's lines before reading them (display and reading only;
    the evaluated system was given its lines).

      1. A box much shorter than a typical line is a dot, a stroke or a sliver
         of a neighbouring line: drop it.
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
    boxes = [b for b in boxes if (b[3] - b[1]) >= 0.45 * med]

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
    """Tidy every line and drop lines left with fewer than two letters."""
    kept = []
    for b, c, r in zip(boxes, crops, results):
        text, flags = tidy_line(r["text"], r["unsure"])
        if sum(1 for ch in text if unicodedata.category(ch).startswith("L")) < 2:
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

    def _read_trocr(self, images, chunk=8):
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
        piece = html.escape(text[s:k]).replace(" ", "&nbsp;")
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
.nq-line { display: grid; grid-template-columns: 2.6rem 1fr; gap: 4px 14px; padding: 14px 0;
  border-top: 1px solid var(--nq-rule); }
.nq-line:first-of-type { border-top: 0; }
.nq-num { grid-row: span 3; font-size: 1.35rem; color: var(--nq-rubric); text-align: right;
  padding-top: 2px; font-variant-numeric: oldstyle-nums; }
.nq-line img { max-width: 100%; height: auto; display: block; border-radius: 2px;
  box-shadow: 0 0 0 1px var(--nq-rule); background: #fff; }
.nq-text { font-family: 'Scheherazade New', 'Amiri', 'Noto Naskh Arabic', serif; font-size: 1.95rem; line-height: 2.1;
  direction: rtl; text-align: right; margin: 6px 0 0; overflow: visible; white-space: normal;
  overflow-wrap: anywhere; }
.nq-unsure { color: var(--nq-gold); text-decoration: underline dotted var(--nq-gold);
  text-decoration-thickness: 3px; text-underline-offset: .45em; text-decoration-skip-ink: none; }
.nq-models { display: grid; grid-template-columns: 4.5rem 1fr; gap: 0 10px; margin-top: 4px;
  align-items: baseline; font-size: .9rem; color: var(--nq-faded); }
.nq-models .nq-text { font-size: 1.4rem; line-height: 1.9; margin: 0; color: var(--nq-faded); }
.nq-faint { color: var(--nq-faded); font-style: italic; }
.nq-boko { font-family: 'Gentium Book Plus', Georgia, serif; font-size: 1.15rem; line-height: 1.6;
  margin: 2px 0 0; direction: ltr; text-align: left; color: var(--nq-ink); }
.nq-en { font-size: 1.02rem; line-height: 1.55; margin: 2px 0 0; direction: ltr; text-align: left;
  font-style: italic; color: var(--nq-faded); }
.nq-empty { padding: 28px 4px; color: var(--nq-faded); font-size: 1.08rem; line-height: 1.6; max-width: 46ch; }

/* the desk: quiet controls, one inked action */
#nq-read-btn { background: var(--nq-gold) !important; color: #FFFFFF !important;
  border: 0 !important; font-size: 1.02rem !important; letter-spacing: .01em; }
.dark #nq-read-btn { color: #0E1730 !important; }
#nq-read-btn::before { content: ""; display: inline-block; width: .55em; height: .55em;
  border-radius: 50%; background: currentColor; margin-right: .6em; vertical-align: .08em; }
#nq-read-btn:focus-visible { outline: 3px solid var(--nq-gold) !important; outline-offset: 2px; }
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
    parts = [f'<div class="nq-read"><p class="nq-summary">Read {n} line{"s" if n != 1 else ""} '
             f'in {seconds:.1f} seconds; lines found by {html.escape(method)}'
             f'{"" if use_lm else ", using the tested system only"}. A letter in blue with '
             f'dots beneath it means the recognisers disagreed about it: check those letters against '
             f'the manuscript.'
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
        if r.get("boko"):
            extra += f'<p class="nq-boko" lang="ha">{html.escape(r["boko"])}</p>'
        if r.get("english"):
            extra += f'<p class="nq-en" lang="en">{html.escape(r["english"])}</p>'
        parts.append(f'<div class="nq-line"><div class="nq-num">{k}</div>'
                     f'<img src="{_image_uri(crop)}" alt="Line {k} of the page">'
                     f'<p class="nq-text" lang="{lang}">{highlight(r["text"], r["unsure"])}</p>'
                     f'{extra}{models}</div>')
    parts.append("</div>")
    return "".join(parts)


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
    Ajami: Boko (rule-based, ajami2boko.py), then English from it.
    Arabic: English directly from the reading.
    """
    if script == "ajami":
        from ajami2boko import transliterate
        for r in results:
            r["boko"] = transliterate(r["text"])
        source, src_lang = "boko", "hau_Latn"
    else:
        source, src_lang = "text", "apc_Arab"
    if not english:
        return results
    todo = [k for k, r in enumerate(results) if r.get(source)]
    try:
        en = translate_lines([results[k][source] for k in todo], src_lang, device)
        for k, e in zip(todo, en):
            results[k]["english"] = e
    except Exception as e:                                # noqa: BLE001
        log(f"translation unavailable: {e!r}")
    return results

# ---------------------------------------------------------------------------
# one page, end to end
# ---------------------------------------------------------------------------

def read_page(reader, img, mode="auto", kraken_bin=None, log=print, script="ajami", use_lm=True,
              tidy=True, translate=False):
    img = ImageOps.exif_transpose(img).convert("RGB")
    W, H = img.size
    if mode == "single" or (mode == "auto" and looks_like_single_line(img)):
        boxes, method = [(0, 0, W, H)], "treating the image as one line"
    else:
        boxes, method = None, None
        if mode in ("auto", "kraken"):
            found = segment_kraken(img, kraken_bin, log=log)
            if found:
                boxes, method = found, "Kraken's page segmenter"
        if not boxes:
            boxes, method = segment_projection(img), "the projection segmenter"
        boxes = group_rows(boxes)
        if tidy:
            boxes = refine_boxes(img, boxes)
    boxes = [(max(0, x0), max(0, y0), min(W, x1), min(H, y1)) for x0, y0, x1, y1 in boxes
             if x1 - x0 > 10 and y1 - y0 > 8][:MAX_LINES]
    crops = [img.crop(b) for b in boxes]
    t0 = time.time()
    results = reader.read(crops, script, use_lm)
    seconds = time.time() - t0
    if tidy:
        boxes, crops, results = tidy_results(boxes, crops, results)
    if translate:
        add_latin_and_english(results, getattr(reader, "device", "cpu"), log, script=script)
    text = "\n".join(r["text"] for r in results)
    if any(r.get("boko") for r in results):
        text += "\n\nLatin script (Boko, experimental):\n" + "\n".join(r.get("boko", "") for r in results)
    if any(r.get("english") for r in results):
        text += ("\n\nEnglish (machine translation, experimental):\n"
                 + "\n".join(r.get("english", "") for r in results))
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
  <li>Optionally, each line is translated to English by a machine translation model; Ajami lines are
  first written in Latin script (Boko) by rule. Both are experimental and still being improved.</li>
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

    def run(image, script_label, mode_label, trans_on, lm_on, tidy_on, show_models):
        if image is None:
            return None, EMPTY, ""
        script = scripts.get(script_label, "ajami")
        notes = []
        try:
            boxes, crops, results, secs, method, text = read_page(
                reader, image, modes[mode_label], kraken_bin, log=notes.append, script=script,
                use_lm=bool(lm_on), tidy=bool(tidy_on), translate=bool(trans_on))
        except Exception as e:                              # noqa: BLE001
            return None, (f'<div class="nq-empty">Reading stopped with an error: '
                          f'{html.escape(repr(e))}. Try "The image is one line" for a single '
                          f'line, or crop the page to its text.</div>'), ""
        if not results:
            return image, ('<div class="nq-empty">No lines of text were found on this image. '
                           'Crop it to the text, or choose "The image is one line".</div>'), ""
        body = render_html(crops, results, secs, method, show_models, script, bool(lm_on))
        return draw_boxes(image, boxes), body, text

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
                go = _make(gr.Button, "Read the page", variant="primary", elem_id="nq-read-btn")
                with _make(gr.Accordion, "Advanced options", open=False):
                    lm_on = _make(gr.Checkbox, value=True,
                                  label="Enhanced reading (recommended): read each line several ways and "
                                        "prefer real words. Untick to use the system exactly as it was tested.")
                    tidy = _make(gr.Checkbox, value=True,
                                 label="Tidy the output: skip dots and specks, widen lines to their full "
                                       "length, hide stray Latin letters and lone letters")
                    show = _make(gr.Checkbox, value=False, label="Show each recogniser's reading")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                boxed = _make(gr.Image, type="pil", label="Lines found", interactive=False)
            with gr.Column(scale=4):
                out = gr.HTML(EMPTY)
        txt = _make(gr.Textbox, label="The text, to copy", lines=8, rtl=True,
                    show_copy_button=True, buttons=["copy"], elem_id="nq-copy")
        gr.HTML(FOOT, elem_id="nq-foot")
        go.click(run, inputs=[inp, script, mode, trans, lm_on, tidy, show], outputs=[boxed, out, txt])
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
    print("page read end to end with a stand-in reader; HTML and boxed image rendered")

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
    got = refine_boxes(page.convert("RGB"), [(420, 50, 760, 100), (395, 145, 411, 160)])
    assert len(got) == 1 and got[0][0] <= 45 and got[0][2] >= 755, got
    print("tidying: dots, Latin and lone letters removed; a short box widened to its line; a dot box dropped")
    res = add_latin_and_english([{"text": "وَاڧَرْ إِࢽْڢِرَاجِ تَڢَرْکࣸو", "unsure": [False] * 28, "models": {}}],
                                english=False)
    assert res[0]["boko"] == "waƙar infiraji tafarko", res[0]["boko"]
    res[0]["english"] = "The first song of Infiraji"
    page_html = render_html([crops[0]], res, 1.0, "a test", False, "ajami")
    assert 'class="nq-boko"' in page_html and 'class="nq-en"' in page_html and "experimental" in page_html
    print("Latin script and English: transliterated, shown under the line, marked experimental")
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
