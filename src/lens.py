"""
lens.py
-------
Ajami Lens: point it at a manuscript page and it reads the page.

  page image -> find the lines -> CRNN-1, CRNN-2 and TrOCR read every line
             -> the confidence vote combines them (the final system of the
                evaluation) -> the page with each line boxed, beside the text,
                with characters the models disagreed on highlighted.

Line finding is not part of the evaluated system: the corpus came with line
boxes. The Lens uses Kraken's pretrained page segmenter when available (run
from its own environment, because Kraken and TrOCR need incompatible library
versions), and falls back to a simple projection-profile segmenter that suits
clean single-column scans. Recognition uses exactly the preprocessing,
models and ensemble rule that produced the 10.42% test result.

Usage
  python lens.py --self-test                      # no models needed
  python lens.py --models /content/models --image page.png --out lens.html
  python lens.py --models /content/models --kraken-bin /content/kenv/bin/kraken --share

--models must contain crnn1/, crnn2/ and trocr/ (each a downloaded Kaggle
notebook output).
"""

from __future__ import annotations

import argparse
import base64
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
MAX_LEN = 128          # TrOCR generation limit, as trained
MODEL_ORDER = ["crnn1", "crnn2", "trocr"]   # order used by ensemble.py (vote ties)
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


def looks_like_single_line(img):
    return img.height <= 400 and img.width >= 3 * img.height


# ---------------------------------------------------------------------------
# the ensemble, with per-character certainty
# ---------------------------------------------------------------------------

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


class Reader:
    """Loads CRNN-1, CRNN-2 and TrOCR once; reads lists of line images."""

    def __init__(self, models_dir, device=None):
        import torch
        from alphabet import Alphabet
        from crnn import CRNN
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        root = Path(models_dir)
        self.crnns = []
        for name in ("crnn1", "crnn2"):
            alphabet = Alphabet.load(_find(root / name, "alphabet.json"))
            ckpt = torch.load(_find(root / name, "best.pt"), map_location=self.device)
            model = CRNN(n_classes=alphabet.size_with_blank, input_height=HEIGHT).to(self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self.crnns.append((name, model, alphabet))
            print(f"loaded {name} (epoch {ckpt.get('epoch')})")

        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
        from train_trocr import configure_special_tokens
        best = next((p.parent for p in sorted((root / "trocr").rglob("config.json"))
                     if p.parent.name == "best"), None) or _find(root / "trocr", "config.json").parent
        self.tok = AutoTokenizer.from_pretrained(best)
        self.proc = AutoImageProcessor.from_pretrained(best)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(best).to(self.device)
        configure_special_tokens(self.trocr, self.tok, MAX_LEN)
        self.trocr.eval()
        print(f"loaded trocr from {best} on {self.device}")

    def _read_crnn(self, model, alphabet, images):
        from predict import ctc_decode_with_confidence
        torch = self.torch
        arrays, widths = [], []
        for im in images:
            g = im.convert("L")
            w = max(1, min(MAX_WIDTH, int(round(g.width * HEIGHT / g.height))))
            a = np.array(g.resize((w, HEIGHT), Image.LANCZOS), dtype=np.uint8)
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
            out.append(ctc_decode_with_confidence(probs[:T, i, :], alphabet.index_to_char))
        return out

    def _read_trocr(self, images, chunk=8):
        from train_trocr import predict
        out = []
        for s in range(0, len(images), chunk):
            part = [im.convert("L").convert("RGB") for im in images[s:s + chunk]]
            pix = self.proc(part, return_tensors="pt").pixel_values
            batch = {"pixel_values": pix, "image_ids": [""] * len(part),
                     "texts": [""] * len(part), "langs": [""] * len(part)}
            for r in predict(self.trocr, self.tok, [batch], self.device, MAX_LEN):
                out.append((r["prediction"], json.loads(r["confidence"])))
        return out

    def read(self, images):
        if not images:
            return []
        per_model = {}
        for name, model, alphabet in self.crnns:
            per_model[name] = self._read_crnn(model, alphabet, images)
        per_model["trocr"] = self._read_trocr(images)
        results = []
        for i in range(len(images)):
            hyps = []
            for name in MODEL_ORDER:
                text, conf = per_model[name][i]
                conf = (list(conf) + [0.5] * len(text))[: len(text)]
                hyps.append(Hypothesis(name, text, conf))
            text, flags = ensemble_line(hyps)
            results.append({"text": text, "unsure": flags,
                            "models": {h.model: h.text for h in hyps}})
        return results


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

ACCENT = (31, 95, 191)


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
        d.rectangle([x0, y0, x1, y1], outline=ACCENT, width=lw)
        label = str(n)
        tw = size * (0.6 * len(label) + 0.6)
        d.rectangle([x1 - tw, y0, x1, y0 + size * 1.3], fill=ACCENT)
        d.text((x1 - tw + size * 0.3, y0 + size * 0.1), label, fill="white", font=font)
    return im


def _thumb(img, max_w=900):
    im = img.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, max(1, int(im.height * max_w / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def highlight(text, flags):
    """Uncertain characters in amber; a letter and its marks stay together."""
    out, k = [], 0
    while k < len(text):
        s = k
        k += 1
        while k < len(text) and unicodedata.category(text[k]).startswith("M"):
            k += 1
        piece = html.escape(text[s:k]).replace(" ", "&nbsp;")
        out.append(f'<span class="unsure">{piece}</span>' if any(flags[s:k]) else piece)
    return "".join(out) or '<span class="muted">(nothing read)</span>'


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic&family=Noto+Sans+Arabic&display=swap');
.lens { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.lens .summary { margin: 0 0 12px; opacity: .8; }
.lens .line { border: 1px solid rgba(127,127,127,.3); border-radius: 10px; padding: 10px 12px; margin: 0 0 12px; }
.lens .num { font-weight: 600; margin-bottom: 6px; }
.lens img { max-width: 100%; height: auto; display: block; border-radius: 6px; margin-bottom: 8px; background: #fff; }
.lens .ajami { font-family: 'Noto Naskh Arabic', 'Noto Sans Arabic', 'Segoe UI', Arial, serif;
  font-size: 1.5rem; line-height: 2.1; direction: rtl; text-align: right; overflow-x: auto; white-space: nowrap; }
.lens .unsure { background: rgba(255, 179, 0, .35); border-radius: 3px; }
.lens .models { margin-top: 6px; font-size: .9rem; }
.lens .models div { display: grid; grid-template-columns: 70px 1fr; gap: 8px; align-items: center; }
.lens .models .ajami { font-size: 1.15rem; line-height: 1.9; }
.lens .muted { opacity: .6; }
"""


def render_html(crops, results, seconds, method, show_models):
    parts = [f'<div class="lens"><p class="summary">{len(results)} line(s) read in '
             f'{seconds:.1f} s · lines found by {html.escape(method)} · '
             f'<span class="unsure">&nbsp;highlighted&nbsp;</span> = the models '
             f'disagreed; check these characters.</p>']
    for n, (crop, r) in enumerate(zip(crops, results), 1):
        models = ""
        if show_models:
            models = '<div class="models">' + "".join(
                f'<div><span class="muted">{name}</span>'
                f'<span class="ajami">{html.escape(t).replace(" ", "&nbsp;") or "—"}</span></div>'
                for name, t in r["models"].items()) + "</div>"
        parts.append(f'<div class="line"><div class="num">Line {n}</div>'
                     f'<img src="{_thumb(crop)}" alt="line {n}">'
                     f'<div class="ajami" lang="ha-Arab">{highlight(r["text"], r["unsure"])}</div>'
                     f'{models}</div>')
    parts.append("</div>")
    return "".join(parts)


def standalone_page(body):
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Ajami Lens</title><style>{CSS} body{{max-width:1000px;margin:24px auto;"
            f"padding:0 16px}}</style></head><body><h1>Ajami Lens</h1>{body}</body></html>")


# ---------------------------------------------------------------------------
# one page, end to end
# ---------------------------------------------------------------------------

def read_page(reader, img, mode="auto", kraken_bin=None, log=print):
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
    boxes = [(max(0, x0), max(0, y0), min(W, x1), min(H, y1)) for x0, y0, x1, y1 in boxes
             if x1 - x0 > 10 and y1 - y0 > 8][:MAX_LINES]
    crops = [img.crop(b) for b in boxes]
    t0 = time.time()
    results = reader.read(crops)
    seconds = time.time() - t0
    text = "\n".join(r["text"] for r in results)
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


def launch(demo, **kwargs):
    """
    Start the app. Gradio 6 takes the page styling in launch(); older
    versions took it in Blocks(), where build_app already passed it.
    """
    try:
        demo.launch(css=CSS, **kwargs)
    except TypeError:
        demo.launch(**kwargs)


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

    def run(image, mode_label, show_models):
        if image is None:
            return None, "<p>Upload a page or a line first.</p>", ""
        notes = []
        try:
            boxes, crops, results, secs, method, text = read_page(
                reader, image, modes[mode_label], kraken_bin, log=notes.append)
        except Exception as e:                              # noqa: BLE001
            return None, f"<p>Something went wrong: {html.escape(repr(e))}</p>", ""
        if not results:
            return image, "<p>No lines were found on this image.</p>", ""
        body = render_html(crops, results, secs, method, show_models)
        return draw_boxes(image, boxes), body, text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # Gradio 6: css belongs to launch()
        blocks = _make(gr.Blocks, title="Ajami Lens", css=CSS)
    with blocks as demo:
        gr.Markdown(
            "# Ajami Lens\n"
            "Read a handwritten **Hausa or Fulfulde Ajami** manuscript. Upload a page (or one "
            "line), press **Read**, and get the text.\n\n"
            "Each line is read by three models (two CRNNs and an adapted TrOCR) and combined by "
            "a confidence vote, the system that reached **10.42% character error rate** on a "
            "held-out test set, below the published benchmark of 10.66%. Characters the models "
            "disagreed on are highlighted: check those first.\n\n"
            "*Research prototype from a University of Ibadan final-year project. It reads clean "
            "scans of manuscripts similar to its training material best. On the free server a "
            "full page can take a few minutes.*")
        with gr.Row():
            inp = _make(gr.Image, type="pil", label="Manuscript page or line",
                        sources=["upload", "webcam", "clipboard"])
            with gr.Column():
                mode = _make(gr.Radio, choices=list(modes), value="Automatic",
                             label="How to find the lines")
                show = _make(gr.Checkbox, value=False, label="Also show each model's reading")
                go = _make(gr.Button, "Read", variant="primary")
        with gr.Row():
            boxed = _make(gr.Image, type="pil", label="Lines found")
            out = gr.HTML()
        txt = _make(gr.Textbox, label="The text, to copy", lines=8, rtl=True,
                    show_copy_button=True)
        go.click(run, inputs=[inp, mode, show], outputs=[boxed, out, txt])
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
        for name in MODEL_ORDER:
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
        def read(self, images):
            return [{"text": "بَرࣸ دࣸࢽْ", "unsure": [False, True, False, False, False, False, False],
                     "models": {m: "بَرࣸ" for m in MODEL_ORDER}} for _ in images]

    boxes, crops, results, secs, method, text = read_page(FakeReader(), page.convert("RGB"),
                                                          mode="projection")
    html_out = standalone_page(render_html(crops, results, secs, method, True))
    assert html_out.count('class="line"') == 6 and "unsure" in html_out
    draw_boxes(page, boxes)
    print("page read end to end with a stand-in reader; HTML and boxed image rendered")
    print("all lens self-tests passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="folder with crnn1/, crnn2/ and trocr/")
    ap.add_argument("--kraken-bin", default=None,
                    help="the kraken program in its own environment, e.g. /content/kenv/bin/kraken")
    ap.add_argument("--image", help="read one image and write an HTML page instead of the app")
    ap.add_argument("--mode", default="auto", choices=["auto", "kraken", "projection", "single"])
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
                                                              args.kraken_bin)
        Path(args.out).write_text(standalone_page(render_html(crops, results, secs, method, True)),
                                  encoding="utf-8")
        draw_boxes(img, boxes).save(Path(args.out).with_suffix(".boxes.jpg"))
        print(f"{len(results)} lines via {method} in {secs:.1f}s -> {args.out}")
        print(text)
        return
    launch(build_app(reader, args.kraken_bin), share=args.share)


if __name__ == "__main__":
    main()
