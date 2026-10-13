"""
showcase.py
-----------
Builds a self-contained HTML page showing, for real test lines, the
manuscript image, the correct transcription, the best single model's
output, the ensemble's output and the published benchmark's output, with
every error highlighted.

Reads ensemble_results/test_predictions.csv (written by ensemble.py --final),
so it shows the evaluated test predictions exactly as scored: nothing is
re-run, re-tuned or chosen by hand. Lines are picked by a stated rule
(largest number of errors fixed), and a few lines where the ensemble did
worse are included as well, so the page is honest about both directions.

Usage:
  python src/showcase.py --images path/to/lines
  python src/showcase.py --list          # just print which images are needed
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import cer, edit_distance  # noqa: E402


def align_flags(ref, hyp):
    """
    Edit-distance alignment of hyp against ref.
    Returns (flags, missing): flags[i] is True if hyp[i] is wrong or extra;
    missing[i] counts reference characters dropped just before hyp[i]
    (missing[len(hyp)] = dropped at the end).
    """
    R, H = len(ref), len(hyp)
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i
    for j in range(H + 1):
        d[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    flags = [False] * H
    missing = [0] * (H + 1)
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            flags[j - 1] = ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            flags[j - 1] = True
            j -= 1
        else:
            missing[j] += 1
            i -= 1
    return flags, missing


def highlight(ref, hyp):
    """
    HTML for hyp with errors in red. Characters are grouped with their
    combining marks, so a wrong vowel mark colours its whole letter rather
    than breaking the shaping of the word.
    """
    flags, missing = align_flags(ref, hyp)
    out, k = [], 0
    while k < len(hyp):
        start = k
        k += 1
        while k < len(hyp) and unicodedata.category(hyp[k]).startswith("M"):
            k += 1
        cluster = hyp[start:k]
        gap = sum(missing[start:k])
        if gap:
            out.append('<span class="miss" title="missing character">·</span>' * min(gap, 3))
        text = html.escape(cluster).replace(" ", "&nbsp;")
        out.append(f'<span class="err">{text}</span>' if any(flags[start:k]) else text)
    if missing[len(hyp)]:
        out.append('<span class="miss" title="missing character">·</span>' * min(missing[len(hyp)], 3))
    return "".join(out) or '<span class="miss">(empty)</span>'


def line_cer(ref, hyp):
    return edit_distance(ref, hyp) / max(1, len(ref))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="ensemble_results/test_predictions.csv")
    ap.add_argument("--images", default=None, help="folder with <ImageID>.png line images")
    ap.add_argument("--single", default="pred_crnn1", help="column of the best single model")
    ap.add_argument("--final", default="pred_confidence", help="column of the final system")
    ap.add_argument("--n", type=int, default=8, help="lines where the ensemble fixed most errors")
    ap.add_argument("--worse", type=int, default=2, help="lines where the ensemble did worse")
    ap.add_argument("--list", action="store_true", help="only print the image files needed")
    ap.add_argument("--out", default="showcase.html")
    args = ap.parse_args()

    df = pd.read_csv(args.preds, keep_default_na=False, dtype=str)
    for col in (args.single, args.final):
        if col not in df.columns:
            sys.exit(f"column {col} not in {args.preds}; columns: {list(df.columns)}")
    has_bench = "pred_benchmark" in df.columns

    df["e_single"] = [edit_distance(r, h) for r, h in zip(df.reference, df[args.single])]
    df["e_final"] = [edit_distance(r, h) for r, h in zip(df.reference, df[args.final])]
    df["fixed"] = df.e_single - df.e_final
    df["len"] = df.reference.str.len()
    better = df[df.fixed > 0].sort_values(["fixed", "len"], ascending=[False, True])
    worse = df[df.fixed < 0].sort_values("fixed")
    picks = pd.concat([better.head(args.n), worse.head(args.worse)])

    if args.list:
        print("image files needed:")
        for i in picks.ImageID:
            print(f"  {i}.png")
        return

    n_better, n_worse = int((df.fixed > 0).sum()), int((df.fixed < 0).sum())
    n_same = len(df) - n_better - n_worse
    cer_single = cer(list(df.reference), list(df[args.single]))
    cer_final = cer(list(df.reference), list(df[args.final]))
    cer_bench = cer(list(df.reference), list(df.pred_benchmark)) if has_bench else None
    fixed_total = int(df.e_single.sum() - df.e_final.sum())

    def img_tag(image_id):
        if not args.images:
            return '<div class="noimg">(line image not included)</div>'
        p = Path(args.images) / f"{image_id}.png"
        if not p.exists():
            return f'<div class="noimg">(image not found: {html.escape(p.name)})</div>'
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="manuscript line {html.escape(image_id)}">'

    cards = []
    for k, r in enumerate(picks.itertuples(index=False)):
        tag = ("Ensemble fixed " + str(r.fixed) + " error" + ("s" if r.fixed != 1 else "")
               if r.fixed > 0 else "Ensemble did worse here (" + str(-r.fixed) + " more)")
        rows = [("Correct text", html.escape(r.reference).replace(" ", "&nbsp;"), "", "ref"),
                ("Best single model (CRNN-1)", highlight(r.reference, getattr(r, args.single)),
                 f"{line_cer(r.reference, getattr(r, args.single)) * 100:.1f}% CER", ""),
                ("Ensemble (final system)", highlight(r.reference, getattr(r, args.final)),
                 f"{line_cer(r.reference, getattr(r, args.final)) * 100:.1f}% CER", "final")]
        if has_bench:
            rows.append(("Published benchmark", highlight(r.reference, r.pred_benchmark),
                         f"{line_cer(r.reference, r.pred_benchmark) * 100:.1f}% CER", ""))
        body = "".join(
            f'<div class="row {cls}"><div class="lab">{lab}<span>{score}</span></div>'
            f'<div class="txt" dir="rtl" lang="ha-Arab">{txt}</div></div>'
            for lab, txt, score, cls in rows)
        kind = "good" if r.fixed > 0 else "bad"
        cards.append(
            f'<section class="card"><div class="head"><b>Line {k + 1}</b> · {html.escape(r.lang)}'
            f' · <span class="{kind}">{tag}</span></div>{img_tag(r.ImageID)}{body}'
            f'<div class="id">{html.escape(r.ImageID)}</div></section>')

    bench_line = (f"<li>Published benchmark: <b>{cer_bench * 100:.2f}%</b></li>"
                  if cer_bench is not None else "")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ajami HTR — ensemble before and after</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic&display=swap">
<style>
:root{{--bg:#fbfaf7;--fg:#1f1f1f;--muted:#6b6b6b;--card:#fff;--line:#e4e1da;--err:#c62828;--errbg:#fde7e7;--ok:#2e7d32;--accent:#1f5fbf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#a0a0a0;--card:#1e1e1e;--line:#333;--err:#ff8a80;--errbg:#3a1f1f;--ok:#81c784;--accent:#7fb0ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:28px 18px 60px}}
h1{{font-size:1.6rem;margin:0 0 .3rem}} .sub{{color:var(--muted);margin:0 0 1.2rem}}
.sum{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:22px}}
.sum ul{{margin:.3rem 0 0;padding-left:1.2rem}}
.legend{{color:var(--muted);font-size:.9rem;margin:-8px 0 18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 18px}}
.head{{margin-bottom:8px}} .good{{color:var(--ok)}} .bad{{color:var(--err)}}
.card img{{max-width:100%;height:auto;display:block;border-radius:6px;margin:6px 0 10px;background:#fff}}
.noimg{{color:var(--muted);font-style:italic;margin:4px 0 10px}}
.row{{display:grid;grid-template-columns:200px 1fr;gap:12px;padding:8px 0;border-top:1px solid var(--line)}}
.row.final{{background:color-mix(in srgb,var(--accent) 8%,transparent);border-radius:6px;padding:8px 6px}}
.lab{{font-size:.9rem;color:var(--muted)}} .lab span{{display:block;font-weight:600;color:var(--fg)}}
.txt{{font-family:'Noto Naskh Arabic','Noto Sans Arabic','Segoe UI','Arial',serif;font-size:1.45rem;line-height:2.1;overflow-x:auto;white-space:nowrap}}
.err{{color:var(--err);background:var(--errbg);border-radius:3px}} .miss{{color:var(--err);font-weight:700}}
.id{{color:var(--muted);font-size:.75rem;margin-top:6px}}
@media (max-width:640px){{.row{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Ajami handwriting recognition: single model vs ensemble</h1>
<p class="sub">Real lines from the held-out test set, never seen in training or model selection. Every prediction is exactly as scored in the final evaluation.</p>
<div class="sum"><b>Whole test set ({len(df)} lines)</b><ul>
<li>Best single model (CRNN-1): <b>{cer_single * 100:.2f}%</b> character error rate</li>
<li>Ensemble (final system): <b>{cer_final * 100:.2f}%</b> &mdash; {fixed_total} fewer character errors</li>
{bench_line}
<li>Lines improved by the ensemble: {n_better} · unchanged: {n_same} · worse: {n_worse}</li></ul></div>
<p class="legend">Red = a wrong or extra character (a letter is coloured together with its vowel marks). Red dot = a missing character. Shown: the {len(better.head(args.n))} lines where the ensemble fixed the most errors, and {len(worse.head(args.worse))} where it did worse.</p>
{''.join(cards)}
</main></body></html>"""
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out}: {len(picks)} lines; improved {n_better}, unchanged {n_same}, worse {n_worse}")


if __name__ == "__main__":
    main()
