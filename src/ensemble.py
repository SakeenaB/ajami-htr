"""
ensemble.py
-----------
The meta-learner: combines the base models' saved predictions into one
transcription per line, and produces the Chapter 4 result tables.

Works entirely from the CSV files written by predict.py. No GPU, no model
weights, no libraries beyond numpy and pandas.

What it does
  1. For every line, aligns the base models' outputs into a confusion network
     (alignment.py, Section 3.6.1-3.6.2).
  2. Works out, for each disputed column, which candidates were actually
     right: forcing a right candidate costs no more whole-line error than the
     best possible choice (forward-backward over the network; consistent
     with the oracle bound).
  3. Trains a scorer (logistic regression, Equations 3.8-3.10) on validation
     half B to predict which candidate is right from the features in
     Equation 3.9 - agreement, each model's confidence, whether the character
     is an Ajami mark, the language of the line, and so on.
  4. Estimates every method's accuracy on half B by 5-fold cross-validation
     (the scorer never sees the lines it is scored on), for every
     combination of base models. This is how the final system is chosen -
     without touching the test set.
  5. With --final: scores the chosen system, every baseline and the
     published benchmark on the test set, once, with a paired bootstrap.

Typical use
  python ensemble.py --preds preds_crnn1.csv preds_crnn2.csv preds_trocr.csv
  python ensemble.py --preds ... --final \
      --benchmark ajami-no-share_test_results_epoch_675_wih_imageid.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from alignment import (GAP, Hypothesis, apply_constraints, build_network,  # noqa: E402
                       candidates, oracle_errors, vote)
from metrics import cer, diacritic_cer, edit_distance, wer  # noqa: E402

try:
    from alphabet import AJAMI_CHARS
except ImportError:                                            # pragma: no cover
    AJAMI_CHARS = set()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_lines(paths, parts):
    """
    Read predict.py outputs and gather, per line, every model's hypothesis.
    Only lines present in every file are kept, so all methods are compared
    on exactly the same lines.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, keep_default_na=False, dtype=str)
        need = {"model", "part", "ImageID", "lang", "reference",
                "prediction", "confidence"}
        missing = need - set(df.columns)
        if missing:
            sys.exit(f"{p} is missing columns {sorted(missing)}")
        frames.append(df[df["part"].isin(parts)])
    models = [f["model"].iloc[0] for f in frames]
    if len(set(models)) != len(models):
        sys.exit(f"duplicate model names: {models}")

    by_line = {}
    for df in frames:
        for r in df.itertuples(index=False):
            key = (r.part, r.ImageID)
            line = by_line.setdefault(key, {"part": r.part, "id": r.ImageID,
                                            "lang": r.lang, "ref": r.reference,
                                            "hyps": {}})
            conf = json.loads(r.confidence) if r.confidence else []
            text = r.prediction
            conf = (conf + [0.5] * len(text))[: len(text)]   # length guard
            line["hyps"][r.model] = Hypothesis(r.model, text, conf)

    lines = [l for l in by_line.values() if len(l["hyps"]) == len(models)]
    dropped = len(by_line) - len(lines)
    if dropped:
        print(f"note: {dropped} lines dropped (not predicted by every model)")
    lines.sort(key=lambda l: (l["part"], l["id"]))
    return models, lines


# ---------------------------------------------------------------------------
# features (Equation 3.9)
# ---------------------------------------------------------------------------

def candidate_features(column, ch, j, n_cols, models, lang):
    props = [m for m, (c, _) in enumerate(column) if c == ch]
    confs = [column[m][1] for m in props if column[m][1] is not None]
    fulfulde = 1.0 if lang == "Fulfulde" else 0.0
    is_gap = ch is GAP
    f = {
        "bias": 1.0,
        "n_agree": len(props) / len(models),
        "conf_sum": sum(confs),
        "conf_max": max(confs) if confs else 0.0,
        "is_gap": float(is_gap),
        "is_ajami": float(not is_gap and ch in AJAMI_CHARS),
        "is_mark": float(not is_gap and unicodedata.category(ch) == "Mn"),
        "is_space": float(ch == " "),
        "n_candidates": len(candidates(column)) / len(models),
        "position": j / max(1, n_cols),
        "fulfulde": fulfulde,
    }
    for m, name in enumerate(models):
        proposed = float(m in props)
        c = column[m][1] if (m in props and column[m][1] is not None) else 0.0
        # A model "voting" for a gap carries no confidence of its own; its
        # proposal flag still counts.
        f[f"from_{name}"] = proposed
        f[f"conf_{name}"] = c
        f[f"conf_{name}_x_fulfulde"] = c * fulfulde
        f[f"from_{name}_x_gap"] = proposed * float(is_gap)
    return f


def forced_costs(network, reference):
    """
    For every column and every candidate in it: the lowest whole-line error
    achievable if that column is forced to output that candidate and every
    other column is chosen as well as possible.

    A candidate is 'right' if forcing it costs no more than the best
    candidate in that column. Unlike reading labels off a single oracle path,
    this gives every disputed column a label even where several alignments
    of the reference tie - which matters, because those ties are often
    exactly the columns where one model inserted a stray mark.

    Computed with one forward and one backward pass over the network.
    """
    R, C = len(reference), len(network)
    INF = float("inf")
    cands = [candidates(col) for col in network]

    F = [[INF] * (C + 1) for _ in range(R + 1)]     # prefix cost
    F[0][0] = 0
    for i in range(R + 1):
        for j in range(C + 1):
            v = F[i][j]
            if v == INF:
                continue
            if j < C:
                skip = 0 if GAP in cands[j] else 1
                F[i][j + 1] = min(F[i][j + 1], v + skip)
                if i < R:
                    hit = 0 if reference[i] in cands[j] else 1
                    F[i + 1][j + 1] = min(F[i + 1][j + 1], v + hit)
            if i < R:
                F[i + 1][j] = min(F[i + 1][j], v + 1)

    B = [[INF] * (C + 1) for _ in range(R + 1)]     # suffix cost
    B[R][C] = 0
    for i in range(R, -1, -1):
        for j in range(C, -1, -1):
            if i == R and j == C:
                continue
            best = INF
            if j < C:
                skip = 0 if GAP in cands[j] else 1
                best = min(best, skip + B[i][j + 1])
                if i < R:
                    hit = 0 if reference[i] in cands[j] else 1
                    best = min(best, hit + B[i + 1][j + 1])
            if i < R:
                best = min(best, 1 + B[i + 1][j])
            B[i][j] = best

    out = []
    for j in range(C):
        costs = {}
        for c in cands[j]:
            best = INF
            for i in range(R + 1):
                if F[i][j] == INF:
                    continue
                if c is GAP:
                    best = min(best, F[i][j] + B[i][j + 1])
                else:
                    best = min(best, F[i][j] + 1 + B[i][j + 1])       # inserted
                    if i < R:
                        best = min(best, F[i][j] + (reference[i] != c)
                                   + B[i + 1][j + 1])                   # aligned
            costs[c] = best
        out.append(costs)
    return out, F[R][C]


def line_examples(line, models):
    """Training examples from every disputed column whose candidates are
    not all equally good."""
    hyps = [line["hyps"][m] for m in models]
    net = build_network(hyps)
    costs, _ = forced_costs(net, line["ref"])
    X, y = [], []
    for j, (col, cost) in enumerate(zip(net, costs)):
        if len(cost) < 2:
            continue
        lo, hi = min(cost.values()), max(cost.values())
        if lo == hi:
            continue                          # choice makes no difference
        for ch, c in cost.items():
            X.append(candidate_features(col, ch, j, len(net), models, line["lang"]))
            y.append(1.0 if c == lo else 0.0)
    return X, y


# ---------------------------------------------------------------------------
# logistic regression (numpy, so nothing extra to install)
# ---------------------------------------------------------------------------

class Scorer:
    """
    L2-regularised logistic regression, trained by full-batch gradient
    descent on standardised features. Small, transparent and enough here:
    each disputed column offers only two or three candidates.
    """

    def __init__(self, l2=1e-2, iters=1500, lr=0.5):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, rows, y):
        self.keys = sorted(rows[0].keys())
        X = np.array([[r[k] for k in self.keys] for r in rows], float)
        y = np.asarray(y, float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        bias = self.keys.index("bias")
        self.mu[bias], self.sd[bias] = 0.0, 1.0
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (p - y) / len(y) + self.l2 * w
            grad[bias] -= self.l2 * w[bias]
            w -= self.lr * grad
        self.w = w
        return self

    def score(self, row):
        z = (np.array([row[k] for k in self.keys]) - self.mu) / self.sd
        return float(z @ self.w)

    def weights(self):
        return dict(zip(self.keys, self.w.round(3)))


def learned_decode(line, models, scorer, constrain=True):
    hyps = [line["hyps"][m] for m in models]
    net = build_network(hyps)
    out = []
    for j, col in enumerate(net):
        cands = candidates(col)
        if len(cands) == 1:
            best = cands[0]
        else:
            best = max(cands, key=lambda ch: scorer.score(
                candidate_features(col, ch, j, len(net), models, line["lang"])))
        if best is not GAP:
            out.append(best)
    text = "".join(out)
    return apply_constraints(text) if constrain else text


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def run_simple(lines, models, method):
    outs = []
    for line in lines:
        hyps = [line["hyps"][m] for m in models]
        if method == "oracle":
            outs.append(None)
            continue
        net = build_network(hyps)
        mode = "majority" if method == "majority" else "confidence"
        outs.append(apply_constraints(vote(net, mode=mode)))
    return outs


def run_learned_cv(lines, models, k=5):
    """Cross-validated learned predictions: each line scored by a model
    trained on the other folds only."""
    folds = [i % k for i in range(len(lines))]
    examples = [line_examples(l, models) for l in lines]
    outs = [None] * len(lines)
    for f in range(k):
        Xtr, ytr = [], []
        for i, (X, y) in enumerate(examples):
            if folds[i] != f:
                Xtr += X
                ytr += y
        if not Xtr:
            continue
        scorer = Scorer().fit(Xtr, ytr)
        for i, line in enumerate(lines):
            if folds[i] == f:
                outs[i] = learned_decode(line, models, scorer)
    return outs


def fit_learned(lines, models):
    Xtr, ytr = [], []
    for l in lines:
        X, y = line_examples(l, models)
        Xtr += X
        ytr += y
    return Scorer().fit(Xtr, ytr)


def oracle_cer(lines, models):
    errs = chars = 0
    for line in lines:
        net = build_network([line["hyps"][m] for m in models])
        errs += oracle_errors(net, line["ref"])
        chars += len(line["ref"])
    return errs / chars if chars else 0.0


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------

def collapse_spaces(s):
    return " ".join(s.split())


def report(refs, hyps, langs):
    r = {"cer": cer(refs, hyps), "wer": wer(refs, hyps),
         "diacritic_cer": diacritic_cer(refs, hyps),
         "cer_spaces_collapsed": cer([collapse_spaces(x) for x in refs],
                                     [collapse_spaces(x) for x in hyps])}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        r[f"cer_{lang.lower()}"] = cer([refs[i] for i in idx], [hyps[i] for i in idx])
    return r


def error_positions(ref, hyp):
    """Reference positions a hypothesis gets wrong (substitution or deletion),
    plus positions where it inserts extra characters - via edit-distance
    backtrace."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), int)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    wrong, i, j = set(), R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                wrong.add(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            wrong.add(i - 1)
            i -= 1
        else:
            wrong.add(("ins", i))
            j -= 1
    return wrong


def error_overlap(lines, models):
    """
    For each pair of models: of all the places where at least one of them is
    wrong, the share where BOTH are wrong (Jaccard overlap of error sets).
    Low overlap = they fail differently = room for the ensemble.
    """
    sets = {m: [] for m in models}
    for k, line in enumerate(lines):
        for m in models:
            sets[m].append({(k, p) for p in error_positions(line["ref"],
                                                            line["hyps"][m].text)})
    out = {}
    for a, b in itertools.combinations(models, 2):
        both = either = 0
        for sa, sb in zip(sets[a], sets[b]):
            both += len(sa & sb)
            either += len(sa | sb)
        out[f"{a} & {b}"] = both / either if either else 0.0
    return out


def corpus_cer_parts(refs, hyps):
    e = np.array([edit_distance(r, h) for r, h in zip(refs, hyps)], float)
    n = np.array([len(r) for r in refs], float)
    return e, n


def paired_bootstrap(refs, hyps_a, hyps_b, reps=2000, seed=0):
    """Difference in corpus CER (a - b) over resampled lines. Negative means
    a has fewer errors. Returns mean, 95% interval, and the share of
    resamples in which a is better."""
    rng = np.random.default_rng(seed)
    ea, n = corpus_cer_parts(refs, hyps_a)
    eb, _ = corpus_cer_parts(refs, hyps_b)
    idx = np.arange(len(refs))
    diffs = []
    for _ in range(reps):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append((ea[s].sum() - eb[s].sum()) / n[s].sum())
    diffs = np.array(diffs)
    return {"diff_mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))],
            "share_a_better": float((diffs < 0).mean())}


def per_diacritic(refs, systems):
    from collections import Counter
    ref_c = Counter(c for r in refs for c in r if c in AJAMI_CHARS)
    rows = []
    for ch, n in sorted(ref_c.items(), key=lambda kv: -kv[1]):
        row = {"codepoint": f"U+{ord(ch):04X}", "reference": n}
        for name, hyps in systems.items():
            row[name] = sum(h.count(ch) for h in hyps)
        rows.append(row)
    return pd.DataFrame(rows)


def pct(x):
    return f"{x*100:6.2f}%"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="predict.py output files, one per base model")
    ap.add_argument("--final", action="store_true",
                    help="score the test set. Do this ONCE, after every "
                         "choice has been made on validation half B.")
    ap.add_argument("--benchmark", default=None,
                    help="the benchmark's test predictions (CER-HV csv with "
                         "ImageID, GroundTruth, Prediction) for the paired "
                         "comparison")
    ap.add_argument("--method", default=None,
                    choices=["majority", "confidence", "learned"],
                    help="force a method instead of the best on half B")
    ap.add_argument("--members", nargs="+", default=None,
                    help="force a set of base models instead of the best on half B")
    ap.add_argument("--out", default="ensemble_results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- selection on validation half B ----------------
    models, val = load_lines(args.preds, ["valB"])
    print(f"base models: {models}")
    print(f"validation half B: {len(val)} lines\n")
    refs = [l["ref"] for l in val]
    langs = [l["lang"] for l in val]

    single = {m: cer(refs, [l["hyps"][m].text for l in val]) for m in models}
    order = sorted(models, key=lambda m: single[m])   # best first: tie-break order
    print("single models on half B")
    for m in order:
        print(f"  {m:<10} CER {pct(single[m])}")

    print("\nerror overlap on half B (share of errors made by BOTH models; "
          "lower = more complementary)")
    overlap = error_overlap(val, order)
    for pair, v in overlap.items():
        print(f"  {pair:<24} {pct(v)}")

    print("\ncombinations on half B "
          "(learned = 5-fold cross-validated; oracle = best possible)")
    print(f"  {'members':<34} {'majority':>9} {'confidence':>11} "
          f"{'learned':>9} {'oracle':>9}")
    table = []
    for k in range(2, len(order) + 1):
        for subset in itertools.combinations(order, k):
            subset = list(subset)
            row = {"members": subset}
            for method in ("majority", "confidence"):
                row[method] = cer(refs, run_simple(val, subset, method))
            row["learned"] = cer(refs, run_learned_cv(val, subset))
            row["oracle"] = oracle_cer(val, subset)
            table.append(row)
            print(f"  {'+'.join(subset):<34} {pct(row['majority']):>9} "
                  f"{pct(row['confidence']):>11} {pct(row['learned']):>9} "
                  f"{pct(row['oracle']):>9}")

    best_single = order[0]
    candidates_ = [(row[m], m, row["members"]) for row in table
                   for m in ("majority", "confidence", "learned")]
    best_val, best_method, best_members = min(candidates_, key=lambda t: t[0])
    method = args.method or best_method
    members = args.members or best_members
    if args.method or args.members:
        match = [r for r in table if r["members"] == members]
        best_val = match[0][method] if match else float("nan")
    print(f"\nchosen on half B: {method} over {'+'.join(members)} "
          f"(CER {pct(best_val)}; best single model {best_single} "
          f"{pct(single[best_single])})")

    scorer = fit_learned(val, members) if method == "learned" else None
    if scorer is not None:
        w = scorer.weights()
        top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:10]
        print("largest learned weights: " +
              ", ".join(f"{k} {v:+.2f}" for k, v in top))

    selection = {"models": models, "single_valB": single,
                 "overlap_valB": overlap, "table_valB": table,
                 "chosen": {"method": method, "members": members,
                            "valB_cer": best_val}}
    (out_dir / "selection_valB.json").write_text(json.dumps(selection, indent=1))
    print(f"\nsaved {out_dir/'selection_valB.json'}")

    if not args.final:
        print("\ntest set not scored. Re-run with --final when every choice "
              "is settled.")
        return

    # ---------------- final: the test set, once ----------------
    _, test = load_lines(args.preds, ["test"])
    trefs = [l["ref"] for l in test]
    tlangs = [l["lang"] for l in test]
    systems = {}
    for m in models:
        systems[m] = [l["hyps"][m].text for l in test]
    systems["majority"] = run_simple(test, members, "majority")
    systems["confidence"] = run_simple(test, members, "confidence")
    if scorer is None:
        scorer = fit_learned(val, members)
    systems["learned"] = [learned_decode(l, members, scorer) for l in test]
    final_name = method

    bench = None
    if args.benchmark:
        b = pd.read_csv(args.benchmark, keep_default_na=False, dtype=str)
        bmap = dict(zip(b["ImageID"], b["Prediction"]))
        if all(l["id"] in bmap for l in test):
            bench = [bmap[l["id"]] for l in test]
            systems["benchmark"] = bench
        else:
            missing = sum(l["id"] not in bmap for l in test)
            print(f"WARNING: {missing} test lines missing from the benchmark "
                  "file; benchmark comparison skipped")

    print(f"\nTEST SET ({len(test)} lines) - scored once")
    print(f"  {'system':<12} {'CER':>8} {'WER':>8} {'diacritic':>10} "
          f"{'CER(sp)':>9} {'Hausa':>8} {'Fulfulde':>9}")
    results = {}
    for name, hyps in systems.items():
        r = report(trefs, hyps, tlangs)
        results[name] = r
        flag = "  <- final system" if name == final_name else ""
        print(f"  {name:<12} {pct(r['cer']):>8} {pct(r['wer']):>8} "
              f"{pct(r['diacritic_cer']):>10} {pct(r['cer_spaces_collapsed']):>9} "
              f"{pct(r.get('cer_hausa', 0)):>8} {pct(r.get('cer_fulfulde', 0)):>9}{flag}")
    results["oracle"] = oracle_cer(test, members)
    print(f"  {'oracle':<12} {pct(results['oracle']):>8}   (best possible "
          f"selection from {'+'.join(members)})")

    best_test_single = min(models, key=lambda m: results[m]["cer"])
    comparisons = {"vs_best_single": paired_bootstrap(
        trefs, systems[final_name], systems[best_test_single])}
    if bench is not None:
        comparisons["vs_benchmark"] = paired_bootstrap(
            trefs, systems[final_name], bench)
    print("\npaired bootstrap (2,000 resamples of test lines)")
    for k, v in comparisons.items():
        print(f"  {final_name} {k}: CER difference {v['diff_mean']*100:+.2f} "
              f"points, 95% interval [{v['ci95'][0]*100:+.2f}, "
              f"{v['ci95'][1]*100:+.2f}], better in {v['share_a_better']*100:.1f}% "
              f"of resamples")

    print("\nerror overlap on test")
    test_overlap = error_overlap(test, models)
    for pair, v in test_overlap.items():
        print(f"  {pair:<24} {pct(v)}")

    diac = per_diacritic(trefs, {k: systems[k] for k in
                                 [final_name, best_test_single] +
                                 (["benchmark"] if bench is not None else [])})
    print("\nper-diacritic counts on test (reference vs each system)")
    print(diac.to_string(index=False))

    pd.DataFrame({"ImageID": [l["id"] for l in test], "lang": tlangs,
                  "reference": trefs,
                  **{f"pred_{k}": v for k, v in systems.items()}}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)
    diac.to_csv(out_dir / "per_diacritic_test.csv", index=False)
    (out_dir / "final_test.json").write_text(json.dumps(
        {"final_system": final_name, "members": members, "results": results,
         "comparisons": comparisons, "overlap_test": test_overlap},
        indent=1))
    print(f"\nsaved results to {out_dir}/")


if __name__ == "__main__":
    main()
