"""
metrics.py
----------
Evaluation metrics for the recognition system.

Three metrics, as specified in Section 3.7:

  * Character Error Rate (CER) - Equation 3.12
  * Word Error Rate (WER)      - Equation 3.13
  * Diacritic-restricted error rate over the twelve Ajami codepoints

Two decisions here are worth understanding, because both change the number
that gets reported.

1. CER is computed at CORPUS level: the total edits across every test line
   divided by the total reference characters. It is NOT the mean of per-line
   rates. On the published predictions of Al-azzawi et al. (2026) the corpus
   figure is 10.66% while the mean of per-line rates is 13.45% - the same
   predictions, a five-point difference. The corpus definition is the one
   behind the published benchmark, so it is the one used here.

2. Nothing is normalised, stripped or lower-cased before comparison. The
   invisible directional marks in the corpus are counted as characters,
   because the benchmark counted them.
"""

from __future__ import annotations

from alphabet import AJAMI_CHARS


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def edit_distance(reference, hypothesis):
    """
    Levenshtein distance: the fewest single-token edits - insertions,
    deletions, substitutions - that turn the hypothesis into the reference.
    Equation 3.11.

    Implemented directly rather than taken from a library so that the
    definition used is visible and defensible, and so the tuple form below can
    share the same recursion. Two rolling rows are kept instead of the full
    matrix, which makes memory linear in the shorter sequence.
    """
    if reference == hypothesis:
        return 0
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current = [i]
        for j, h in enumerate(hypothesis, start=1):
            cost = 0 if r == h else 1
            current.append(min(
                previous[j] + 1,        # deletion from the reference
                current[j - 1] + 1,     # insertion into the reference
                previous[j - 1] + cost  # match or substitution
            ))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Corpus-level metrics
# ---------------------------------------------------------------------------

def cer(references, hypotheses):
    """
    Corpus-level Character Error Rate.

    Sum the edits over all lines, sum the reference characters over all lines,
    then divide. Long lines therefore carry proportionally more weight, which
    is what makes the figure comparable across corpora of differing line
    lengths.
    """
    references = [_s(r) for r in references]
    hypotheses = [_s(h) for h in hypotheses]
    _same_length(references, hypotheses)

    edits = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    chars = sum(len(r) for r in references)
    return edits / chars if chars else 0.0


def wer(references, hypotheses):
    """
    Corpus-level Word Error Rate: the same computation over whitespace-
    separated tokens instead of characters. Less sensitive than CER, but
    closer to what a reader searching the transcription experiences.
    """
    references = [_s(r) for r in references]
    hypotheses = [_s(h) for h in hypotheses]
    _same_length(references, hypotheses)

    edits = sum(edit_distance(r.split(), h.split())
                for r, h in zip(references, hypotheses))
    words = sum(len(r.split()) for r in references)
    return edits / words if words else 0.0


def cer_per_line(references, hypotheses):
    """
    Per-line CER, returned as a list. Reported alongside the corpus figure but
    never in place of it: the gap between the two indicates how far errors are
    concentrated in short lines.
    """
    out = []
    for r, h in zip(references, hypotheses):
        r, h = _s(r), _s(h)
        out.append(edit_distance(r, h) / len(r) if len(r) else 0.0)
    return out


# ---------------------------------------------------------------------------
# Diacritic-restricted error rate
# ---------------------------------------------------------------------------

def diacritic_cer(references, hypotheses, charset=None):
    """
    Character error rate computed over the Ajami-specific characters alone.

    Every character not in the restricted set is removed from both strings
    first, so what remains is the sequence of Ajami marks and letters in the
    order they appeared. The edit distance between those sequences measures
    exactly what the aggregate CER hides: whether the phonemic marks survived.

    This is the metric no published study reports for Ajami, and it is
    deliberately unforgiving. A system can score well on CER while losing a
    third of a rare mark, because that mark is a vanishing fraction of the
    total characters.
    """
    charset = AJAMI_CHARS if charset is None else set(charset)

    def keep(s):
        return "".join(c for c in _s(s) if c in charset)

    refs = [keep(r) for r in references]
    hyps = [keep(h) for h in hypotheses]
    _same_length(refs, hyps)

    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    chars = sum(len(r) for r in refs)
    return edits / chars if chars else 0.0


def per_diacritic_counts(references, hypotheses, charset=None):
    """
    For each Ajami character: how often it appears in the reference, how often
    in the hypothesis, and the net difference.

    A negative net means the system is deleting that mark more often than it
    invents it. This is a count, not an alignment, so it does not distinguish
    a deletion from a substitution elsewhere - use diacritic_cer for the
    rigorous figure and this for the diagnostic breakdown.
    """
    from collections import Counter
    charset = AJAMI_CHARS if charset is None else set(charset)

    in_ref = Counter(c for r in references for c in _s(r) if c in charset)
    in_hyp = Counter(c for h in hypotheses for c in _s(h) if c in charset)

    rows = []
    for c in sorted(set(in_ref) | set(in_hyp), key=lambda x: -in_ref[x]):
        rows.append({
            "char": c,
            "codepoint": f"U+{ord(c):04X}",
            "in_reference": in_ref[c],
            "in_hypothesis": in_hyp[c],
            "net": in_hyp[c] - in_ref[c],
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def evaluate(references, hypotheses, languages=None):
    """
    Full evaluation. If a list of language labels is supplied, results are
    reported per language as well as overall - required here because Hausa
    outnumbers Fulfulde roughly seven to one, and a combined figure would let
    Hausa performance conceal Fulfulde performance entirely.
    """
    result = {
        "n_lines": len(references),
        "cer": cer(references, hypotheses),
        "wer": wer(references, hypotheses),
        "diacritic_cer": diacritic_cer(references, hypotheses),
        "cer_per_line_mean": (
            sum(cer_per_line(references, hypotheses)) / len(references)
            if references else 0.0
        ),
    }

    if languages is not None:
        result["by_language"] = {}
        for lang in sorted(set(languages)):
            idx = [i for i, l in enumerate(languages) if l == lang]
            refs = [references[i] for i in idx]
            hyps = [hypotheses[i] for i in idx]
            result["by_language"][lang] = {
                "n_lines": len(refs),
                "cer": cer(refs, hyps),
                "wer": wer(refs, hyps),
                "diacritic_cer": diacritic_cer(refs, hyps),
            }
    return result


def format_report(result):
    lines = [
        f"lines evaluated      : {result['n_lines']}",
        f"CER (corpus level)   : {result['cer']*100:.2f}%",
        f"WER (corpus level)   : {result['wer']*100:.2f}%",
        f"diacritic-only CER   : {result['diacritic_cer']*100:.2f}%",
        f"CER (mean per line)  : {result['cer_per_line_mean']*100:.2f}%   "
        f"(secondary; corpus figure above is the comparable one)",
    ]
    if "by_language" in result:
        lines.append("")
        for lang, r in result["by_language"].items():
            lines.append(
                f"  {lang:<9} n={r['n_lines']:<5} "
                f"CER {r['cer']*100:6.2f}%   "
                f"WER {r['wer']*100:6.2f}%   "
                f"diacritic {r['diacritic_cer']*100:6.2f}%"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _s(x):
    """Empty transcriptions arrive from pandas as NaN, a float, not a string."""
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _same_length(a, b):
    if len(a) != len(b):
        raise ValueError(
            f"{len(a)} references but {len(b)} hypotheses - they must correspond"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Small cases where the answer is known by hand."""
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1          # one substitution
    assert edit_distance("abc", "ab") == 1           # one deletion
    assert edit_distance("ab", "abc") == 1           # one insertion
    assert edit_distance("kitten", "sitting") == 3   # the textbook example
    assert edit_distance("", "abc") == 3

    # 2 edits over 6 reference characters
    assert abs(cer(["abcdef"], ["abcdXf"]) - 1 / 6) < 1e-9
    assert abs(cer(["abc", "de"], ["abc", "dX"]) - 1 / 5) < 1e-9

    assert abs(wer(["a b c"], ["a b X"]) - 1 / 3) < 1e-9

    # corpus level differs from the mean of per-line rates
    refs = ["abcdefghij", "ab"]
    hyps = ["abcdefghiX", "aX"]
    assert abs(cer(refs, hyps) - 2 / 12) < 1e-9
    per_line = cer_per_line(refs, hyps)
    assert abs(sum(per_line) / 2 - 0.3) < 1e-9       # 0.1 and 0.5

    # diacritic-restricted: the base letters match, only the mark is dropped
    mark = "\u08F8"
    noon = "\u08BD"
    ref = f"\u0628{mark}\u0631{noon}"
    hyp = "\u0628\u0631\u0646"                       # mark gone, noon -> plain noon
    assert diacritic_cer([ref], [hyp]) == 1.0        # both Ajami chars lost
    assert cer([ref], [hyp]) < 1.0                   # aggregate hides it

    counts = per_diacritic_counts([ref], [hyp])
    assert counts[0]["net"] == -1

    # NaN from pandas must not crash
    assert cer([float("nan")], [""]) == 0.0

    print("all self-tests passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate transcriptions.")
    parser.add_argument("--csv", help="CSV with GroundTruth and Prediction columns")
    parser.add_argument("--ref-col", default="GroundTruth")
    parser.add_argument("--hyp-col", default="Prediction")
    parser.add_argument("--lang-col", default=None)
    args = parser.parse_args()

    _self_test()

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        refs = df[args.ref_col].tolist()
        hyps = df[args.hyp_col].tolist()
        langs = df[args.lang_col].tolist() if args.lang_col else None
        print()
        print(format_report(evaluate(refs, hyps, langs)))
        print()
        print("per-diacritic breakdown")
        for row in per_diacritic_counts(refs, hyps):
            print(f"  {row['codepoint']}  ref {row['in_reference']:>5}  "
                  f"hyp {row['in_hypothesis']:>5}  net {row['net']:+}")
