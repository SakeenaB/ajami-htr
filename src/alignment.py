"""
alignment.py
------------
String-level reconciliation of the base models' transcriptions (Section 3.6).

Each base model produces a finished transcription of a line, with a
confidence for every character. The strings can differ in length, so they
cannot be compared position by position. This module:

  1. aligns them with progressive multiple sequence alignment
     (pairwise edit distance, Equation 3.7; Algorithm 3.3), closest pair
     first, inserting gaps where one model has a character and another
     does not;
  2. lays the result out as a confusion network - one column per aligned
     position, one entry per model;
  3. provides the selection rules: majority vote (ROVER), confidence-
     weighted vote (Equation 3.8), and the per-candidate features the
     learned meta-learner consumes (Equation 3.9);
  4. applies the consistency constraints (Section 3.6.4);
  5. computes the oracle bound - the lowest error any selection rule could
     reach on this network - which separates "the right answer was never
     proposed" from "the selection rule picked badly".

Pure Python, no GPU. Run this file directly for the self-test.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

try:
    from alphabet import AJAMI_CHARS
except ImportError:                                   # standalone use
    AJAMI_CHARS = {chr(c) for c in (0x8A8, 0x8A9, 0x8B3, 0x8BD, 0x8F5, 0x8F6,
                                    0x8F7, 0x8F8, 0x8F9, 0x8FB, 0x8FC, 0x8FD)}

GAP = None


@dataclass
class Hypothesis:
    """One model's transcription of one line."""
    model: str
    text: str
    conf: list = field(default_factory=list)   # one value per character

    def __post_init__(self):
        if not self.conf:
            self.conf = [1.0] * len(self.text)
        if len(self.conf) != len(self.text):
            raise ValueError(f"{self.model}: {len(self.text)} characters but "
                             f"{len(self.conf)} confidences")


# ---------------------------------------------------------------------------
# pairwise edit distance (Equation 3.7)
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# progressive multiple sequence alignment (Algorithm 3.3)
# ---------------------------------------------------------------------------

def _column_cost(column, ch):
    """
    Cost of placing character `ch` (or a gap) in an existing profile column:
    the fraction of members already in the column that disagree with it.
    Averaging over members is what lets a third string be aligned against a
    pair that has already been merged.
    """
    members = [c for c in column]
    if not members:
        return 0.0
    return sum(1 for c in members if c != ch) / len(members)


def _align_to_profile(profile, seq):
    """
    Dynamic-programming alignment of a new sequence against a profile.

    profile : list of columns, each a list of characters/GAPs (one per
              model already aligned)
    seq     : list of characters
    returns : (new_profile_columns_as_index_pairs) - a list of
              (profile_column_index or None, seq_index or None)
    """
    P, S = len(profile), len(seq)
    INF = float("inf")
    dp = [[INF] * (S + 1) for _ in range(P + 1)]
    back = [[None] * (S + 1) for _ in range(P + 1)]
    dp[0][0] = 0.0
    for i in range(P + 1):
        for j in range(S + 1):
            if i == 0 and j == 0:
                continue
            best, move = INF, None
            if i > 0 and j > 0:                       # column vs character
                c = dp[i - 1][j - 1] + _column_cost(profile[i - 1], seq[j - 1])
                if c < best:
                    best, move = c, "diag"
            if i > 0:                                 # column vs gap
                c = dp[i - 1][j] + _column_cost(profile[i - 1], GAP)
                if c < best:
                    best, move = c, "up"
            if j > 0:                                 # new all-gap column
                c = dp[i][j - 1] + 1.0
                if c < best:
                    best, move = c, "left"
            dp[i][j], back[i][j] = best, move

    path, i, j = [], P, S
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            path.append((i - 1, j - 1)); i -= 1; j -= 1
        elif move == "up":
            path.append((i - 1, None)); i -= 1
        else:
            path.append((None, j - 1)); j -= 1
    path.reverse()
    return path


def build_network(hyps):
    """
    Align the hypotheses into a confusion network.

    Returns a list of columns. Each column is a list, in the same order as
    `hyps`, of (character or GAP, confidence or None).

    Merge order matters because the alignment is greedy: the first pair
    fixes the frame the others are aligned into. Starting with the two
    strings that disagree least keeps spurious gaps from propagating.
    """
    n = len(hyps)
    if n == 0:
        return []
    texts = [list(h.text) for h in hyps]
    if n == 1:
        return [[(c, hyps[0].conf[k])] for k, c in enumerate(texts[0])]

    # closest pair first
    pairs = sorted(((edit_distance(texts[a], texts[b]), a, b)
                    for a in range(n) for b in range(a + 1, n)))
    _, a, b = pairs[0]
    order = [a, b] + [k for k in range(n) if k not in (a, b)]

    # profile columns hold, per merged model, the index into its text (or None)
    first = order[0]
    profile_idx = [{first: k} for k in range(len(texts[first]))]

    def chars_of(col):
        return [texts[m][col[m]] if col.get(m) is not None else GAP
                for m in merged]

    merged = [first]
    for m in order[1:]:
        profile_chars = [chars_of(col) for col in profile_idx]
        path = _align_to_profile(profile_chars, texts[m])
        new_profile = []
        for p, s in path:
            col = dict(profile_idx[p]) if p is not None else {}
            col[m] = s
            new_profile.append(col)
        profile_idx = new_profile
        merged.append(m)

    network = []
    for col in profile_idx:
        entry = []
        for m in range(n):
            k = col.get(m)
            if k is None:
                entry.append((GAP, None))
            else:
                entry.append((texts[m][k], hyps[m].conf[k]))
        network.append(entry)
    return network


# ---------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------

def candidates(column):
    """Distinct proposals in a column, in first-proposed order."""
    seen = []
    for ch, _ in column:
        if ch not in seen:
            seen.append(ch)
    return seen


def vote(network, mode="confidence", weights=None, gap_conf=0.5):
    """
    Pick one candidate per column.

    mode "majority"   : ROVER-style count of models proposing it.
    mode "confidence" : sum of weight x confidence (Equation 3.8 with fixed
                        weights). A gap carries no confidence of its own, so
                        it is scored with gap_conf.
    Ties go to the earlier model in the input order.
    """
    n = len(network[0]) if network else 0
    weights = weights or [1.0] * n
    out = []
    for column in network:
        scores = {}
        for m, (ch, c) in enumerate(column):
            if mode == "majority":
                s = weights[m]
            else:
                s = weights[m] * (gap_conf if ch is GAP else (c if c is not None else 0.0))
            scores[ch] = scores.get(ch, 0.0) + s
        best = max(candidates(column), key=lambda ch: scores[ch])
        if best is not GAP:
            out.append(best)
    return "".join(out)


def column_features(network):
    """
    Per-candidate features for the learned meta-learner (Equation 3.9).

    Yields (column_index, candidate, features). The meta-learner is trained
    to score candidates from these, using validation lines only - never the
    training split, where the base models are overconfident.
    """
    L = max(1, len(network))
    for j, column in enumerate(network):
        cands = candidates(column)
        for ch in cands:
            props = [m for m, (c, _) in enumerate(column) if c == ch]
            confs = [column[m][1] for m in props if column[m][1] is not None]
            yield j, ch, {
                "n_agree": len(props),
                "conf_sum": sum(confs),
                "conf_max": max(confs) if confs else 0.0,
                "is_gap": ch is GAP,
                "is_ajami": ch in AJAMI_CHARS if ch is not GAP else False,
                "is_mark": (ch is not GAP
                            and unicodedata.category(ch) == "Mn"),
                "n_candidates": len(cands),
                "position": j / L,
                **{f"from_model_{m}": int(m in props)
                   for m in range(len(column))},
            }


# ---------------------------------------------------------------------------
# consistency constraints (Section 3.6.4)
# ---------------------------------------------------------------------------

def _is_mark(ch):
    return unicodedata.category(ch) == "Mn"


def _is_latin(ch):
    return "LATIN" in unicodedata.name(ch, "")


def _is_arabic(ch):
    return "ARABIC" in unicodedata.name(ch, "")


def apply_constraints(text):
    """
    Remove outputs no single model could legitimately have produced.

      * Diacritic attachment: a combining mark must follow a base character,
        not a space or the start of the line.
      * Script consistency: inside a word that is mostly Arabic script, stray
        Latin letters are dropped.
    """
    out = []
    for ch in text:
        if _is_mark(ch) and (not out or out[-1] == " "):
            continue
        out.append(ch)

    words = "".join(out).split(" ")
    cleaned = []
    for w in words:
        arabic = sum(_is_arabic(c) for c in w)
        latin = sum(_is_latin(c) for c in w)
        if arabic and latin and arabic >= latin:
            w = "".join(c for c in w if not _is_latin(c))
        cleaned.append(w)
    return " ".join(cleaned)


def reconcile(hyps, mode="confidence", weights=None, constrain=True):
    """Full pipeline for one line: align -> vote -> constrain."""
    text = vote(build_network(hyps), mode=mode, weights=weights)
    return apply_constraints(text) if constrain else text


# ---------------------------------------------------------------------------
# oracle bounds
# ---------------------------------------------------------------------------

def oracle_errors(network, reference):
    """
    Lowest edit distance any column-wise selection could reach.

    Dynamic programming over (reference position, network column). At each
    column the oracle may emit any proposed candidate, including a gap. If the
    oracle error is low while the ensemble's is high, the right characters
    were proposed and the selection rule is the weakness; if both are high,
    no model proposed them.
    """
    R, C = len(reference), len(network)
    INF = float("inf")
    dp = [[INF] * (C + 1) for _ in range(R + 1)]
    dp[0][0] = 0
    for i in range(R + 1):
        for j in range(C + 1):
            v = dp[i][j]
            if v == INF:
                continue
            if j < C:
                cands = candidates(network[j])
                # emit nothing for this column (free if a gap was proposed)
                skip = 0 if GAP in cands else 1
                dp[i][j + 1] = min(dp[i][j + 1], v + skip)
                # match this column to reference[i]
                if i < R:
                    hit = 0 if reference[i] in cands else 1
                    dp[i + 1][j + 1] = min(dp[i + 1][j + 1], v + hit)
            if i < R:                                 # reference char unmatched
                dp[i + 1][j] = min(dp[i + 1][j], v + 1)
    return dp[R][C]


def line_oracle_errors(hyps, reference):
    """Best single model for this line - a weaker, line-level oracle."""
    return min(edit_distance(h.text, reference) for h in hyps)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _self_test():
    H = Hypothesis
    arrow, noon, fatha = "\u08f8", "\u08bd", "\u064e"

    # 1. agreement passes straight through
    hs = [H("crnn", "abc"), H("trocr", "abc"), H("kraken", "abc")]
    assert reconcile(hs) == "abc"

    # 2. one model drops a mark; the other two restore it
    ref = f"ب{fatha}ر{arrow}د"
    hs = [H("crnn", f"ب{fatha}رد"),
          H("trocr", ref),
          H("kraken", ref)]
    assert reconcile(hs, mode="majority") == ref

    # 3. different lengths are aligned, not compared position by position
    net = build_network([H("a", "abcd"), H("b", "abd"), H("c", "abcd")])
    assert len(net) == 4, f"expected 4 columns, got {len(net)}"
    gaps = [m for m, (ch, _) in enumerate(net[2]) if ch is GAP]
    assert gaps == [1], f"gap should sit in model b at column 2, got {gaps}"

    # 4. confidence breaks a 1-1-1 split that majority cannot
    hs = [H("crnn", "x", [0.9]), H("trocr", "y", [0.2]), H("kraken", "z", [0.3])]
    assert reconcile(hs, mode="confidence") == "x"

    # 5. each model wrong in a different place: oracle reaches zero,
    #    every single model has one error
    ref = "abcdef"
    hs = [H("a", "Xbcdef"), H("b", "abYdef"), H("c", "abcdeZ")]
    net = build_network(hs)
    assert oracle_errors(net, ref) == 0
    assert line_oracle_errors(hs, ref) == 1
    assert reconcile(hs, mode="majority") == ref

    # 6. the oracle can never beat what was proposed
    hs = [H("a", "abc"), H("b", "abc"), H("c", "abc")]
    assert oracle_errors(build_network(hs), "abd") == 1

    # 7. constraints: orphan mark dropped; stray Latin in Arabic word dropped
    assert apply_constraints(f"{fatha}ب ر") == "ب ر"
    assert apply_constraints("بxر") == "بر"
    assert apply_constraints("abc") == "abc"

    # 8. features exist for every candidate, gap included
    net = build_network([H("a", "ab"), H("b", "a"), H("c", "ab")])
    feats = list(column_features(net))
    gap_rows = [f for _, ch, f in feats if ch is GAP]
    assert gap_rows and gap_rows[0]["n_agree"] == 1

    # 9. a real Ajami line with one model dropping arrowheads, one dropping
    #    the dotless noon
    ref = "بَذَيْ بَرࣸدࣸࢽْ غࣸدࣸࢽْ سَکِيبَا"
    crnn = ref.replace(arrow, "")
    trocr = ref.replace(noon, "ن")
    kraken = ref
    hs = [H("crnn", crnn), H("trocr", trocr), H("kraken", kraken)]
    out = reconcile(hs, mode="majority")
    assert out == ref, f"\n got {out}\nwant {ref}"
    assert oracle_errors(build_network(hs), ref) == 0

    print("all alignment tests passed")


if __name__ == "__main__":
    _self_test()
