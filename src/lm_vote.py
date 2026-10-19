"""
lm_vote.py
----------
A character language model inside the confidence vote.

The vote (alignment.vote) picks, column by column, the candidate with the
highest confidence-weighted score. This module searches the same alignment
network for the whole-line choice that is best under

    log( normalised vote score )  +  lam * log P_LM( character | history )

with a bias on choosing a gap, so that near-miss words (كشر for كتير) lose to
readings that the training transcriptions make likely. The language model is
an interpolated Kneser-Ney character n-gram model trained on the training
transcriptions only; it can only choose among characters the base models
proposed, never invent text.

Exploratory status: tuned on validation half A and measured on half B (Arabic:
10.83% -> 9.35% CER; Ajami, cross-validated: 9.80% -> 9.57%); the test set was
scored before this existed and has not been re-scored. Settings per script
live in lens.PROFILES.
"""
from __future__ import annotations

import collections
import math

from alignment import GAP, apply_constraints, build_network, candidates

BOS, EOS = "\x02", "\x03"


class Counts:
    """Raw and continuation character n-gram counts up to maxn, built once."""

    def __init__(self, texts, maxn=8):
        self.maxn = maxn
        self.raw = [collections.Counter() for _ in range(maxn + 1)]
        self.vocab = set()
        for t in texts:
            s = [BOS] * (maxn - 1) + list(t) + [EOS]
            self.vocab.update(s)
            for i in range(maxn - 1, len(s)):
                for k in range(1, maxn + 1):
                    self.raw[k][(tuple(s[i - k + 1:i]), s[i])] += 1
        self.cont = [collections.Counter() for _ in range(maxn + 1)]
        for k in range(1, maxn):
            for (h, c) in self.raw[k + 1]:
                self.cont[k][(h[1:], c)] += 1


class KNLM:
    """Interpolated Kneser-Ney character n-gram model of one order."""

    def __init__(self, counts, order=8, D=0.75):
        self.N, self.D = order, D
        self.counts = counts
        self.count = [None] * (order + 1)
        self.count[order] = counts.raw[order]
        for k in range(1, order):
            self.count[k] = counts.cont[k]
        self.hist = [collections.Counter() for _ in range(order + 1)]
        self.types = [collections.Counter() for _ in range(order + 1)]
        for k in range(1, order + 1):
            for (h, c), n in self.count[k].items():
                self.hist[k][h] += n
                self.types[k][h] += 1
        self.V = len(counts.vocab)
        self.cache = {}

    @classmethod
    def from_texts(cls, texts, order=8):
        return cls(Counts(texts, maxn=order), order=order)

    def add_texts(self, texts, weight=3):
        """
        Learn from corrected lines on the spot: their n-grams are added to the
        counts (raw counts weighted, so a correction outweighs one ordinary
        training line), keeping every derived table consistent.
        """
        c, N = self.counts, self.N
        for t in texts:
            s = [BOS] * (c.maxn - 1) + list(t) + [EOS]
            c.vocab.update(s)
            for i in range(c.maxn - 1, len(s)):
                for k in range(1, c.maxn + 1):
                    key = (tuple(s[i - k + 1:i]), s[i])
                    new_raw = c.raw[k][key] == 0
                    c.raw[k][key] += weight
                    if k == N:                                  # highest order: raw counts
                        self.hist[N][key[0]] += weight
                        if new_raw:
                            self.types[N][key[0]] += 1
                    if new_raw and 2 <= k <= N:                 # a new continuation for order k-1
                        ck = (key[0][1:], key[1])
                        new_cont = c.cont[k - 1][ck] == 0
                        c.cont[k - 1][ck] += 1
                        if k - 1 < N:
                            self.hist[k - 1][ck[0]] += 1
                            if new_cont:
                                self.types[k - 1][ck[0]] += 1
        self.V = len(c.vocab)
        self.cache = {}

    def prob(self, h, c, k=None):
        k = self.N if k is None else k
        if k == 0:
            return 1.0 / (self.V + 1)
        hk = tuple(h[-(k - 1):]) if k > 1 else ()
        key = (hk, c, k)
        if key in self.cache:
            return self.cache[key]
        tot = self.hist[k].get(hk, 0)
        lower = self.prob(h, c, k - 1)
        if tot == 0:
            p = lower
        else:
            n = self.count[k].get((hk, c), 0)
            p = max(n - self.D, 0) / tot + self.D * self.types[k][hk] / tot * lower
        self.cache[key] = p
        return p

    def logprob(self, h, c):
        return math.log(max(self.prob(h, c), 1e-12))


def _column_scores(col, gap_conf=0.5, weights=None):
    sc = {}
    for m, (ch, c) in enumerate(col):
        w = 1.0 if weights is None else weights[m]
        sc[ch] = sc.get(ch, 0.0) + w * (gap_conf if ch is GAP else (c if c is not None else 0.0))
    return sc


def lm_decode(hyps, lm, lam=0.25, gap_bias=-0.8, beam=8, constrain=True, weights=None, network=None):
    """
    Returns (text, unsure flags). With lam = 0 and gap_bias = 0 this is the
    plain confidence vote. Flags mark a character that only one model proposed
    or that won less than half of its column's score, as in lens.ensemble_line.
    """
    net = network if network is not None else build_network(hyps)
    beams = [(0.0, "", tuple([BOS] * (lm.N - 1)), [])]      # score, text, history, flags
    for col in net:
        scores = _column_scores(col, weights=weights)
        total = sum(scores.values()) or 1.0
        cands = candidates(col)
        new = []
        for sc, text, hist, flags in beams:
            for ch in cands:
                s = sc + math.log(max(scores[ch] / total, 1e-9))
                if ch is GAP:
                    new.append((s + gap_bias, text, hist, flags))
                    continue
                s += lam * lm.logprob(hist, ch)
                support = sum(1 for c, _ in col if c == ch)
                unsure = support == 1 or scores[ch] / total < 0.5
                new.append((s, text + ch, hist[1:] + (ch,), flags + [unsure]))
        best = {}
        for b in new:
            if b[1] not in best or b[0] > best[b[1]][0]:
                best[b[1]] = b
        beams = sorted(best.values(), key=lambda b: -b[0])[:beam]
    finals = [(sc + lam * lm.logprob(hist, EOS), text, flags) for sc, text, hist, flags in beams]
    _, text, flags = max(finals, key=lambda f: f[0])
    if constrain:
        fixed = apply_constraints(text)
        if fixed != text:                      # constraints may drop characters
            flags = (flags + [False] * len(fixed))[: len(fixed)]
            text = fixed
    return text, flags


def _self_test():
    import random
    from alignment import Hypothesis, reconcile
    rng = random.Random(0)
    texts = ["".join(rng.choice("ابتثجحخدرسشصعفقكلمنهوي ") for _ in range(rng.randint(10, 40))) for _ in range(300)]
    lm = KNLM.from_texts(texts, order=5)
    for _ in range(200):
        base = rng.choice(texts)
        hyps = []
        for name in ("a", "b", "c"):
            t = "".join(c for c in base if rng.random() > 0.1)
            hyps.append(Hypothesis(name, t, [round(rng.random(), 3) for _ in t]))
        plain, _ = lm_decode(hyps, lm, lam=0.0, gap_bias=0.0, beam=1)
        assert plain == reconcile(hyps), "lam = 0 must reproduce the confidence vote"
        text, flags = lm_decode(hyps, lm)
        assert len(flags) == len(text)
        allowed = set("".join(h.text for h in hyps))
        assert set(text) <= allowed, "the decoder may only choose proposed characters"
    print("lm_vote: lam 0 reproduces the vote on 200 cases; LM choices stay within the proposals")
    # learning from a correction: the corrected name wins afterwards
    lm2 = KNLM.from_texts(["الى بيت الشيخ موسى"] * 30, order=5)
    hyps = [Hypothesis("a", "الشيخ يذبك", [0.9] * 10), Hypothesis("b", "الشيخ يزبك", [0.6] * 10),
            Hypothesis("c", "الشيخ يذبك", [0.7] * 10)]
    before, _ = lm_decode(hyps, lm2, lam=0.5, gap_bias=0.0)
    lm2.add_texts(["بيت الشيخ يزبك"] * 2)
    after, _ = lm_decode(hyps, lm2, lam=0.5, gap_bias=0.0)
    assert before == "الشيخ يذبك" and after == "الشيخ يزبك", (before, after)
    # incremental tables equal a rebuild from scratch
    ref = KNLM.from_texts(["الى بيت الشيخ موسى"] * 30 + ["بيت الشيخ يزبك"] * 6, order=5)
    for h, ch in [((BOS,) * 4, "ب"), (tuple("الشيخ")[-4:], " "), (tuple("ح يز")[-4:], "ب")]:
        assert abs(ref.prob(h, ch) - lm2.prob(h, ch)) < 1e-9, (h, ch, ref.prob(h, ch), lm2.prob(h, ch))
    print("lm_vote: a corrected word wins after learning; incremental update equals a rebuild")


if __name__ == "__main__":
    _self_test()
