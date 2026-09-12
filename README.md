# Ajami HTR — Stacked Ensemble

Final year project, Department of Computer Science, University of Ibadan.
Bashir Sekinat Omotoke (230895), supervised by Dr. O. A. Abiola.

A stacked ensemble architecture for handwritten text recognition of historical
West African manuscripts in Hausa Ajami, Fulfulde Ajami and classical Arabic.

## Target

A single CRNN trained on Ajami reaches 10.66% corpus-level CER
(Al-azzawi, Barney & Liwicki, 2026). This project tests whether reconciling
three architecturally distinct recognisers — CRNN, TrOCR and Kraken — through
a string-level meta-learner improves on that, and reports a diacritic-restricted
error rate over the twelve Ajami-specific codepoints, which no prior study does.

## Data

Yousuf et al. (2026), Zenodo record 15691686 v1, CC-BY-4.0.
6,206 transcribed line images across 29 manuscripts (5,400 Hausa, 806 Fulfulde),
partitioned 70/15/15 following Al-azzawi et al. (2026) so that results are
directly comparable to the published benchmark.

Data is not committed. It is prepared on Kaggle and downloaded as
`dataset.csv` plus a `lines/` directory of PNG crops.

## Modules

| file | purpose |
|---|---|
| `src/alphabet.py` | character set, encoding, Ajami round-trip verification |
| `src/metrics.py`  | corpus-level CER and WER, diacritic-restricted error rate |
| `src/dataset.py`  | PyTorch Dataset and CTC-compatible collate function |

## Setup

```
conda activate ajami
pip install -r requirements.txt
```

## Use

```
python src/alphabet.py --csv dataset.csv --out alphabet.json
python src/metrics.py
python src/dataset.py --csv dataset.csv --images lines --alphabet alphabet.json --split train
```

`metrics.py` run with no arguments executes its self-tests, which include the
textbook Levenshtein cases and a check that corpus-level CER differs from the
mean of per-line rates.
