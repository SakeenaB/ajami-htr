# Iʿjām (إعجام): Stacked Ensemble OCR for Historical West African Manuscripts

**An Improved Optical Character Recognition (OCR) System for Digitizing Historical West African Manuscripts Using a Stacked Ensemble Meta-Learner**

Final year project, Department of Computer Science, Faculty of Computing, University of Ibadan, 2025/2026.
Bashir Sekinat Omotoke (230895), supervised by Dr O. A. Abiola.

In the Arabic script, letters of the same shape are told apart only by their dots: ب, ت, ث. West African scholars wrote Hausa and Fulfulde in this script (Ajami) for centuries, and the manuscripts are decaying. Recognisers trained on modern Arabic treat the dots and vowel marks as noise and get 65–84% of Ajami characters wrong. This project combines architecturally different handwriting recognisers through a stacked meta-learner, beats the published single-model benchmark, and delivers the system as **Iʿjām**, a web application that reads whole pages and marks the letters it is unsure of. *Iʿjām* is the Arabic word for adding the dots that tell letters apart.

## Results

Each test set was scored once, after the final system had been chosen on validation data.

**Hausa and Fulfulde Ajami** (Yousuf et al., 2026; 931 test lines)

| System | CER | WER | Ajami-mark CER |
|---|---|---|---|
| Best single model (CRNN-1) | 11.28% | 53.69% | 10.32% |
| **Ensemble (confidence vote: CRNN-1, CRNN-2, TrOCR)** | **10.42%** | **51.59%** | **9.55%** |
| Published benchmark, single CRNN (Al-azzawi et al., 2026) | 10.66% | 52.86% | 10.10% |
| Oracle (perfect choice among members; a ceiling, not a system) | 7.01% | | |

- Better than its best member in 100% of 2,000 paired bootstrap resamples (−0.86 points, 95% interval −1.02 to −0.69).
- Better than the benchmark in 96.4% of resamples (−0.24 points, interval −0.50 to +0.02): very probable, not yet statistically certain.
- By language: Hausa 9.11%, Fulfulde 19.13%. Fulfulde has about seven times fewer training lines.

**Historical Arabic** (Muharaf, Saeed et al., 2024; 1,334 test lines)

| System | CER | WER |
|---|---|---|
| Best single model (CRNN-1) | 11.70% | 39.41% |
| **Ensemble (confidence vote: CRNN-1, CRNN-3, CRNN-2, TrOCR)** | **10.17%** | **34.95%** |
| Oracle | 5.60% | |

On the 1,303 test lines also read by the dataset creators' published system: **10.12% against 12.36%**, better in every one of 2,000 resamples (−2.24 points, interval −3.02 to −1.37).

## How it works

1. **Line preparation.** Greyscale lines, 64 pixels high. Right-to-left lines are **mirrored** so the CTC models scan them in writing order. This was the decisive fix: validation error fell from 73.5% to 14.1% within 30 epochs. Augmentation (geometric, morphological, photometric) never removes a dot (96–101% of dots retained). Adaptive Sauvola binarisation is used for finding lines on pages and for Kraken.
2. **Base recognisers.** CRNNs (CNN + BiLSTM + CTC), each trained with its own augmentation recipe and random seed: two for Ajami, three for Arabic (initialised from the Ajami models). TrOCR, with its vocabulary extended (45 tokens for Ajami, 31 for Arabic) and its special tokens corrected. Kraken was trained too, but was left out by the selection rule (see Limitations).
3. **Meta-learner.** The readings are aligned letter by letter into a confusion network, then each column is decided by a majority vote, a confidence vote or a learned scorer. The confidence vote was chosen on validation half B, for both scripts.
4. **Evaluation protocol.** Validation is split into halves: tune on A, choose on B, then score the test set once. Every choice was recorded with a date before the test set was opened. The scoring code reproduces the benchmark's published 10.66% exactly from its own outputs.

## Iʿjām, the application

- Reads whole pages: finds the lines (Kraken or a projection segmenter), reads each line with every member, and combines them.
- Marks the letters the recognisers disagreed on (blue, with dots beneath).
- **Enhanced reading** (on by default; untick to run the system exactly as tested): a character language model in the vote, plus, for Arabic, reading each line several ways.
- **Final check of Arabic readings** by Claude Opus 5.5 (optional, needs an API key): three checks in parallel, combined by vote, shown as an *Enhanced* line beneath the reading with every changed letter marked, never replacing it.
- **Latin script and English** (optional): with an API key, a language model writes standard Latin spelling from the Ajami reading and translates it; without a key, a rule-based transliteration and NLLB-200 are used.
- **Learns from corrections** during a session ("Teach Iʿjām my corrections"), has a Stop button, works on phones, and handles up to three readings at once.
- About 45 seconds for a 15-line page on a T4 GPU.

### Running it on Google Colab

Choose a **T4 GPU** runtime, then run this cell. It asks for a Hugging Face **read** token (the trained models are in a private repository) and, optionally, a Claude API key.

```python
!pip install -q gradio anthropic 2>&1 | tail -1
!pip install -q virtualenv && virtualenv -q /content/kenv && /content/kenv/bin/pip install -q kraken 2>&1 | tail -1

import os, getpass
from huggingface_hub import snapshot_download

hf = getpass.getpass("Hugging Face token (hidden): ").strip()
key = getpass.getpass("Claude API key, sk-ant-... (hidden; press Enter to skip): ").strip()
if key:
    os.environ["ANTHROPIC_API_KEY"] = key

snapshot_download("Sakeenah/ajami-lens-models", token=hf, local_dir="/content/models")
!rm -rf /content/repo && git clone -q https://github.com/SakeenaB/ajami-htr.git /content/repo
!python /content/repo/src/lens.py --models /content/models --kraken-bin /content/kenv/bin/kraken --share
```

A public `https://….gradio.live` link appears when the app is ready (10–15 minutes on first start). On Kaggle, store `HF_TOKEN` and `ANTHROPIC_API_KEY` as Kaggle Secrets instead and use `/kaggle/temp` paths. Run `python src/lens.py --self-test` to run the application's self-tests.

## Exploratory extensions (validation half B only)

These were measured after the test sets had been scored, so they are reported on validation lines only.

| Extension | Result |
|---|---|
| Arabic: character language model in the vote | 10.83% → 9.35% CER |
| Arabic: + reading each line several ways | → 9.16% |
| Arabic: + final check by Claude Opus 5.5, guided by the recognisers' readings | → 7.65% (202 lines better, 15 worse) |
| Ajami: character language model | 9.80% → 9.57% |
| Ajami: final check by a large language model | no meaningful gain |
| Latin Hausa, 98 Infiraji lines vs a published Latin text | word error 59.96% (rules) → 34.45% (language model) |
| A native Hausa speaker's review, 30 held-out lines | Latin 4.84 / 5, English 4.78 / 5 |

## Repository

| Path | Contents |
|---|---|
| `src/dataset.py`, `src/augment.py` | Reading the collections, preparing and mirroring lines, dot-safe augmentation |
| `src/crnn.py`, `src/train.py`, `src/predict.py` | The CRNN, its training, and decoding with a confidence for every character |
| `src/trocr_vocab.py`, `src/trocr_codec.py`, `src/train_trocr.py` | Extending, encoding for and training TrOCR |
| `src/kraken_tools.py` | Preparing data for Kraken and reading its output |
| `src/bidi_order.py` | Reading order for lines that mix Arabic with numbers or Latin text |
| `src/alignment.py`, `src/ensemble.py` | The confusion network, the three meta-learners, selection on half B, test scoring |
| `src/metrics.py` | CER, WER, Ajami-mark error, paired bootstrap |
| `src/lm_vote.py` | Character language model in the vote; learning from corrections |
| `src/ajami2boko.py` | Rule-based Ajami to Latin-script Hausa, with word division |
| `src/lens.py` | The Iʿjām application |
| `Demo/` | Sample pages for trying Iʿjām |
| `showcase.html` | Before-and-after examples from the test set |
| `ensemble_results/` | Selection records and test results |

Each module runs its own self-tests before it is used, for example `python src/metrics.py`.

## Data

The data are not committed to this repository.

- **Ajami:** Yousuf et al. (2026), Zenodo record 15691686 (v1), CC BY 4.0. 6,206 transcribed line images from 29 manuscripts (5,400 Hausa, 806 Fulfulde), split 70/15/15 as in Al-azzawi et al. (2026).
- **Arabic:** Muharaf (Saeed et al., 2024), 24,495 transcribed lines from 1,216 pages, with the creators' own splits; released for non-commercial research.

## Limitations and future work

- The Ajami margin over the benchmark is very probable but not yet statistically certain.
- Fulfulde error is about twice Hausa error; more Fulfulde lines are the clearest fix.
- Kraken failed on grey lines (about 72% error). Binarised, it reached about 22% on validation, but its training was unstable, so it is not in the evaluated system. Retraining it stably and re-running the selection is the next step.
- Planned: retraining from users' saved corrections, stronger Arabic members such as HAFITH, an Ajami-aware language model, and a user study with scholars.

## References

- Al-azzawi, S., Barney, E., & Liwicki, M. (2026). CER-HV: A CER-based human-in-the-loop framework for cleaning datasets applied to Arabic-script HTR. arXiv:2601.16713.
- Saeed, M., Chan, A., Mijar, A., Moukarzel, J., Habchi, G., Younes, C., Elias, A., Wong, C.-W., & Khater, A. (2024). Muharaf: Manuscripts of handwritten Arabic dataset for cursive text recognition. NeurIPS 37 Datasets and Benchmarks Track.
- Yousuf, O., Aminu, A., Muhammad, M. S., Usman, B., Hashim, M. K., Nivre, J., Megyesi, B., & Høgel, C. (2026). A handwritten text recognition dataset for Ajami manuscripts in Fulfulde and Hausa. ICDAR 2025, LNCS 16026, 620–637. Springer.

The full reference list and method are in the project report.

## Acknowledgements

With thanks to Dr O. A. Abiola for supervision; to the creators of the Ajami and Muharaf collections for making their data public; to Dr Yemisi Olawale for her consultation on how historians use transcriptions; and to the Hausa speaker who reviewed the Latin script and English.
