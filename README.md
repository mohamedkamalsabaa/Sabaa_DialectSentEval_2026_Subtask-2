# Sabaa - DialectSentEval 2026, Subtask 2

Code for the Sabaa submission to Subtask 2 of DialectSentEval 2026, Arabic
dialect sentiment swap: rewriting a sentence so its sentiment polarity is
inverted while its content is preserved.

## Results

Official leaderboard scores.

| Split | Sentiment accuracy | BLEU | chrF |
|---|---|---|---|
| Development | 0.9564 | 52.26 | 70.64 |
| Test | 0.7447 | 49.07 | 69.33 |

## Approach

Two Arabic sequence-to-sequence models - [AraT5v2-base-1024](https://huggingface.co/UBC-NLP/AraT5v2-base-1024)
and [AraBART](https://huggingface.co/moussaKam/AraBART) - are fine-tuned on
symmetrically augmented MA'AKS data. At inference each model proposes several
rewrites by beam search, and the pooled candidates are reranked by a composite
score with three terms:

- **sentiment** - confidence of [CAMeLBERT-DA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment),
  the task's own evaluation classifier, signed by whether the candidate landed
  on the target polarity;
- **consensus** - mean chrF against the other candidates in the pool, following
  the Minimum Bayes Risk principle, which suppresses truncations and
  hallucinations;
- **content** - chrF against the source, which is what makes minimal edits
  preferable to paraphrases.

**The test set withholds `source_polarity`**, so the swap direction has to be
inferred before anything can be rewritten. This turned out to dominate our
test-time error: the system reached the polarity it was *aiming* for 90.37% of
the time, but the direction it aimed at was only right about 80% of the time.
`03_supplementary_analysis.ipynb` measures this directly.

## Notebooks

| Notebook | Needs | Produces | Runtime |
|---|---|---|---|
| `01_development.ipynb` | Train + Val | dev `predictions.zip` and scores | ~2.5 h |
| `02_final_evaluation.ipynb` | Train + Test | test `predictions.zip` | ~2.5 h |
| `03_supplementary_analysis.ipynb` | polarity-inference analysis | ~2 min |

Each notebook is self-contained and runs top to bottom on Kaggle with **GPU
T4 ×2** and **internet enabled**. The `find_file` helper locates the
spreadsheets wherever they land under `/kaggle/input`, and the same notebooks
run unchanged on a `/workspace` mount.

Both training notebooks fine-tune from scratch: Kaggle clears `/kaggle/working`
between sessions, so no saved model survives. Within a single session you can
skip the training cell and go straight to loading.

## Configuration

Shared by both generators: max length 128, batch 4 with 2 gradient accumulation
steps, AdamW at 5e-5, 500 warmup steps, weight decay 0.01, 15 epochs, fp16.
Label smoothing is deliberately **off** — it lowers training loss but flattens
the output distribution, which measurably degrades the reranking this system
depends on.

Decoding differs between the two runs, and the notebooks preserve that
difference rather than hiding it, since each reproduces the run that produced
its reported numbers:

| | Candidates per model | Reranking weights | Source polarity |
|---|---|---|---|
| Development | 5 | 0.60 / 0.20 / 0.20 | given |
| Test (submitted) | 8 | 0.45 / 0.10 / 0.45 | inferred |

## Data

The MA'AKS corpus is distributed by the shared task organisers and is **not
included here**.

## Installation

```bash
pip install -r requirements.txt
```

## Notes for anyone reproducing this

Three implementation details cost us time and are worth knowing:

- **`save_strategy="no"` is deliberate.** The Trainer's periodic checkpointing
  serialises through safetensors, which rejects non-contiguous tensors and fails
  mid-run under fp16. The model is saved once at the end after an explicit
  `.contiguous()` pass.
- **CAMeLBERT-DA ships as a legacy `.bin` checkpoint.** transformers ≥ 5 refuses
  to `torch.load` it on PyTorch < 2.6 (CVE-2025-32434), while older transformers
  has no such check. `load_sentiment_classifier()` neutralises the guard only
  when it exists and restores it immediately.
- **The classifier returns lower-cased labels.** Comparisons against
  `Positive`/`Negative` need `.capitalize()`, or the sentiment term silently
  scores zero.

## Citation

If you use this code, please cite the shared task and the dataset:

```bibtex
@inproceedings{ezzini2026dialectsenteval,
  title = {DialectSentEval 2026: Arabic Dialect Sentiment Analysis and Swapping Shared Task},
  author = {Ezzini, Saad and Abudalfa, Shadi and Alharbi, Maram and Chafik, Salmane and Alatawi, Hind and El-Haj, Mo and Abdelali, Ahmed and Alnahari, Osamah and Lamsiyah, Salima},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference},
  year = {2026},
  address = {Budapest, Hungary},
  month = {October}
}

@article{mughaus2026ma,
  title = {Ma'aks: manually-curated parallel dataset for Arabic text sentiment swap},
  author = {Mughaus, Raed and Abudalfa, Shadi and Luqman, Hamzah and Abdu, Fahad and AlAli, Mohammed and Al-Dowayan, Nawaf and Abdelali, Ahmed},
  journal = {Language Resources and Evaluation},
  volume = {60}, number = {1}, pages = {1}, year = {2026},
  publisher = {Springer}
}
```

## License

MIT for the code in this repository. The MA'AKS corpus is governed by its own
licence from the shared task organisers.
