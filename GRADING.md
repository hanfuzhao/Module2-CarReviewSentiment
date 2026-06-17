# Grading guide, where each rubric item lives

A quick map from the hackathon requirements to this repo, for fast evaluation.

## 1. Pitch: problem statement, model and transfer learning, augmentation, results

| Pitch requirement | Where |
|---|---|
| Problem statement | [README](README.md) intro |
| Pre-trained model and transfer learning | DistilBERT fine-tuned in `scripts/model.py::DeepModel.fit` |
| Data augmentation techniques | `scripts/augment.py`, EDA and back-translation, applied in `setup.py` |
| Preliminary results | [README](README.md) Results, `data/outputs/metrics.json` |

The pitch itself is delivered separately as the recorded presentation, not stored in this repository.

## 2. Code and deployment

| Requirement | Where |
|---|---|
| GitHub repo, full codebase | this repo |
| GitHub best practices, branches and PRs | see the Pull Requests tab, every feature merged through a reviewed PR |
| Live deployed app | https://HanfuZhao781-car-review-sentiment.hf.space, a Docker Space |

## 3. NLP prototype with evaluation at the core

| Requirement | Where |
|---|---|
| An NLP approach | 3-class sentiment classification plus lexicon-routed aspect breakdown |
| At least 2 complementary metrics or stress tests | macro-F1 with per-class recall and confusion matrix, a negation stress test, a sarcasm and faint-praise stress test, and a confidence-gated abstention curve, all in `scripts/experiment.py` |
| Evaluation reveals limitations | imbalance leads to near-zero negative recall, negation flips predictions, sarcasm is read as positive, see `data/outputs/metrics.json` |
| Evaluation guides improvement | the low-resource augmentation study shows EDA lifting macro-F1 and minority recall at cold start for the classical model, and honestly not for the transformer, while confidence-gated abstention turns low confidence into a deployable safeguard |

## Reproduce everything

```bash
make install && make train
make app
```

`make fast` runs the whole pipeline on tiny subsets in a couple of minutes to confirm the plumbing without waiting for full training.

## The two complementary evaluation lenses

1. Aggregate metrics that survive class imbalance. Macro-F1 and negative-class recall, not raw accuracy, which an always-positive model games to about 86 percent.
2. Targeted linguistic stress tests. Curated negation and sarcasm sets that isolate specific failure modes the star-derived test set cannot, exposing exactly where understanding breaks down.

Together they tell the story the rubric asks for: evaluation reveals limitations of imbalance, negation, and sarcasm, and guides improvement through augmentation and abstention.
