# Car Review Sentiment Analyzer

540 Summer, Module 2 Hackathon. Per-aspect sentiment analysis of real car owner reviews, with evaluation as the focus.

Live demo: https://HanfuZhao781-car-review-sentiment.hf.space
GitHub: https://github.com/hanfuzhao/Module2-CarReviewSentiment
Model: https://huggingface.co/HanfuZhao781/car-review-sentiment-distilbert

Platforms like Dongchedi aggregate thousands of owner reviews into per-dimension ratings for engine, comfort, value, reliability, and more, and buyers trust those ratings. If the model misreads sentiment, especially negation like "not good" or sarcasm like "great, another trip to the dealer", it flips an aggregated rating and misleads a purchase decision. This project builds that classifier and then spends most of its effort measuring where it breaks.

## What it does

Paste an owner review and get an overall sentiment of negative, neutral, or positive with a confidence score, plus a per-aspect breakdown. Each car aspect mentioned in the review is scored separately, so "powerful engine but terrible fuel economy" shows up as engine positive and fuel economy negative.

## The three required models

1. Naive baseline. Predicts the majority class, which is positive, and ignores the text. The accuracy floor. See `scripts/model.py::NaiveBaseline`.
2. Classical ML. TF-IDF with 1 to 2 grams plus class-weighted logistic regression. See `scripts/model.py::ClassicalModel`.
3. Deep learning, deployed. DistilBERT fine-tuned for 3-class sentiment via transfer learning. See `scripts/model.py::DeepModel`.

## Transfer learning and data augmentation

Transfer learning starts from `distilbert-base-uncased`, pretrained on general English, and fine-tunes a sentiment classification head on car reviews.

Data augmentation addresses the imbalance: the data is about 86 percent positive, so the negative and neutral classes are starved. We grow them with EDA (synonym replacement, random swap, random deletion; Wei and Zou 2019) and back-translation. We also measure where it helps. It lifts the minority classes in the low-resource cold-start regime, described in Results, but not at full data scale.

## Results, real, test set of 3,652 held-out reviews

| Model | Accuracy | Macro-F1 | Negative recall |
|---|---|---|---|
| Naive, majority | 86.0% | 30.8% | 0.0% |
| Classical, TF-IDF and LogReg | 85.4% | 57.2% | 38.4% |
| Deep, fine-tuned DistilBERT, deployed | 86.7% | 56.6% | 34.8% |

Why accuracy is misleading: because 86 percent of reviews are positive, a model that always says positive scores about 86 percent accuracy while being useless. The naive baseline proves it with 86.0% accuracy and 0.0% negative recall. The metrics that matter are macro-F1 and negative class recall, which is how well it catches dissatisfied owners. Transfer learning earns its keep: macro-F1 climbs from 30.8% for naive to 57.2% for classical to 56.6% for DistilBERT.

Evaluation shows where augmentation helps. The realistic Dongchedi scenario is a newly launched car with few reviews, the cold start. In that low-resource regime of 1,200 reviews, EDA augmentation clearly rescues the classical model:

| Classical, low-resource, 1,200 reviews | Macro-F1 | Negative recall | Neutral recall |
|---|---|---|---|
| Base, no augmentation | 31.9% | 0.9% | 0.7% |
| Plus EDA augmentation | 40.2% | 10.7% | 7.0% |

The fine-tuned transformer is already sample-efficient. Even at cold start, EDA leaves it essentially unchanged, with macro-F1 moving from 51.1% to 46.3%. Evaluation revealed both facts: augmentation is the right tool for a weak model on scarce data, and a needless one for the transformer we actually deploy.

Stress tests reveal the hard limits. The deployed model handles affirmative sentences at 70.0% but negated ones at 0.0%, and it misreads 100.0% of sarcasm as positive. These are mitigated in deployment by confidence-gated abstention: answering only above a confidence threshold lifts accuracy from 86.7% to 97.7% at 77.5% coverage, and routes the rest to human review.

Full numbers, confusion matrices, and stress-test breakdowns are produced by `python setup.py` and saved to `data/outputs/metrics.json`.

## Quick start

```bash
make install
make data
make eda
make train
make app
make fast
```

`make app` serves the web app at http://localhost:5000. `make fast` runs the whole pipeline on tiny subsets in a couple of minutes to confirm the plumbing.

The fine-tuned DistilBERT, about 268 MB, is not committed because of GitHub's 100 MB limit. `make train` reproduces it locally, and the deployed app loads it from the Hugging Face Hub. The small naive and classical models are committed under `models/`.

## Repository layout

```
main.py                 Flask inference app, overall and per-aspect sentiment
setup.py                trains all 3 models and runs every experiment
scripts/
  data.py               dataset access, label derivation, splits
  make_dataset.py       download, metadata, committed sample
  eda.py                class distribution and length stats
  aspects.py            aspect lexicon and clause segmentation
  augment.py            EDA and back-translation augmentation
  model.py              NaiveBaseline, ClassicalModel, DeepModel
  experiment.py         metrics, negation and sarcasm stress tests, abstention
  make_slides.py        builds PITCH.pptx from metrics.json and plots
templates/  static/     web UI
data/
  raw/edmunds/          metadata.json and committed sample.csv
  outputs/              metrics.json and plots
models/                 naive and classical, the deep model lives on the HF Hub
```

## Deployment

Deployed as a Docker Space on Hugging Face. The app runs inference only: the committed naive and classical models load from `models/`, the fine-tuned DistilBERT is pulled from the Hub at build time, and a scheduled GitHub Action pings `/health` so the Space stays awake. To run the container locally:

```bash
docker build -t car-sentiment .
docker run -p 7860:7860 car-sentiment
```

## Git workflow

Built across feature branches, each merged through a reviewed Pull Request into `main`. See the repository's Pull Requests tab.

## Data and citation

Edmunds consumer car reviews via the `florentgbelidji/car-reviews` Hugging Face mirror, about 37k reviews with 1 to 5 star ratings. Sentiment labels are derived by distant supervision from the star rating: 1 to 2 is negative, 3 is neutral, 4 to 5 is positive. EDA augmentation follows Wei and Zou, Easy Data Augmentation Techniques, EMNLP 2019.
