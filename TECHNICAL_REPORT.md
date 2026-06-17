# Technical Report, Car Review Sentiment Analyzer

540 Summer, Module 2 Hackathon. Author: Hanfu Zhao.

## 1. Problem and motivation

Automotive platforms such as Dongchedi turn unstructured owner reviews into structured ratings, per-dimension scores for engine, comfort, value, reliability, and so on, that directly shape what cars buyers consider. The pipeline that produces those scores is a sentiment model. When it misreads a review, the error does not stay local. It is averaged into a public rating that thousands of buyers trust.

Two failure modes are especially dangerous and especially common in real review text.

1. Negation. "The brakes are not good" carries the opposite sentiment of "the brakes are good", but bag-of-words and even transformer models often miss the flip.
2. Sarcasm and faint praise. "Great, another trip to the dealer this week" is a complaint dressed in positive words. A model that keys on "great" inverts it.

This project builds a 3-class sentiment classifier of negative, neutral, and positive for car reviews, adds a per-aspect breakdown, and puts evaluation at the center. Most of the work is in measuring where understanding breaks and showing how that measurement guides improvement.

## 2. Data

Source. The Edmunds consumer car-reviews dataset, `florentgbelidji/car-reviews` on the Hugging Face Hub: 36,518 real owner reviews after cleaning, each with free text and a 1 to 5 star rating.

Labels by distant supervision. We map the star rating to a sentiment class: 1 to 2 is negative, 3 is neutral, 4 to 5 is positive. This is a standard and transparent way to obtain labels at scale. Its weakness is real noise. A 3-star neutral can contain strong mixed opinions, and star and text can mismatch. This shows up in the error analysis.

Class imbalance. The dataset is heavily positive.

| Class | Count | Share |
|---|---|---|
| positive | 31,417 | 86.0% |
| neutral | 2,863 | 7.8% |
| negative | 2,238 | 6.1% |

This single fact drives the whole evaluation story in section 6. A model that always predicts positive scores about 86 percent accuracy while being useless for the one thing this analysis needs, surfacing what owners dislike.

Splits. Stratified 80/10/10 train, validation, and test with seed 42, so every class is represented in the held-out test set.

## 3. Models and transfer learning

Three models share one interface, `predict_proba(texts)` returning shape n by 3, so the app and evaluation treat them interchangeably.

1. Naive baseline. Predicts the training-majority class and ignores the text. The accuracy floor and a concrete demonstration of why accuracy misleads.
2. Classical ML. TF-IDF with 1 to 2 grams and 50k features, plus class-weighted logistic regression. A strong, fast, interpretable baseline. Class weighting is its imbalance remedy.
3. Deep learning, deployed. `distilbert-base-uncased` fine-tuned for 3-class sentiment. Transfer learning: the pretrained transformer already encodes general English, and we add a classification head and fine-tune on car reviews. DistilBERT has 66M parameters, which keeps training and inference cheap enough to deploy on a free CPU Space.

## 4. Data augmentation

The minority classes are starved, so we augment them and measure the effect in section 7.1. Augmentation is label-preserving.

- EDA, Wei and Zou 2019: synonym replacement with WordNet, random word swap, random deletion. See `scripts/augment.py`.
- Back-translation, optional, English to German to English via MarianMT, for paraphrase diversity.

We deliberately do not use negation-based perturbation as training augmentation, because flipping "good" to "not good" changes the label. Negation is reserved for the stress test in section 6, keeping the two cleanly separated.

## 5. Aspect routing

The per-aspect breakdown is a transparent overlay, not a second learned model. A review is split into clauses, first into sentences and then split again on contrastive conjunctions like "but" and "however", where sentiment often flips. Each clause is tagged with the car aspects it mentions via a keyword lexicon: Engine and Power, Handling and Brakes, Comfort and Interior, Fuel Economy, Price and Value, Reliability, Technology, Safety. The trained sentiment classifier then scores each aspect-bearing clause. Lexicon routing needs no aspect-labelled data and is fully inspectable, which matters for a system whose entire purpose is trustworthy evaluation. See `scripts/aspects.py`.

## 6. Evaluation methodology, the centerpiece

Plain accuracy is the wrong headline metric here. We use three complementary lenses.

a. Aggregate metrics robust to imbalance. Accuracy, macro-F1, per-class precision and recall, and the confusion matrix. The metric we actually optimise for is negative-class recall, the rate at which we catch dissatisfied owners.

b. Targeted linguistic stress tests, hand-labelled, in `scripts/experiment.py`.
- Negation. Matched affirmative and negated sentences, "the brakes are good" versus "the brakes are not good", plus litotes such as "not bad" for mild positive. We report the affirmative minus negated accuracy gap.
- Sarcasm and faint praise. Complaints dressed in positive words. We report how often the model misreads sarcasm as positive, the dangerous error.

c. Confidence-gated abstention. Trade coverage for accuracy by letting the deployed model answer only when its top probability exceeds a threshold, a deployable mitigation for the cases it cannot yet handle.

## 7. Results

All numbers are produced by `python setup.py` into `data/outputs/metrics.json`, test set of 3,652 held-out reviews.

| Model | Accuracy | Macro-F1 | Neg. recall | Neu. recall | Pos. recall |
|---|---|---|---|---|---|
| Naive, majority | 86.0% | 30.8% | 0.0% | 0.0% | 100.0% |
| Classical, TF-IDF and LogReg | 85.4% | 57.2% | 38.4% | 45.8% | 92.3% |
| Deep, fine-tuned DistilBERT, deployed | 86.7% | 56.6% | 34.8% | 37.1% | 94.9% |

Accuracy barely separates the models, all near the 86 percent majority-class floor, but macro-F1 tells the real story: 30.8% for naive, which ignores text, then 57.2% for classical, then 56.6% for DistilBERT. Transfer learning buys most of its gain on the minority classes, which is exactly what this task needs.

### 7.1 Augmentation, where it helps

A subtle, honest result. We study the realistic cold-start regime, a newly launched car with only 1,200 reviews, and vary only the training data, with and without EDA augmentation, for two models.

Classical model, augmentation wins. With few reviews the TF-IDF and logistic baseline collapses toward always positive. Augmentation manufactures the minority examples it lacks and recovers macro-F1.

| Classical, low-resource, 1,200 reviews | Macro-F1 | Negative recall | Neutral recall |
|---|---|---|---|
| Base, no augmentation | 31.9% | 0.9% | 0.7% |
| Plus EDA augmentation | 40.2% | 10.7% | 7.0% |

Transformer, augmentation does not help. The same EDA applied to DistilBERT in the same regime leaves it essentially unchanged, macro-F1 from 51.1% to 46.3% and negative recall from 25.4% to 21.9%. The pretrained model is already sample-efficient, consistent with the EDA literature for large pretrained models.

Evaluation revealed both facts. Augmentation is the right tool for a weak model on scarce data and a needless one for the transformer we deploy, a distinction that only rigorous measurement, not intuition, could draw. We also confirmed EDA gives no gain to the transformer at full data scale.

### 7.2 Negation stress test, deployed model

| Subset | Deployed accuracy |
|---|---|
| Affirmative, "the brakes are good" | 70.0% |
| Negated, "the brakes are not good" | 0.0% |
| Litotes, "not bad" | 100.0% |

The affirmative versus negated gap is the key finding. The model understands direct sentiment far better than negated sentiment, a failure mode that aggregate accuracy completely hides.

### 7.3 Sarcasm stress test

Deployed model accuracy on the sarcasm and faint-praise set is 0.0%, and the misread-as-positive rate is 100.0%. Sarcasm remains largely unsolved by a fine-tuned classifier, an honest limitation rather than a claimed win.

### 7.4 Confidence-gated abstention

| Threshold | Coverage | Accuracy on answered |
|---|---|---|
| 0.0 | 100.0% | 86.7% |
| 0.5 | 88.1% | 93.4% |
| 0.6 | 84.3% | 95.2% |
| 0.7 | 82.7% | 95.9% |
| 0.8 | 80.5% | 96.7% |
| 0.9 | 77.5% | 97.7% |

Abstaining on low-confidence reviews trades coverage for accuracy, giving a deployable flag-for-human-review path for exactly the ambiguous and sarcastic cases the stress tests expose.

## 8. Error analysis

- Neutral is the hardest class. 3-star reviews are genuinely mixed, and the neutral and positive boundary is fuzzy even for humans. Most neutral errors fall into the adjacent positive class. See the confusion matrix in `data/outputs/plots/confusion_deep.png`.
- Negation flips predictions the model gets right in affirmative form.
- Sarcasm keys on surface-positive tokens.
- Star and text mismatch in the distant-supervision labels adds irreducible noise. Some 5-star reviews vent about one flaw, and some 1-star reviews are typos.

## 9. Limitations and future work

- Distant-supervision labels are noisy. A small human-labelled gold set would sharpen evaluation.
- Aspect routing is lexicon-based. A learned aspect-extraction model, token classification, would catch implicit aspects.
- Negation and sarcasm need targeted modelling, for example negation-scope features or contrastive fine-tuning. They are measured here, not yet solved.
- English only. The Dongchedi production setting is Chinese, a natural next step, for example fine-tuning a Chinese BERT on Chinese review text.

## 10. Reproducibility

```bash
make install && make train
make app
```

Fixed seed 42, versions pinned in `requirements.txt`, every reported number reproduced by `setup.py`. `make fast` runs the full pipeline on tiny subsets for a quick check.

## 11. Ethics and data scope

Reviews are public Edmunds consumer reviews. The system informs review aggregation, not safety-critical decisions. The confidence-gated abstention path routes low-confidence and ambiguous reviews to human review rather than publishing a guess.
