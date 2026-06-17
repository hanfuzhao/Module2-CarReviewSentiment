# 5-Minute Pitch Script, hard stop at 5:00 including the demo

Author: Hanfu Zhao
Live demo: https://HanfuZhao781-car-review-sentiment.hf.space
Code: https://github.com/hanfuzhao/Module2-CarReviewSentiment

This script is written to run in about 4 minutes 45 seconds at a normal pace, demo included, leaving a 15 second buffer. Lines marked Action are what you do, not what you say. Open the live app before you start and leave it on a second tab.

Time budget, total 5:00:
- Problem: 0:00 to 0:40
- Approach: 0:40 to 1:20
- Augmentation: 1:20 to 1:55
- Results: 1:55 to 3:15
- Demo: 3:15 to 4:45
- Buffer: 4:45 to 5:00

## Problem, 0:00 to 0:40

I am Hanfu. I have an offer at Dongchedi, an automotive platform, so both my hackathon projects point there. Module 1 read cars from photos. This one reads what owners say. Platforms like Dongchedi turn thousands of reviews into word of mouth ratings for engine, comfort, and value that buyers trust. Those come from a sentiment model, and sentiment has two hard failure modes: negation, the brakes are not good, and sarcasm, great, another trip to the dealer. Get them wrong and a bad rating reaches thousands of buyers. So this is a classifier where the real work is measuring where it breaks.

## Approach, 0:40 to 1:20

I built three models on 36,000 real Edmunds reviews. A naive baseline that always predicts the majority class, the floor. A classical model, TF-IDF with logistic regression. And the deployed model, transfer learning: DistilBERT pretrained on English, fine-tuned with a sentiment head on car reviews. It is small enough to run on a free CPU server. On top, a per-aspect breakdown splits a review into clauses and scores each aspect on its own.

## Augmentation, 1:20 to 1:55

One catch: the data is 86 percent positive. A model that always says positive scores 86 percent accuracy and is useless. So I augment the minority classes with EDA, synonym swaps and deletions, plus back-translation. The interesting question is when that actually helps, and evaluation answered it for me.

## Results, 1:55 to 3:15

Three findings.

First, accuracy lies. The naive baseline gets 86 percent accuracy with zero negative recall. So I judge on macro-F1 and negative recall, where transfer learning is clear: macro-F1 goes 31 percent for naive, 57 for classical, and 57 for DistilBERT.

Second, evaluation told me when augmentation matters. In a cold-start case, a new car with only 1,200 reviews, the classical model collapses, and EDA lifts its macro-F1 from 32 to 40 percent. But the same augmentation does nothing for the transformer, which is already strong. Same technique, opposite verdict.

Third, stress tests expose what is still broken. On plain sentences the model scores 70 percent, but on negated ones it drops to zero, and it reads all of the sarcasm as positive. I do not hide that. I mitigate it with confidence-gated abstention: answering only when confident raises accuracy to 98 percent and flags the rest for a human.

## Demo, 3:15 to 4:45

Action: switch to the live app.

Now live. One review, full breakdown.

Action: click the Mixed review example.

Engine positive, fuel economy negative, price negative, comfort positive, all from one review. That is exactly the word of mouth view a platform needs.

Action: click the Litotes example, then switch the model dropdown to Classical.

Model gap, live. Not bad, not uncomfortable. The deep model says positive, correctly. The classical model sees the words bad and uncomfortable and says negative.

Action: switch back to Deep, click the Negation example.

Honesty. On heavy negation the model drops to fifty percent, and the app flags it for review instead of bluffing.

Action: click the Sarcasm example.

And the frontier. Great, another trip to the dealer. Confidently wrong, ninety-nine percent positive. My stress test measures exactly this, which is why the pipeline keeps a human in the loop. Accurate, honest, and deployable. Thank you.

## If you are running behind
- At 3:15 you must be starting the demo. If late, cut the classical model switch and go straight to negation, then sarcasm.
- At 4:45 say the final line. Never run past 5:00.
- If the app is slow, keep talking through Results. Never wait in silence.

## Q and A backup facts
- Distant-supervision labels give 37k labelled reviews for free from star ratings. The star and text mismatch is real noise, quantified in the report.
- DistilBERT over full BERT: about 40 percent smaller and faster, fits a free CPU Space and the time budget. Full BERT is the next step.
- Lexicon aspects, not learned: no aspect-labelled data, and a transparent overlay suits a project about trustworthy evaluation.
- Production is Chinese, so the next step is a Chinese BERT on Chinese review text.
