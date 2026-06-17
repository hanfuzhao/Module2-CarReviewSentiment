# Deployment runbook

Two artifacts ship: the fine-tuned model to the HF Hub, and the app to an HF Docker Space. GitHub holds the code.

## 0. Prerequisites
- `gh auth status` shows logged in, GitHub hanfuzhao
- `hf auth whoami` shows logged in, HF HanfuZhao781
- `python setup.py` has produced `models/deep_distilbert/`

## 1. Push the fine-tuned model to the HF Hub
```bash
hf upload HanfuZhao781/car-review-sentiment-distilbert models/deep_distilbert . \
  --repo-type=model --commit-message "Fine-tuned DistilBERT for car-review sentiment"
```
The app loads it via the `DEEP_MODEL` env var, set in the Dockerfile.

## 2. Create the GitHub repo and push, with PR history
```bash
gh repo create hanfuzhao/Module2-CarReviewSentiment --public --source=. --remote=origin --push
```
Feature branches were merged through PRs into main. See the PR tab.

## 3. Create the HF Docker Space and deploy
```bash
hf repo create HanfuZhao781/car-review-sentiment --repo-type=space --space-sdk docker
git remote add space https://huggingface.co/spaces/HanfuZhao781/car-review-sentiment
git push space main
```
Live at https://HanfuZhao781-car-review-sentiment.hf.space

The Space README needs Docker frontmatter as the first lines:
```yaml
---
title: Car Review Sentiment
emoji: car
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

## 4. Keep it awake
`.github/workflows/keep-alive.yml` pings `/health` every 30 minutes. Enable Actions on the GitHub repo, which happens automatically once pushed.

## Notes
- The model, about 268 MB, is not in GitHub because of the 100 MB limit. It lives on the Hub and is pulled at Docker build time.
- Local dev loads the model from `models/deep_distilbert/`. The Space loads it from the Hub via `DEEP_MODEL=HanfuZhao781/car-review-sentiment-distilbert`.
