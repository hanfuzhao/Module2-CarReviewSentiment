import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pickle
from pathlib import Path
import numpy as np
from scripts.data import ID2LABEL, LABELS

MAX_LEN = 128
HF_BACKBONE = "distilbert-base-uncased"


class NaiveBaseline:

    def __init__(self):
        self.priors = None
        self.majority = None

    def fit(self, labels):
        counts = np.bincount(np.asarray(labels), minlength=len(LABELS)).astype(float)
        self.priors = counts / counts.sum()
        self.majority = int(self.priors.argmax())
        return self

    def predict_proba(self, texts):
        return np.tile(self.priors, (len(texts), 1))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ClassicalModel:

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_features=50000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
            ]
        )

    def fit(self, texts, labels):
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _best_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DeepModel:

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._device = None

    def fit(
        self,
        train_df,
        val_df=None,
        epochs=3,
        batch_size=16,
        lr=2e-05,
        class_weights=None,
        out_dir="models/deep_distilbert",
    ):
        import torch
        from torch import nn
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(HF_BACKBONE)
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_BACKBONE,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )

        def encode(df):
            enc = tokenizer(list(df["text"]), truncation=True, max_length=MAX_LEN)
            return _DictDataset(enc, list(df["label"].astype(int)))

        train_ds = encode(train_df)
        eval_ds = encode(val_df) if val_df is not None else None
        weights = (
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        )

        class WeightedTrainer(Trainer):

            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                w = weights.to(outputs.logits.device) if weights is not None else None
                loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        from transformers import DataCollatorWithPadding

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=100,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            seed=42,
        )
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        trainer.train()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        self.model, self.tokenizer = (model, tokenizer)
        return self

    def _ensure_device(self):
        if self._device is None:
            self._device = _best_device()
            self.model.to(self._device)
            self.model.eval()

    def predict_proba(self, texts, batch_size=32):
        import torch

        self._ensure_device()
        probs = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    @staticmethod
    def load(path="models/deep_distilbert"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return DeepModel(model=model, tokenizer=tokenizer)


class _DictDataset:

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
