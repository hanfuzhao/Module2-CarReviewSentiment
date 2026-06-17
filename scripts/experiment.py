import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
from scripts.data import LABEL2ID, LABELS

PLOTS = Path("data/outputs/plots")
NEGATION_CASES = [
    ("The brakes are excellent and very responsive.", "positive", "affirmative"),
    ("This engine is powerful and smooth.", "positive", "affirmative"),
    ("The fuel economy is great for the class.", "positive", "affirmative"),
    ("The interior is comfortable and quiet.", "positive", "affirmative"),
    ("It has been reliable for years.", "positive", "affirmative"),
    ("The handling is precise and confident.", "positive", "affirmative"),
    ("The brakes are terrible and unsafe.", "negative", "affirmative"),
    ("The engine is weak and noisy.", "negative", "affirmative"),
    ("The ride is uncomfortable and harsh.", "negative", "affirmative"),
    ("It has been unreliable since day one.", "negative", "affirmative"),
    ("The brakes are not good at all.", "negative", "negated"),
    ("This engine is not powerful or smooth.", "negative", "negated"),
    ("The fuel economy is not great for the class.", "negative", "negated"),
    ("The interior is not comfortable and never quiet.", "negative", "negated"),
    ("It has not been reliable at all.", "negative", "negated"),
    ("The handling is not precise or confident.", "negative", "negated"),
    ("I would not call this car reliable.", "negative", "negated"),
    ("I can't say the seats are comfortable.", "negative", "negated"),
    ("Don't expect good fuel economy from this.", "negative", "negated"),
    ("There is no way this is worth the price.", "negative", "negated"),
    ("The brakes are not bad at all.", "positive", "litotes"),
    ("The ride is not uncomfortable.", "positive", "litotes"),
    ("It is not an unreliable car.", "positive", "litotes"),
    ("You won't be disappointed by the engine.", "positive", "litotes"),
]
SARCASM_CASES = [
    ("Great, another trip to the dealer this week.", "negative"),
    ("I just love spending every weekend fixing this thing.", "negative"),
    ("Sure, it's reliable, if you enjoy walking home.", "negative"),
    ("Wonderful, the AC broke again in the first month.", "negative"),
    ("Oh fantastic, another warning light on the dash.", "negative"),
    ("What a steal, it only cost me two tow trucks.", "negative"),
    ("Brilliant engineering, the door handle fell off.", "negative"),
    ("Yeah, the fuel economy is amazing if you never drive it.", "negative"),
    ("Best part of owning this car is the time at the mechanic.", "negative"),
    ("Such a smooth ride, my coffee ends up on the ceiling.", "negative"),
    ("It's fine, I guess, nothing special.", "neutral"),
    ("Not the worst car I have owned.", "neutral"),
    ("It does the job, more or less.", "neutral"),
    ("Decent enough for the price, but that's about it.", "neutral"),
]


def _predict_labels(model, texts):
    return model.predict_proba(list(texts)).argmax(axis=1)


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            l: {
                "precision": round(rep[l]["precision"], 4),
                "recall": round(rep[l]["recall"], 4),
                "f1": round(rep[l]["f1-score"], 4),
            }
            for l in LABELS
        },
        "negative_recall": round(rep["negative"]["recall"], 4),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))
        ).tolist(),
    }


def evaluate_model(model, df):
    y_true = df["label"].astype(int).to_numpy()
    y_pred = _predict_labels(model, df["text"].tolist())
    return metrics(y_true, y_pred)


def run_negation(model):
    texts = [t for t, _, _ in NEGATION_CASES]
    gold = np.array([LABEL2ID[g] for _, g, _ in NEGATION_CASES])
    groups = [grp for _, _, grp in NEGATION_CASES]
    pred = _predict_labels(model, texts)

    def acc(mask):
        m = np.array(mask)
        return round(float((pred[m] == gold[m]).mean()), 4) if m.any() else None

    aff = [g == "affirmative" for g in groups]
    neg = [g == "negated" for g in groups]
    lit = [g == "litotes" for g in groups]
    return {
        "overall_accuracy": round(float((pred == gold).mean()), 4),
        "affirmative_accuracy": acc(aff),
        "negated_accuracy": acc(neg),
        "litotes_accuracy": acc(lit),
        "affirmative_minus_negated_gap": round((acc(aff) or 0) - (acc(neg) or 0), 4),
        "examples": [
            {"text": t, "gold": g, "pred": LABELS[p]} for (t, g, _), p in zip(NEGATION_CASES, pred)
        ],
    }


def run_sarcasm(model):
    texts = [t for t, _ in SARCASM_CASES]
    gold = np.array([LABEL2ID[g] for _, g in SARCASM_CASES])
    pred = _predict_labels(model, texts)
    false_positive_rate = round(float((pred == LABEL2ID["positive"]).mean()), 4)
    return {
        "accuracy": round(float((pred == gold).mean()), 4),
        "misread_as_positive_rate": false_positive_rate,
        "examples": [
            {"text": t, "gold": g, "pred": LABELS[p]} for (t, g), p in zip(SARCASM_CASES, pred)
        ],
    }


def confidence_gating(model, df, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    proba = model.predict_proba(df["text"].tolist())
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    y = df["label"].astype(int).to_numpy()
    curve = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        acc = float((pred[keep] == y[keep]).mean()) if keep.any() else None
        curve.append(
            {
                "threshold": t,
                "coverage": round(cov, 4),
                "accuracy": round(acc, 4) if acc is not None else None,
            }
        )
    return curve


def plot_confusion(cm, title, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=9,
            )
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=120)
    plt.close(fig)


def plot_abstention(curve, fname="abstention_curve.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    cov = [c["coverage"] for c in curve if c["accuracy"] is not None]
    acc = [c["accuracy"] for c in curve if c["accuracy"] is not None]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cov, acc, "o-", color="#337ab7")
    for c in curve:
        if c["accuracy"] is not None:
            ax.annotate(
                f"thr={c['threshold']}",
                (c["coverage"], c["accuracy"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("accuracy on answered")
    ax.set_title("Confidence-gated abstention")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=120)
    plt.close(fig)
