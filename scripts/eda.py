import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.data import LABELS, load_raw

OUT = Path("data/outputs")
PLOTS = OUT / "plots"


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df["n_words"] = df["text"].str.split().map(len)
    counts = df["label_name"].value_counts().reindex(LABELS)
    imbalance = float(counts.max() / counts.min())
    summary = {
        "n_total": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_fractions": {k: round(float(v) / len(df), 4) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": round(imbalance, 1),
        "words_mean": round(float(df["n_words"].mean()), 1),
        "words_median": int(df["n_words"].median()),
        "words_p95": int(df["n_words"].quantile(0.95)),
        "majority_class_accuracy": round(float(counts.max()) / len(df), 4),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    ax.bar(LABELS, [counts[l] for l in LABELS], color=colors)
    ax.set_title(f"Class distribution (imbalance {imbalance:.0f}:1)")
    ax.set_ylabel("reviews")
    for i, l in enumerate(LABELS):
        ax.text(i, counts[l], f"{counts[l]:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "class_distribution.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_words"].clip(upper=400), bins=40, color="#337ab7")
    ax.axvline(
        summary["words_median"], color="k", ls="--", lw=1, label=f"median {summary['words_median']}"
    )
    ax.set_title("Review length (words, clipped at 400)")
    ax.set_xlabel("words")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "review_length.png", dpi=120)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
