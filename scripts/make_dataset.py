import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import pandas as pd
from scripts.data import DATASET_ID, LABELS, METADATA_PATH, SAMPLE_CSV, SEED, get_splits, load_raw

SAMPLE_PER_CLASS = 60


def main():
    df = load_raw(force=True)
    train, val, test = get_splits(force=True)
    counts = df["label_name"].value_counts().reindex(LABELS).fillna(0).astype(int)
    sample = (
        df.groupby("label_name", group_keys=False)
        .apply(lambda g: g.sample(min(SAMPLE_PER_CLASS, len(g)), random_state=SEED))
        .reset_index(drop=True)
    )
    sample[["text", "rating", "label_name", "vehicle"]].to_csv(SAMPLE_CSV, index=False)
    metadata = {
        "dataset_id": DATASET_ID,
        "source": "Edmunds consumer car reviews, Hugging Face mirror",
        "labels": LABELS,
        "label_rule": "rating 1 to 2 is negative, 3 is neutral, 4 to 5 is positive",
        "n_total": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "splits": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "seed": SEED,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Total reviews:       {len(df):,}")
    print(f"Class counts:        {metadata['class_counts']}")
    print(f"Splits (tr/val/te):  {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"Committed sample:    {SAMPLE_CSV} ({len(sample)} rows)")
    print(f"Metadata:            {METADATA_PATH}")


if __name__ == "__main__":
    main()
