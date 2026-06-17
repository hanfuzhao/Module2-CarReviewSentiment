import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATASET_ID = "florentgbelidji/car-reviews"
RAW_DIR = Path("data/raw/edmunds")
FULL_CSV = RAW_DIR / "reviews_full.csv"
SAMPLE_CSV = RAW_DIR / "sample.csv"
SPLIT_DIR = Path("data/processed")
METADATA_PATH = RAW_DIR / "metadata.json"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
SEED = 42
MIN_CHARS = 15
MAX_CHARS = 2000


def label_from_rating(rating: int) -> int:
    if rating <= 2:
        return LABEL2ID["negative"]
    if rating == 3:
        return LABEL2ID["neutral"]
    return LABEL2ID["positive"]


def clean_text(text: str) -> str:
    text = re.sub("\\s+", " ", str(text)).strip()
    return text[:MAX_CHARS]


def load_raw(force: bool = False) -> pd.DataFrame:
    if FULL_CSV.exists() and (not force):
        return pd.read_csv(FULL_CSV)
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    df = df.rename(columns={"Review": "text", "Rating": "rating", "Vehicle_Title": "vehicle"})
    df = df[["text", "rating", "vehicle"]].copy()
    df["text"] = df["text"].map(clean_text)
    df["vehicle"] = df["vehicle"].fillna("").map(lambda s: re.sub("\\s+", " ", str(s)).strip())
    df = df[df["text"].str.len() >= MIN_CHARS]
    df = df[df["rating"].between(1, 5)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = df["rating"].astype(int).map(label_from_rating)
    df["label_name"] = df["label"].map(ID2LABEL)
    df.to_csv(FULL_CSV, index=False)
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    rng = np.random.RandomState(seed)
    train, val, test = ([], [], [])
    for _, grp in df.groupby("label"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * 0.1))
        n_val = int(round(n * 0.1))
        test.append(grp.loc[idx[:n_test]])
        val.append(grp.loc[idx[n_test : n_test + n_val]])
        train.append(grp.loc[idx[n_test + n_val :]])
    out = [
        pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for parts in (train, val, test)
    ]
    return (out[0], out[1], out[2])


def get_splits(force: bool = False):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {s: SPLIT_DIR / f"{s}.csv" for s in ("train", "val", "test")}
    if all((p.exists() for p in paths.values())) and (not force):
        return tuple((pd.read_csv(paths[s]) for s in ("train", "val", "test")))
    df = load_raw(force=force)
    train, val, test = make_splits(df)
    for s, d in zip(("train", "val", "test"), (train, val, test)):
        d.to_csv(paths[s], index=False)
    return (train, val, test)


def class_weights(train: pd.DataFrame) -> np.ndarray:
    counts = train["label"].value_counts().sort_index().to_numpy().astype(float)
    w = counts.sum() / (len(counts) * counts)
    return w


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} reviews")
    print(df["label_name"].value_counts())
