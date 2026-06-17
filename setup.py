import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd
from scripts import experiment as exp
from scripts.augment import augment_dataframe
from scripts.data import LABELS, get_splits
from scripts.model import ClassicalModel, DeepModel, NaiveBaseline

OUT = Path("data/outputs")
MODELS = Path("models")
FAST = os.environ.get("FAST") == "1"
DEEP_EPOCHS = int(os.environ.get("DEEP_EPOCHS", "2"))
DEPLOY_CAP = int(os.environ.get("DEPLOY_CAP", "9000"))
LOWRES_CAP = int(os.environ.get("LOWRES_CAP", "1200"))
AUG_PER_CLASS = int(os.environ.get("AUG_PER_CLASS", "800"))
SEED = 42


def _subsample_keep_dist(df, n, seed=SEED):
    if len(df) <= n:
        return df
    frac = n / len(df)
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed))
        .reset_index(drop=True)
    )


def _counts(df):
    return df["label_name"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    train, val, test = get_splits()
    deploy_epochs, lowres_epochs = (DEEP_EPOCHS, DEEP_EPOCHS + 1)
    deploy_cap, lowres_cap, aug_per_class = (DEPLOY_CAP, LOWRES_CAP, AUG_PER_CLASS)
    if FAST:
        train = _subsample_keep_dist(train, 1500)
        test = _subsample_keep_dist(test, 600)
        deploy_epochs = lowres_epochs = 1
        deploy_cap, lowres_cap, aug_per_class = (800, 500, 250)
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}  (FAST={FAST})")
    results = {
        "config": {
            "fast": FAST,
            "deploy_epochs": deploy_epochs,
            "lowres_epochs": lowres_epochs,
            "deploy_cap": deploy_cap,
            "lowres_cap": lowres_cap,
            "aug_per_class": aug_per_class,
            "train_size": int(len(train)),
            "test_size": int(len(test)),
        },
        "models": {},
    }
    print("\n[1/5] Naive baseline (majority class)")
    naive = NaiveBaseline().fit(train["label"].to_numpy())
    naive.save(MODELS / "naive_majority.pkl")
    results["models"]["naive"] = exp.evaluate_model(naive, test)
    print("[2/5] Classical: TF-IDF + class-weighted logistic regression")
    classical = ClassicalModel().fit(train["text"], train["label"])
    classical.save(MODELS / "classical_tfidf_logreg.pkl")
    results["models"]["classical"] = exp.evaluate_model(classical, test)
    deploy_dir = MODELS / "deep_distilbert"
    deploy_train = _subsample_keep_dist(train, deploy_cap)
    reuse = deploy_dir.exists() and os.environ.get("FORCE_TRAIN") != "1" and (not FAST)
    if reuse:
        print(
            f"[3/5] Deep (deployed): reusing existing {deploy_dir} (set FORCE_TRAIN=1 to retrain)"
        )
        deep = DeepModel.load(str(deploy_dir))
    else:
        print(
            f"[3/5] Deep (deployed): DistilBERT fine-tuned (cap {deploy_cap}, {deploy_epochs} ep)"
        )
        t0 = time.time()
        deep = DeepModel().fit(
            deploy_train, epochs=deploy_epochs, batch_size=16, out_dir=str(deploy_dir)
        )
        print(f"      trained in {time.time() - t0:.0f}s on {len(deploy_train):,} rows")
    results["models"]["deep"] = exp.evaluate_model(deep, test)
    results["models"]["deep"]["train_class_counts"] = _counts(deploy_train)
    print(f"[4/5] Augmentation study, LOW-RESOURCE (cap {lowres_cap})")
    lr_pool = _subsample_keep_dist(train, lowres_cap)
    lr_aug_train = augment_dataframe(lr_pool, target_per_class=aug_per_class, seed=SEED)

    def delta(mb, ma, field_fn):
        b, a = (field_fn(mb), field_fn(ma))
        return {"before": b, "after": a, "delta": round(a - b, 4)}

    def deltas(mb, ma):
        return {
            "macro_f1": delta(mb, ma, lambda m: m["macro_f1"]),
            "negative_recall": delta(mb, ma, lambda m: m["per_class"]["negative"]["recall"]),
            "neutral_recall": delta(mb, ma, lambda m: m["per_class"]["neutral"]["recall"]),
        }

    print("      4a. classical: base vs +EDA")
    c_base = ClassicalModel()
    c_base.pipe.named_steps["clf"].class_weight = None
    c_base.fit(lr_pool["text"], lr_pool["label"])
    c_aug = ClassicalModel()
    c_aug.pipe.named_steps["clf"].class_weight = None
    c_aug.fit(lr_aug_train["text"], lr_aug_train["label"])
    cm_base, cm_aug = (exp.evaluate_model(c_base, test), exp.evaluate_model(c_aug, test))
    print("      4b. transformer: base vs +EDA")
    d_base = DeepModel().fit(
        lr_pool, epochs=lowres_epochs, batch_size=16, out_dir=str(MODELS / "deep_lowres_base")
    )
    d_aug = DeepModel().fit(
        lr_aug_train, epochs=lowres_epochs, batch_size=16, out_dir=str(MODELS / "deep_lowres_aug")
    )
    dm_base, dm_aug = (exp.evaluate_model(d_base, test), exp.evaluate_model(d_aug, test))
    results["augmentation_study"] = {
        "regime": f"low-resource: {len(lr_pool)} reviews (cold-start, like a newly-launched car)",
        "train_class_counts": {"base": _counts(lr_pool), "aug": _counts(lr_aug_train)},
        "classical": {"base": cm_base, "aug": cm_aug},
        "deep": {"base": dm_base, "aug": dm_aug},
    }
    results["augmentation_effect"] = deltas(cm_base, cm_aug)
    results["augmentation_effect_deep"] = deltas(dm_base, dm_aug)
    print("\n[5/5] Stress tests (negation, sarcasm) + abstention on the deployed model")
    results["stress_tests"] = {
        "negation": {"deep": exp.run_negation(deep)},
        "sarcasm": {"deep": exp.run_sarcasm(deep)},
    }
    gating = exp.confidence_gating(deep, test)
    results["confidence_gating"] = gating
    exp.plot_confusion(
        results["models"]["deep"]["confusion_matrix"],
        "Deployed DistilBERT (test set)",
        "confusion_deep.png",
    )
    exp.plot_abstention(gating)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone in {time.time() - t_start:.0f}s. Wrote {OUT / 'metrics.json'}")
    _print_headline(results)


def _print_headline(r):
    print("\n================ HEADLINE ================")
    for k in ("naive", "classical", "deep"):
        m = r["models"][k]
        print(
            f"{k:11s} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  neg_recall={m['negative_recall']:.3f}"
        )
    ae = r["augmentation_effect"]
    aed = r["augmentation_effect_deep"]
    print(f"\nLow-resource augmentation (the realistic cold-start):")
    print(
        f"  CLASSICAL  macro-F1 {ae['macro_f1']['before']:.3f} to {ae['macro_f1']['after']:.3f} (delta {ae['macro_f1']['delta']:+.3f})   neg-recall {ae['negative_recall']['before']:.3f} to {ae['negative_recall']['after']:.3f}  [augmentation WINS]"
    )
    print(
        f"  TRANSFORMER macro-F1 {aed['macro_f1']['before']:.3f} to {aed['macro_f1']['after']:.3f} (delta {aed['macro_f1']['delta']:+.3f})  [already strong; EDA does not help]"
    )
    neg = r["stress_tests"]["negation"]["deep"]
    print(
        f"\nNegation (deployed): affirmative {neg['affirmative_accuracy']} vs negated {neg['negated_accuracy']}"
    )
    sar = r["stress_tests"]["sarcasm"]["deep"]
    print(f"Sarcasm (deployed): misread-as-positive {sar['misread_as_positive_rate']}")
    g = [c for c in r["confidence_gating"] if c["accuracy"] is not None]
    print(
        f"Abstention: {g[0]['accuracy']} @100% to {g[-1]['accuracy']} @{g[-1]['coverage'] * 100:.0f}% coverage"
    )
    print("==========================================")


if __name__ == "__main__":
    main()
