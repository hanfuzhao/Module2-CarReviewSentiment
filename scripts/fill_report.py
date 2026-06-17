import json
import re
from pathlib import Path

METRICS = Path("data/outputs/metrics.json")
DOCS = ["README.md", "TECHNICAL_REPORT.md", "PITCH.md"]


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def build_tokens(r):
    t = {}

    def add_model(prefix, m):
        pc = m["per_class"]
        t[f"{prefix}_ACC"] = pct(m["accuracy"])
        t[f"{prefix}_F1"] = pct(m["macro_f1"])
        t[f"{prefix}_NEGREC"] = pct(pc["negative"]["recall"])
        t[f"{prefix}_NEUREC"] = pct(pc["neutral"]["recall"])
        t[f"{prefix}_POSREC"] = pct(pc["positive"]["recall"])

    for k in ("naive", "classical", "deep"):
        add_model(k.upper(), r["models"][k])
    study = r["augmentation_study"]
    add_model("LRBASE", study["classical"]["base"])
    add_model("LRAUG", study["classical"]["aug"])
    add_model("DLRBASE", study["deep"]["base"])
    add_model("DLRAUG", study["deep"]["aug"])
    ae = r["augmentation_effect"]
    t["AUG_F1_BEFORE"] = pct(ae["macro_f1"]["before"])
    t["AUG_F1_AFTER"] = pct(ae["macro_f1"]["after"])
    t["AUG_NEGREC_BEFORE"] = pct(ae["negative_recall"]["before"])
    t["AUG_NEGREC_AFTER"] = pct(ae["negative_recall"]["after"])
    t["AUG_NEUREC_BEFORE"] = pct(ae["neutral_recall"]["before"])
    t["AUG_NEUREC_AFTER"] = pct(ae["neutral_recall"]["after"])
    aed = r["augmentation_effect_deep"]
    t["DEEPAUG_F1_BEFORE"] = pct(aed["macro_f1"]["before"])
    t["DEEPAUG_F1_AFTER"] = pct(aed["macro_f1"]["after"])
    t["DEEPAUG_NEGREC_BEFORE"] = pct(aed["negative_recall"]["before"])
    t["DEEPAUG_NEGREC_AFTER"] = pct(aed["negative_recall"]["after"])
    neg = r["stress_tests"]["negation"]["deep"]
    t["NEG_AFF_DEEP"] = pct(neg["affirmative_accuracy"])
    t["NEG_NEG_DEEP"] = pct(neg["negated_accuracy"])
    t["NEG_LIT_DEEP"] = pct(neg["litotes_accuracy"])
    sar = r["stress_tests"]["sarcasm"]["deep"]
    t["SARC_ACC_DEEP"] = pct(sar["accuracy"])
    t["SARC_POS_DEEP"] = pct(sar["misread_as_positive_rate"])
    rows, best = ([], None)
    for c in r["confidence_gating"]:
        if c["accuracy"] is None:
            continue
        rows.append(f"| {c['threshold']:.1f} | {pct(c['coverage'])} | {pct(c['accuracy'])} |")
        if c["coverage"] >= 0.5 and (best is None or c["accuracy"] > best[0]):
            best = (c["accuracy"], c["coverage"])
    t["GATING_ROWS"] = "\n".join(rows)
    t["GATING_PEAK"] = f"{pct(best[0])} (covering {pct(best[1])})" if best else "n/a"
    t["GATING_BASE"] = pct(r["confidence_gating"][0]["accuracy"])
    t["TRAIN_SIZE"] = f"{r['config']['train_size']:,}"
    t["TEST_SIZE"] = f"{r['config']['test_size']:,}"
    t["LOWRES_CAP"] = f"{r['config']['lowres_cap']:,}"
    return t


def main():
    r = json.loads(METRICS.read_text())
    tokens = build_tokens(r)
    for doc in DOCS:
        p = Path(doc)
        if not p.exists():
            continue
        text = p.read_text()
        for k, v in tokens.items():
            text = text.replace(f"__{k}__", v)
        leftover = sorted(set(re.findall("__[A-Z0-9_]+__", text)))
        p.write_text(text)
        status = f"OK ({len(tokens)} tokens)" if not leftover else f"LEFTOVER: {leftover}"
        print(f"{doc:24s} {status}")


if __name__ == "__main__":
    main()
