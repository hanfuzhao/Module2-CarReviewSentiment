import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, render_template, request
from scripts import aspects as aspect_mod
from scripts.data import ID2LABEL, LABELS
from scripts.model import ClassicalModel, DeepModel, NaiveBaseline

MODELS_DIR = Path("models")
NAIVE_PATH = MODELS_DIR / "naive_majority.pkl"
CLASSICAL_PATH = MODELS_DIR / "classical_tfidf_logreg.pkl"
DEEP_SOURCE = os.environ.get("DEEP_MODEL", str(MODELS_DIR / "deep_distilbert"))
METRICS_PATH = Path("data/outputs/metrics.json")
CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_ACC = {"deep": 0.0, "classical": 0.0, "naive": 0.0}
SENTIMENT_COLORS = {"negative": "#d9534f", "neutral": "#f0ad4e", "positive": "#5cb85c"}


def _load_accuracies():
    acc = dict(_DEFAULT_ACC)
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text()).get("models", {})
        for k in acc:
            if k in m and "accuracy" in m[k]:
                acc[k] = m[k]["accuracy"]
    return acc


class PredictionService:

    def __init__(self):
        self.naive = NaiveBaseline.load(NAIVE_PATH) if NAIVE_PATH.exists() else None
        self.classical = ClassicalModel.load(CLASSICAL_PATH) if CLASSICAL_PATH.exists() else None
        self.deep = DeepModel.load(DEEP_SOURCE)
        self.deep.predict_proba(["warm up"])
        acc = _load_accuracies()
        self.models = {
            "deep": {
                "label": "Deep: fine-tuned DistilBERT",
                "accuracy": acc["deep"],
                "obj": self.deep,
            },
            "classical": {
                "label": "Classical: TF-IDF + logistic regression",
                "accuracy": acc["classical"],
                "obj": self.classical,
            },
            "naive": {
                "label": "Naive: majority class",
                "accuracy": acc["naive"],
                "obj": self.naive,
            },
        }

    def _proba(self, model_key, texts):
        return self.models[model_key]["obj"].predict_proba(texts)

    @staticmethod
    def _label(proba_row):
        i = int(np.argmax(proba_row))
        return {
            "label": ID2LABEL[i],
            "confidence": float(proba_row[i]),
            "probs": {ID2LABEL[j]: round(float(proba_row[j]), 4) for j in range(len(LABELS))},
            "color": SENTIMENT_COLORS[ID2LABEL[i]],
        }

    def _aspect_breakdown(self, model_key, text):
        items = aspect_mod.analyze(text)
        if not items:
            return []
        clauses = [it["clause"] for it in items]
        proba = self._proba(model_key, clauses)
        per_aspect = {}
        for it, row in zip(items, proba):
            lab = self._label(row)
            for asp in it["aspects"]:
                per_aspect.setdefault(asp, []).append({"clause": it["clause"], **lab})
        out = []
        for asp, mentions in per_aspect.items():
            avg = np.mean([[m["probs"][l] for l in LABELS] for m in mentions], axis=0)
            agg = self._label(np.array(avg))
            out.append(
                {
                    "aspect": asp,
                    "sentiment": agg["label"],
                    "color": agg["color"],
                    "confidence": agg["confidence"],
                    "n": len(mentions),
                    "mentions": mentions,
                }
            )
        order = {"negative": 0, "neutral": 1, "positive": 2}
        out.sort(key=lambda a: (order[a["sentiment"]], -a["n"]))
        return out

    def predict(self, text, model_key="deep"):
        if model_key not in self.models or self.models[model_key]["obj"] is None:
            model_key = "deep"
        meta = self.models[model_key]
        overall = self._label(self._proba(model_key, [text])[0])
        result = {
            "model": model_key,
            "model_label": meta["label"],
            "model_accuracy": meta["accuracy"],
            "overall": overall,
            "aspects": self._aspect_breakdown(model_key, text),
        }
        if model_key == "naive":
            result["feedback"] = {
                "level": "low_confidence",
                "message": "The majority-class baseline ignores the text and always predicts the most common label (positive).",
            }
        elif model_key == "deep" and overall["confidence"] < CONFIDENCE_THRESHOLD:
            result["feedback"] = {
                "level": "low_confidence",
                "message": f"Only {overall['confidence']:.0%} confident. The review may be mixed, sarcastic, or ambiguous - flagging for review.",
            }
        else:
            result["feedback"] = {
                "level": "confident",
                "message": f"{overall['confidence']:.0%} confident overall: {overall['label']}.",
            }
        return result


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
_service = None


def get_service():
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        ok = Path(DEEP_SOURCE).exists() or "/" in DEEP_SOURCE
        return jsonify({"status": "ok", "models": list(get_service().models.keys())})
    except Exception as e:
        return (jsonify({"status": "error", "detail": str(e)}), 500)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    model_key = data.get("model", "deep")
    if not text:
        return (jsonify({"error": "No text provided"}), 400)
    if len(text) > 5000:
        text = text[:5000]
    try:
        return jsonify(get_service().predict(text, model_key))
    except Exception as e:
        return (jsonify({"error": f"Inference failed: {e}"}), 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Loading models...")
    get_service()
    print(f"Ready. Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
