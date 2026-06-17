import re

ASPECTS = {
    "Engine & Power": [
        "engine",
        "power",
        "powerful",
        "acceleration",
        "accelerate",
        "horsepower",
        "hp",
        "torque",
        "motor",
        "transmission",
        "gearbox",
        "shift",
        "shifting",
        "turbo",
        "throttle",
        "fast",
        "quick",
        "sluggish",
        "responsive",
    ],
    "Handling & Brakes": [
        "handling",
        "handle",
        "steering",
        "steer",
        "brake",
        "brakes",
        "braking",
        "corner",
        "cornering",
        "suspension",
        "ride",
        "grip",
        "stable",
        "stability",
    ],
    "Comfort & Interior": [
        "comfort",
        "comfortable",
        "seat",
        "seats",
        "seating",
        "interior",
        "cabin",
        "quiet",
        "noise",
        "noisy",
        "legroom",
        "spacious",
        "roomy",
        "cramped",
        "trunk",
        "cargo",
        "space",
        "materials",
        "fit and finish",
    ],
    "Fuel Economy": [
        "mpg",
        "fuel",
        "gas",
        "gasoline",
        "mileage",
        "economy",
        "economical",
        "consumption",
        "efficient",
        "efficiency",
        "range",
        "guzzler",
        "hybrid",
    ],
    "Price & Value": [
        "price",
        "priced",
        "value",
        "cost",
        "money",
        "worth",
        "expensive",
        "cheap",
        "affordable",
        "deal",
        "bang for",
        "overpriced",
        "msrp",
    ],
    "Reliability": [
        "reliable",
        "reliability",
        "unreliable",
        "problem",
        "problems",
        "issue",
        "issues",
        "repair",
        "repairs",
        "breakdown",
        "broke",
        "quality",
        "build",
        "defect",
        "recall",
        "dealer",
        "warranty",
        "maintenance",
    ],
    "Technology": [
        "infotainment",
        "screen",
        "touchscreen",
        "navigation",
        "nav",
        "bluetooth",
        "tech",
        "technology",
        "audio",
        "stereo",
        "speakers",
        "carplay",
        "android auto",
        "camera",
        "sensors",
        "display",
    ],
    "Safety": [
        "safety",
        "safe",
        "airbag",
        "airbags",
        "crash",
        "abs",
        "collision",
        "blind spot",
        "lane",
        "assist",
        "protect",
    ],
}
_PATTERNS = {
    aspect: re.compile("\\b(" + "|".join((re.escape(k) for k in kws)) + ")\\b", re.I)
    for aspect, kws in ASPECTS.items()
}
_CONTRAST = re.compile("\\b(but|however|although|though|whereas|while|yet|except)\\b", re.I)


def _sentences(text: str):
    try:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception:
        return [s for s in re.split("(?<=[.!?])\\s+", text) if s.strip()]


def split_clauses(text: str):
    clauses = []
    for sent in _sentences(text):
        parts = _CONTRAST.split(sent)
        buf = ""
        for p in parts:
            if _CONTRAST.fullmatch(p.strip() or "x"):
                continue
            p = p.strip()
            if p:
                clauses.append(p)
    return [c for c in clauses if len(c) > 1]


def detect_aspects(clause: str):
    return [a for a, pat in _PATTERNS.items() if pat.search(clause)]


def analyze(text: str):
    out = []
    for clause in split_clauses(text):
        aspects = detect_aspects(clause)
        if aspects:
            out.append({"clause": clause, "aspects": aspects})
    return out


if __name__ == "__main__":
    demo = "The engine is incredibly powerful and acceleration is quick, but the fuel economy is terrible and the price is way too high. Seats are comfortable though."
    for item in analyze(demo):
        print(item["aspects"], "|", item["clause"])
