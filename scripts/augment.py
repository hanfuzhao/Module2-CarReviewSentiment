import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import random
import re
import pandas as pd

_STOPWORDS = set(
    "a an the and or but if then of to in on at for with by is are was were be been this that these those it its i you he she we they my your our their as so very".split()
)


def _wordnet():
    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet.synsets("car")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return wordnet


def _synonyms(word, wn):
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word.lower() and name.isalpha():
                syns.add(name)
    return list(syns)


def synonym_replacement(words, n, wn, rng):
    words = list(words)
    candidates = [w for w in words if w.lower() not in _STOPWORDS and w.isalpha()]
    rng.shuffle(candidates)
    replaced = 0
    for w in candidates:
        syns = _synonyms(w, wn)
        if syns:
            choice = rng.choice(syns)
            words = [choice if x == w else x for x in words]
            replaced += 1
        if replaced >= n:
            break
    return words


def random_deletion(words, p, rng):
    if len(words) == 1:
        return words
    kept = [w for w in words if rng.random() > p]
    return kept or [rng.choice(words)]


def random_swap(words, n, rng):
    words = list(words)
    for _ in range(n):
        if len(words) < 2:
            break
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = (words[j], words[i])
    return words


def eda(sentence, n_aug=2, alpha=0.1, seed=0):
    wn = _wordnet()
    rng = random.Random(seed)
    words = re.findall("\\w+|[^\\w\\s]", sentence)
    n = max(1, int(alpha * len([w for w in words if w.isalpha()])))
    out = []
    ops = [
        lambda: synonym_replacement(words, n, wn, rng),
        lambda: random_swap(words, n, rng),
        lambda: random_deletion(words, alpha, rng),
    ]
    for i in range(n_aug):
        new = ops[i % len(ops)]()
        text = re.sub("\\s+([.,!?;:])", "\\1", " ".join(new)).strip()
        if text and text.lower() != sentence.lower():
            out.append(text)
    return out


def augment_dataframe(df, target_per_class=8000, max_aug_per_row=3, seed=42):
    rng = random.Random(seed)
    pieces = [df]
    for label, grp in df.groupby("label"):
        need = target_per_class - len(grp)
        if need <= 0:
            continue
        rows = grp.to_dict("records")
        added, k = ([], 0)
        while len(added) < need:
            base = rows[k % len(rows)]
            k += 1
            variants = eda(base["text"], n_aug=min(max_aug_per_row, 3), seed=rng.randint(0, 10000))
            for v in variants:
                added.append({**base, "text": v})
                if len(added) >= need:
                    break
        pieces.append(pd.DataFrame(added))
    out = pd.concat(pieces, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def back_translate(texts, pivot="de", seed=42):
    from transformers import MarianMTModel, MarianTokenizer

    def _load(src, tgt):
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        return (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))

    def _translate(batch, tok, mdl):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        gen = mdl.generate(**enc, max_length=256)
        return [tok.decode(g, skip_special_tokens=True) for g in gen]

    tok_f, mdl_f = _load("en", pivot)
    tok_b, mdl_b = _load(pivot, "en")
    mid = _translate(list(texts), tok_f, mdl_f)
    return _translate(mid, tok_b, mdl_b)


if __name__ == "__main__":
    for v in eda("The engine is powerful but the fuel economy is terrible", n_aug=3, seed=1):
        print("-", v)
