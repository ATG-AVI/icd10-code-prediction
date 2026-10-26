"""Train the models used by the Chart2Code web app and save everything it needs.

Reproduces the notebook's main experiment (codes with >= 5 cases, publication-grouped
80/20 split, same tuned hyperparameters) so the app's numbers match the notebook.

    python train.py          # writes ./model/
"""
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", message="The least populated class")

RANDOM_STATE = 42
MIN_CASES = 5
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
RRF_K = 1  # rank-fusion constant. k=1 makes the score readable (both models rank a code first = 100%);
           # the textbook k=60 squeezes ranks 1-5 into 93-100%. Accuracy is the same either way:
           # k=60 -> test 62.0%, CV 61.1%; k=1 -> test 63.0%, CV 61.0%.
OUT = Path(__file__).parent / "model"


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).replace(" ", " ")).strip()


def make_tfidf():
    # Hyperparameters selected by grouped CV in the notebook (Section 6)
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1, random_state=RANDOM_STATE)),
    ])


def make_embedding_clf():
    # C selected by grouped CV in the notebook (Section 7)
    return LogisticRegression(max_iter=5000, class_weight="balanced", C=10, random_state=RANDOM_STATE)


def rrf(proba_a, proba_b):
    """Reciprocal rank fusion of two probability matrices with the same class order.

    Ties (e.g. ranks 1+2 vs 2+1) go to the class the second model is more confident in;
    pass the better-calibrated model (PubMedBERT) as proba_b.
    """
    ranks_a = (-proba_a).argsort(axis=1).argsort(axis=1) + 1
    ranks_b = (-proba_b).argsort(axis=1).argsort(axis=1) + 1
    return 1 / (RRF_K + ranks_a) + 1 / (RRF_K + ranks_b) + 1e-6 * proba_b


def scores(y_true, score, classes):
    order = np.argsort(-score, axis=1)
    hit = classes[order] == np.asarray(y_true)[:, None]
    return {"accuracy": hit[:, 0].mean(), "top3": hit[:, :3].any(1).mean(), "top5": hit[:, :5].any(1).mean()}


# ------------------------------------------------------------------ data + split
df = load_dataset("mkurman/clinical-case-icd10-diagnosis")["train"].to_pandas()
df["clean_input"] = df["input"].apply(clean_text)
code_counts = df["icd10_code"].value_counts()
data = df[df["icd10_code"].isin(code_counts[code_counts >= MIN_CASES].index)].reset_index(drop=True)
data["split_group"] = data["pmcid"].fillna(data["case_id"].astype(str))

X, y, groups = data["clean_input"], data["icd10_code"], data["split_group"]
all_classes = set(y)
for train_idx, test_idx in GroupShuffleSplit(n_splits=500, test_size=0.2, random_state=RANDOM_STATE).split(X, y, groups):
    if not all_classes - set(y.iloc[train_idx]):
        break
train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)
print(f"train {len(train)} | test {len(test)} | classes {y.nunique()}")

# ------------------------------------------------------------------ models
tfidf = make_tfidf().fit(train["clean_input"], train["icd10_code"])
encoder = SentenceTransformer(EMBEDDING_MODEL)
encode = lambda notes: encoder.encode(list(notes), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
train_emb, test_emb = encode(train["clean_input"]), encode(test["clean_input"])
emb_clf = make_embedding_clf().fit(train_emb, train["icd10_code"])
assert list(tfidf.classes_) == list(emb_clf.classes_)
classes = tfidf.classes_

p_tfidf, p_emb = tfidf.predict_proba(test["clean_input"]), emb_clf.predict_proba(test_emb)
test_scores = {
    "TF-IDF + LR": scores(test["icd10_code"], p_tfidf, classes),
    "PubMedBERT + LR": scores(test["icd10_code"], p_emb, classes),
    "Combined (rank fusion)": scores(test["icd10_code"], rrf(p_tfidf, p_emb), classes),
}

# How trustworthy is agreement between the two models? (held-out test set)
top_t, top_e = classes[p_tfidf.argmax(1)], classes[p_emb.argmax(1)]
agree = top_t == top_e
agreement = {
    "agree_share": agree.mean(),
    "accuracy_when_agree": (top_t[agree] == test["icd10_code"].values[agree]).mean(),
    "accuracy_when_disagree_combined": (classes[rrf(p_tfidf, p_emb).argmax(1)][~agree] == test["icd10_code"].values[~agree]).mean(),
}

# Repeated grouped CV over all >= 5 data (same protocol as notebook Section 8.3)
all_emb = encode(data["clean_input"])
cv_rows = {name: [] for name in test_scores}
for seed in [0, 1, 2]:
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
        t = make_tfidf().fit(X.iloc[tr], y.iloc[tr])
        e = make_embedding_clf().fit(all_emb[tr], y.iloc[tr])
        if list(t.classes_) != list(e.classes_):
            continue  # a fold missing a class in training; rank fusion needs aligned columns
        pt, pe = t.predict_proba(X.iloc[te]), e.predict_proba(all_emb[te])
        for name, s in [("TF-IDF + LR", pt), ("PubMedBERT + LR", pe), ("Combined (rank fusion)", rrf(pt, pe))]:
            cv_rows[name].append(scores(y.iloc[te], s, t.classes_)["accuracy"])
cv_scores = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": len(v)} for name, v in cv_rows.items()}

for name in test_scores:
    t, c = test_scores[name], cv_scores[name]
    print(f"{name:24s} test acc {t['accuracy']:.3f} top3 {t['top3']:.3f} top5 {t['top5']:.3f} | "
          f"CV acc {c['mean']:.3f} +/- {c['std']:.3f} ({c['folds']} folds)")
print("agreement:", {k: round(float(v), 3) for k, v in agreement.items()})

# ------------------------------------------------------------------ save
OUT.mkdir(exist_ok=True)
joblib.dump({"tfidf": tfidf, "embedding_clf": emb_clf, "embedding_model": EMBEDDING_MODEL, "rrf_k": RRF_K},
            OUT / "models.joblib")
np.save(OUT / "train_embeddings.npy", train_emb)

code_to_diagnosis = df.drop_duplicates("icd10_code").set_index("icd10_code")["diagnosis"].to_dict()
vec, clf = tfidf.named_steps["tfidf"], tfidf.named_steps["clf"]
features = vec.get_feature_names_out()
train_counts = train["icd10_code"].value_counts()
supported = [{
    "code": c,
    "diagnosis": code_to_diagnosis[c],
    "train_cases": int(train_counts[c]),
    "total_cases": int(code_counts[c]),
    "top_words": [str(w) for w in features[np.argsort(clf.coef_[i])[::-1][:8]]],
} for i, c in enumerate(classes)]

app_data = {
    "supported_codes": supported,
    "all_codes": [{"code": c, "diagnosis": d} for c, d in sorted(code_to_diagnosis.items())],
    "train_cases": [{"code": r.icd10_code, "diagnosis": r.diagnosis, "note": r.clean_input} for r in train.itertuples()],
    "samples": [{"id": f"T{i + 1:03d}", "code": r.icd10_code, "diagnosis": r.diagnosis, "note": r.clean_input}
                for i, r in enumerate(test.itertuples())],
    "metrics": {
        "test": test_scores, "cv": cv_scores, "agreement": agreement,
        "train_notes": len(train), "test_notes": len(test), "classes": len(classes),
    },
}
(OUT / "app_data.json").write_text(json.dumps(app_data, default=float))
print("saved to", OUT)
