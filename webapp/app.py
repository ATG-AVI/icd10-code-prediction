"""Chart2Code - web app that suggests ICD-10-CM codes for clinical notes.

    python app.py            # then open http://127.0.0.1:8000

Needs ./model/ from train.py. Academic demo, not for clinical use.
"""
import csv
import io
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
DB_PATH = HERE / "records.db"
TOKEN = re.compile(r"(?u)\b\w\w+\b")  # TfidfVectorizer's default token pattern
TOP_K = 5

if not (MODEL_DIR / "models.joblib").exists():
    raise SystemExit("Model files missing. Run `python train.py` first.")

bundle = joblib.load(MODEL_DIR / "models.joblib")
tfidf, emb_clf, RRF_K = bundle["tfidf"], bundle["embedding_clf"], bundle["rrf_k"]
vectorizer, tfidf_clf = tfidf.named_steps["tfidf"], tfidf.named_steps["clf"]
FEATURES = vectorizer.get_feature_names_out()
CLASSES = tfidf_clf.classes_
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
encoder = SentenceTransformer(bundle["embedding_model"])
train_embeddings = np.load(MODEL_DIR / "train_embeddings.npy")
APP_DATA = json.loads((MODEL_DIR / "app_data.json").read_text())
DIAGNOSIS = {c["code"]: c["diagnosis"] for c in APP_DATA["all_codes"]}
AGREEMENT = APP_DATA["metrics"]["agreement"]


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).replace(" ", " ")).strip()


# ------------------------------------------------------------------ database
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


with db() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            note TEXT NOT NULL,
            model TEXT NOT NULL,
            suggested_code TEXT NOT NULL,
            final_code TEXT NOT NULL,
            final_diagnosis TEXT NOT NULL,
            accepted INTEGER NOT NULL,
            models_agreed INTEGER NOT NULL,
            sample_id TEXT,
            true_code TEXT,
            comment TEXT
        )""")


# ------------------------------------------------------------------ prediction
def ranks(proba):
    return (-proba).argsort().argsort() + 1


def evidence(text, x, code):
    """Per-word contribution of the TF-IDF model towards `code`, as character spans."""
    coef = tfidf_clf.coef_[CLASS_INDEX[code]]
    contrib = {FEATURES[j]: float(v * coef[j]) for j, v in zip(x.indices, x.data)}
    tokens = [(m.start(), m.end(), m.group().lower()) for m in TOKEN.finditer(text)]
    spans = []
    for i, (start, end, word) in enumerate(tokens):
        score = contrib.get(word, 0.0)
        if i > 0:  # bigram contributions are shared between their two words
            score += contrib.get(f"{tokens[i - 1][2]} {word}", 0.0) / 2
        if i + 1 < len(tokens):
            score += contrib.get(f"{word} {tokens[i + 1][2]}", 0.0) / 2
        if abs(score) > 1e-6:
            spans.append([start, end, round(score, 5)])
    top_terms = sorted(((t, s) for t, s in contrib.items() if s > 0), key=lambda t: -t[1])[:8]
    return {"spans": spans, "top_terms": [[t, round(s, 4)] for t, s in top_terms]}


def suggestion(code, score, p_tfidf, p_emb, r_tfidf, r_emb):
    i = CLASS_INDEX[code]
    return {
        "code": code,
        "diagnosis": DIAGNOSIS[code],
        "score": round(float(score), 4),
        "prob_tfidf": round(float(p_tfidf[i]), 4),
        "prob_pubmedbert": round(float(p_emb[i]), 4),
        "rank_tfidf": int(r_tfidf[i]),
        "rank_pubmedbert": int(r_emb[i]),
    }


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


app = FastAPI(title="Chart2Code")


@app.post("/api/predict")
def predict(req: PredictRequest):
    started = time.perf_counter()
    raw = req.text  # evidence offsets must match the exact text in the browser's editor
    if len(TOKEN.findall(raw)) < 5:
        raise HTTPException(422, "Add a few more words of the clinical note (at least 5) to get suggestions.")

    x = vectorizer.transform([raw])  # whitespace does not change TF-IDF tokens
    p_tfidf = tfidf_clf.predict_proba(x)[0]
    query = encoder.encode([clean_text(raw)], normalize_embeddings=True)
    p_emb = emb_clf.predict_proba(query)[0]
    r_tfidf, r_emb = ranks(p_tfidf), ranks(p_emb)
    fused = (1 / (RRF_K + r_tfidf) + 1 / (RRF_K + r_emb)) / (2 / (RRF_K + 1))  # 1.0 = both rank it first
    order_key = fused + 1e-6 * p_emb  # same tie-break as train.py: PubMedBERT confidence

    views = {}
    for name, score, key in [("combined", fused, order_key), ("tfidf", p_tfidf / p_tfidf.max(), p_tfidf),
                             ("pubmedbert", p_emb, p_emb)]:
        top = np.argsort(-key)[:TOP_K]
        views[name] = [suggestion(CLASSES[i], score[i], p_tfidf, p_emb, r_tfidf, r_emb) for i in top]

    shown = {s["code"] for v in views.values() for s in v}
    tfidf_top, emb_top = CLASSES[r_tfidf.argmin()], CLASSES[r_emb.argmin()]
    agree = tfidf_top == emb_top

    similarity = train_embeddings @ query[0]
    nearest = np.argsort(-similarity)[:3]
    similar = [{**APP_DATA["train_cases"][i], "similarity": round(float(similarity[i]), 3)} for i in nearest]

    return {
        "views": views,
        "agreement": {
            "agree": bool(agree),
            "tfidf_top": tfidf_top,
            "pubmedbert_top": emb_top,
            "historical_accuracy": AGREEMENT["accuracy_when_agree" if agree else "accuracy_when_disagree_combined"],
        },
        "evidence": {code: evidence(raw, x, code) for code in shown},
        "similar": similar,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Text from an uploaded .txt or .pdf (images are read in the browser with OCR)."""
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 15 MB.")
    name = (file.filename or "").lower()
    if name.endswith(".pdf") or raw[:4] == b"%PDF":
        from pypdf import PdfReader
        try:
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        except Exception:
            raise HTTPException(422, "This PDF could not be read. Try scanning it as an image instead.")
    else:
        text = raw.decode("utf-8", errors="replace")
        if "\x00" in text or text.count("\ufffd") > 0.01 * len(text):
            raise HTTPException(422, "This file is not plain text or PDF. For a photo of a note, use Scan.")
    if not text.strip():
        raise HTTPException(422, "No text found in this file. If it is a scanned PDF, upload a photo of the page instead.")
    return {"text": clean_text(text)}


# ------------------------------------------------------------------ reference data
@app.get("/api/meta")
def meta():
    return {
        "metrics": APP_DATA["metrics"],
        "supported_codes": APP_DATA["supported_codes"],
        "all_codes": APP_DATA["all_codes"],
        "samples": APP_DATA["samples"],
    }


# ------------------------------------------------------------------ records
class RecordIn(BaseModel):
    source: str = Field(max_length=20)
    note: str = Field(min_length=1, max_length=20000)
    model: str = Field(max_length=20)
    suggested_code: str = Field(max_length=10)
    final_code: str = Field(max_length=10)
    models_agreed: bool
    sample_id: str | None = Field(default=None, max_length=10)
    comment: str | None = Field(default=None, max_length=500)


@app.get("/api/records")
def list_records():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM records ORDER BY id DESC")]


@app.post("/api/records", status_code=201)
def add_record(rec: RecordIn):
    if rec.final_code not in DIAGNOSIS:
        raise HTTPException(422, f"Unknown ICD-10 code: {rec.final_code}")
    sample = next((s for s in APP_DATA["samples"] if s["id"] == rec.sample_id), None)
    with db() as con:
        cur = con.execute(
            """INSERT INTO records (created_at, source, note, model, suggested_code, final_code, final_diagnosis,
                                    accepted, models_agreed, sample_id, true_code, comment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), rec.source, clean_text(rec.note), rec.model,
             rec.suggested_code, rec.final_code, DIAGNOSIS[rec.final_code], int(rec.final_code == rec.suggested_code),
             int(rec.models_agreed), rec.sample_id, sample["code"] if sample else None, rec.comment),
        )
        return dict(con.execute("SELECT * FROM records WHERE id = ?", (cur.lastrowid,)).fetchone())


@app.delete("/api/records/{record_id}", status_code=204)
def delete_record(record_id: int):
    with db() as con:
        con.execute("DELETE FROM records WHERE id = ?", (record_id,))


@app.get("/api/records.csv")
def export_records():
    rows = list_records()
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()) if rows else ["id"])
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=coded_records.csv"})


# ------------------------------------------------------------------ static site
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/figures", StaticFiles(directory=HERE.parent / "figures"), name="figures")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
