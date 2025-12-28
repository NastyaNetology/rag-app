from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np
import requests
import pandas as pd
import re

# ---------------------------
# Config
# ---------------------------
LLAMA_BASE_URL = "http://127.0.0.1:8080"
LLAMA_MODEL = "qwen2-0_5b.Q4_K_M.gguf"  # must match /v1/models
INDEX_DIR = Path("data/index")

TOP_K = 5  # how many rows to retrieve for context

# ---------------------------
# Add CSV loading once at startup
# ---------------------------

CSV_PATH = Path("data/top_rated_wines.csv")
df = pd.read_csv(CSV_PATH)
# normalize column names just in case
df.columns = [c.strip().lower() for c in df.columns]


# ---------------------------
# Load vector index on startup
# ---------------------------
app = FastAPI(title="Wine RAG Assistant")

with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(INDEX_DIR / "matrix.pkl", "rb") as f:
    matrix = pickle.load(f)

with open(INDEX_DIR / "texts.pkl", "rb") as f:
    texts = pickle.load(f)  # list[str] - each element is a "row text"


# ---------------------------
# Helpers
# ---------------------------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def retrieve(query: str, top_k: int = TOP_K):
    """
    Returns: list of tuples (score, text)
    Uses cosine similarity between query tf-idf vector and document matrix
    """
    q_vec = vectorizer.transform([query])  # shape (1, vocab)
    # cosine sim: (A·B)/(||A||*||B||)
    # matrix is (N, vocab). q_vec.TF-IDF sparse.
    scores = (matrix @ q_vec.T).toarray().ravel()  # shape (N,)
    # highest scores first
    idx = np.argsort(-scores)[:top_k]
    results = [(scores[i], texts[i]) for i in idx]
    return results


def build_context(results):
    """
    Formats retrieved rows into a context block for the LLM.
    """
    lines = []
    for rank, (score, txt) in enumerate(results, start=1):
        lines.append(f"[{rank}] (score={_safe_float(score):.4f}) {txt}")
    return "\n".join(lines)


def call_llama_chat(prompt: str) -> str:
    """
    Calls llama.cpp OpenAI-compatible chat completions endpoint.
    """
    url = f"{LLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def highest_rating_from_csv():
    if "rating" not in df.columns:
        return None
    return float(df["rating"].max())

# ---------------------------
# API schema
# ---------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    context: str
def top_wines_from_csv(n: int = 3):
    needed = {"name", "region", "rating", "notes"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        return None, f"CSV is missing columns: {sorted(missing)}"

    top = df.sort_values("rating", ascending=False).head(n)
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "name": str(r["name"]),
            "region": str(r["region"]),
            "rating": float(r["rating"]),
            "notes": str(r["notes"]),
        })
    return rows, None

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def health():
    return {"status": "Wine RAG app running", "llama_url": LLAMA_BASE_URL, "model": LLAMA_MODEL}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.lower()

    # ---- FACT MODE (CSV truth) ----
    if "highest rating" in q or "max rating" in q:
        max_rating = highest_rating_from_csv()
        return AskResponse(
            question=req.question,
            answer=str(max_rating),
            context="Computed directly from CSV dataset."
        )

    if "top 3" in q and "wine" in q and ("highest" in q or "best" in q):
        rows, err = top_wines_from_csv(3)
        if err:
            return AskResponse(question=req.question, answer=err, context="CSV check failed.")

        answer_lines = []
        for i, w in enumerate(rows, start=1):
            answer_lines.append(
                f"{i}) {w['name']} — {w['region']} — rating {w['rating']}\nNotes: {w['notes']}"
            )

        return AskResponse(
            question=req.question,
            answer="\n\n".join(answer_lines),
            context="Computed directly from CSV dataset."
        )

    # ---- RAG MODE ----
    results = retrieve(req.question, top_k=TOP_K)
    context = build_context(results)

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't have enough information in my data."

CONTEXT:
{context}

QUESTION:
{req.question}

ANSWER (be concise, mention the wine name when possible):
""".strip()

    answer = call_llama_chat(prompt)

    return AskResponse(
        question=req.question,
        answer=answer,
        context=context,
    )

