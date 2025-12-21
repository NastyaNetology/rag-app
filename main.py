from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import requests
from pathlib import Path
import numpy as np

app = FastAPI()

# Load index on startup
INDEX_DIR = Path("data/index")

with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(INDEX_DIR / "texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open(INDEX_DIR / "matrix.pkl", "rb") as f:
    matrix = pickle.load(f)

class AskRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "RAG app running"}

def retrieve_context(question: str, k: int = 3) -> str:
    q_vec = vectorizer.transform([question])
    scores = (matrix @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(scores)[-k:][::-1]
    return "\n\n".join(texts[i] for i in top_idx)

@app.post("/ask")
def ask(req: AskRequest):
    context = retrieve_context(req.question)

    prompt = f"""
You are an assistant answering questions using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{req.question}
"""

    payload = {
        "model": "qwen2-0_5b.Q4_K_M.gguf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 256,
    }

    r = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    return {
        "answer": data["choices"][0]["message"]["content"],
        "context_used": context,
    }


