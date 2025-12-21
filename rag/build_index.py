import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

DATA_PATH = Path("data/top_rated_wines.csv")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def row_to_text(row: pd.Series) -> str:
    # Adjust column names if your CSV differs
    parts = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return " | ".join(parts)

df = pd.read_csv(DATA_PATH)
texts = [row_to_text(r) for _, r in df.iterrows()]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(INDEX_DIR / "texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open(INDEX_DIR / "matrix.pkl", "wb") as f:
    pickle.dump(X, f)

print(f"Indexed {len(texts)} rows. Saved to {INDEX_DIR}/")
