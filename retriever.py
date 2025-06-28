# retriever.py

import faiss
import sqlite3
import json
from sentence_transformers import SentenceTransformer

import numpy as np

# Load your embedding model
embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# Load FAISS index and ID mapping
index = faiss.read_index("medical_faiss.index")
with open("faiss_ids.txt") as f:
    id_list = [line.strip() for line in f]

# SQLite WHO DB
conn = sqlite3.connect("medical.db")

def retrieve(query: str, top_k: int = 5, min_confidence: float = 0.8):
    # 1️⃣ Embed the query
    vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue  # skip invalid hits
        doc_id = id_list[idx]

        row = conn.execute(
            "SELECT id, condition, source, content FROM facts WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if not row:
            continue

        # 2️⃣ Convert FAISS distance to cosine similarity
        similarity = 1 - dist  # because you likely indexed with cosine distance
        if similarity < min_confidence:
            continue  # skip weak matches

        results.append({
            "id": row[0],
            "condition": row[1],
            "source": row[2],
            "content": row[3],
            "confidence": round(similarity, 4)
        })

    return results
