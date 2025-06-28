import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")


conn = sqlite3.connect("medical.db")
cur = conn.cursor()
# cur.execute("SELECT id,content FROM who_brief")
# brief_rows = cur.fetchall()
cur.execute("SELECT id, content FROM who_brief")
brief_rows = cur.fetchall()

cur.execute("SELECT id, content FROM factsheets")
factsheet_rows = cur.fetchall()

all_rows = []

for row in brief_rows:
    all_rows.append((row[0], "who_brief", row[1]))

for row in factsheet_rows:
    all_rows.append((row[0], "factsheet", row[1]))

ids = []
embeddings = []

for id, source, text in all_rows:
    emb = embedder.encode(text)
    ids.append(f"{id}||{source}")
    embeddings.append(emb)

embeddings = np.vstack(embeddings).astype('float32')

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

with open("faiss_ids.txt", "w") as f:
    for id in ids:
        f.write(f"{id}\n")

faiss.write_index(index, "medical_faiss.index")

print(f"✅ FAISS index created for {len(ids)} chunks.")