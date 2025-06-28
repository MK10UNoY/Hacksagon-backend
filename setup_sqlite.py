# sqlite_setup.py

import os
import json
import sqlite3

# --- DB ---
conn = sqlite3.connect("medical.db")
cur = conn.cursor()

# --- Create WHO brief table ---
cur.execute("""
CREATE TABLE IF NOT EXISTS who_brief (
    id TEXT PRIMARY KEY,
    condition TEXT,
    content TEXT
)
""")

# --- Create WHO factsheets table ---
cur.execute("""
CREATE TABLE IF NOT EXISTS factsheets (
    id TEXT PRIMARY KEY,
    condition TEXT,
    content TEXT
)
""")

# --- WHO brief ---
with open("who_cleaned_combined_.json", encoding="utf-8") as f:
    brief_data = json.load(f)

for disease_id, sections in brief_data.items():
    brief_text = ""
    for key in ["symptoms", "causes", "treatment", "when_to_see_doctor", "prevention"]:
        if key in sections and sections[key]:
            joined = " ".join(sections[key]) if isinstance(sections[key], list) else str(sections[key])
            brief_text += f"{key.replace('_', ' ').title()}: {joined}\n"

    cur.execute("""
    INSERT OR REPLACE INTO who_brief (id, condition, content)
    VALUES (?, ?, ?)
    """, (disease_id, disease_id, brief_text))

print("✅ WHO brief indexed in SQLite!")

# --- WHO factsheets ---
factsheets_dir = "./who_diseases/"

for file in os.listdir(factsheets_dir):
    if file.endswith(".json"):
        with open(os.path.join(factsheets_dir, file), encoding="utf-8") as f:
            fs_data = json.load(f)

        disease_id = file.replace(".json", "")
        factsheet_text = ""

        for key in ["Key facts", "Overview", "Symptoms and patterns", "Diagnosis and treatment", "Self-care"]:
            if key in fs_data and fs_data[key]:
                joined = " ".join(fs_data[key]) if isinstance(fs_data[key], list) else str(fs_data[key])
                factsheet_text += f"{key}: {joined}\n"

        cur.execute("""
        INSERT OR REPLACE INTO factsheets (id, condition, content)
        VALUES (?, ?, ?)
        """, (disease_id, disease_id, factsheet_text))

print("✅ WHO factsheets indexed in SQLite!")

conn.commit()
conn.close()
