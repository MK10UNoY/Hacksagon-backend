# fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client, Client
from retriever import retrieve
from datetime import datetime
import openai

# Supabase config
SUPABASE_URL = "https://hlbztdzuyooltqginqur.supabase.co"
SUPABASE_KEY = "YOUR_SERVICE_ROLE_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# LLM config
openai.api_key = "YOUR_OPENAI_KEY"

# FastAPI init
app = FastAPI()

class QueryRequest(BaseModel):
    user_id: str
    chat_id: str
    user_query: str
    intent: str
    sentiment: str

@app.post("/query")
async def query_handler(req: QueryRequest):
    # Store user query in Supabase
    supabase.table("chat_messages").insert({
        "chat_id": req.chat_id,
        "content": req.user_query,
        "medical_entities": [],
        "metadata": {
            "intent": req.intent,
            "sentiment": req.sentiment
        },
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    # Run trusted local similarity with threshold
    matches = retrieve(req.user_query, top_k=5, min_confidence=0.8)

    if not matches:
        return {"status": "ok", "answer": "No clear WHO match found. Please consult a doctor for unusual or severe symptoms."}

    best = matches[0]
    factsheet = next((m for m in matches if m["condition"] == best["condition"] and m["source"] == "factsheet"), None)

    # Load recent chat for personalization
    chat_history = supabase.table("chat_messages") \
        .select("*") \
        .eq("chat_id", req.chat_id) \
        .order("created_at") \
        .limit(5) \
        .execute().data

    # Final prompt for your LLM (OpenAI or FYI-3)
    prompt = f"""
User Query: {req.user_query}

Intent: {req.intent}
Sentiment: {req.sentiment}

Top WHO match:
Condition: {best['condition']}
Confidence: {best['confidence']}

WHO Brief:
{best['content']}

WHO Factsheet:
{factsheet['content'] if factsheet else ''}

Recent Chat:
{[m['content'] for m in chat_history]}

Based on WHO data only, write a clear, trusted answer.
If severe or emergency, say so. Do not hallucinate.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a trusted WHO assistant. Use only the facts given."},
            {"role": "user", "content": prompt}
        ]
    )

    final_answer = response.choices[0].message.content

    # Store the GPT answer
    supabase.table("chat_messages").insert({
        "chat_id": req.chat_id,
        "content": final_answer,
        "medical_entities": [],
        "metadata": {
            "matched_condition": best["condition"],
            "sources": [m["source"] for m in matches if m["condition"] == best["condition"]]
        },
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return {
        "status": "ok",
        "condition": best["condition"],
        "confidence": best["confidence"],
        "answer": final_answer,
        "sources": [m["source"] for m in matches if m["condition"] == best["condition"]]
    }
