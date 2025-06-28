# fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client, Client
from retriever import retrieve
from datetime import datetime

# --- CONFIG ---
SUPABASE_URL = "https://hlbztdzuyooltqginqur.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FASTAPI INIT ---
app = FastAPI()

class QueryRequest(BaseModel):
    user_id: str
    chat_id: str
    user_query: str
    intent: str
    sentiment: str

@app.post("/query")
async def query_handler(req: QueryRequest):
    # 📝 1️⃣ Store user query in Supabase
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

    # 🔍 2️⃣ Trusted local similarity search
    matches = retrieve(req.user_query, top_k=5, min_confidence=0.8)

    if not matches:
        return {
            "status": "ok",
            "message": "No clear WHO match found. Please consult a doctor."
        }

    best = matches[0]

    # 🧾 3️⃣ Load recent chat context for personalization
    chat_history = supabase.table("chat_messages") \
        .select("*") \
        .eq("chat_id", req.chat_id) \
        .order("created_at") \
        .limit(5) \
        .execute().data

    # 📝 4️⃣ Build final prompt for later FYI-3 integration
    prompt = f"""
User Query: {req.user_query}

Intent: {req.intent}
Sentiment: {req.sentiment}

Top WHO match:
Condition: {best['condition']}
Confidence: {best['confidence']}

WHO Text:
{best['content']}

Recent Chat:
{[m['content'] for m in chat_history]}
"""

    print("📝 Final prompt draft (FYI-3 will use this):")
    print(prompt)

    # 🟢 NO LLM CALL YET — placeholder
    return {
        "status": "ok",
        "matches": matches,
        "context": [m['content'] for m in chat_history],
        "prompt_preview": prompt,
        "note": "LLM integration will plug in here later (FYI-3)."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
