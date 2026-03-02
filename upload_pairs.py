"""
upload_pairs.py — Embeds user questions and uploads Q+A pairs to Supabase.

Usage:
  1. pip install openai supabase python-dotenv
  2. Fill in your .env (see .env.example)
  3. python upload_pairs.py

Embeds ONLY the user question (not the full Q+A) so retrieval matches on
what the user is *asking*, not what Hormozi already said.
"""

import json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_SERVICE_KEY"]  # use the SERVICE key for uploads
JSONL_PATH       = Path("training_pairs.jsonl")
EMBED_MODEL      = "text-embedding-3-small"             # 1536 dims, cheap & fast
BATCH_SIZE       = 100                                   # rows per Supabase insert
EMBED_BATCH      = 200                                   # texts per OpenAI embed call

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 1. Parse the JSONL ──────────────────────────────────────────────────
def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                messages = entry["messages"]
                user_msg = next(m["content"] for m in messages if m["role"] == "user")
                asst_msg = next(m["content"] for m in messages if m["role"] == "assistant")
                pairs.append({"user_question": user_msg, "assistant_answer": asst_msg})
            except Exception as e:
                print(f"  ⚠ Skipping line {i}: {e}")
    return pairs

# ── 2. Batch-embed the user questions ──────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        print(f"  Embedding {start}–{start+len(batch)} of {len(texts)}…")
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
        time.sleep(0.2)  # gentle rate-limit buffer
    return all_embeddings

# ── 3. Upload to Supabase in batches ───────────────────────────────────
def upload_to_supabase(pairs: list[dict], embeddings: list[list[float]]):
    rows = []
    for pair, emb in zip(pairs, embeddings):
        rows.append({
            "user_question":    pair["user_question"],
            "assistant_answer": pair["assistant_answer"],
            "embedding":        emb,
        })
    
    uploaded = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        supabase.table("hormozi_pairs").insert(batch).execute()
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{len(rows)} rows")
    
    return uploaded

# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not JSONL_PATH.exists():
        print(f"❌ File not found: {JSONL_PATH}")
        print(f"   Place your training_pairs.jsonl in the same folder as this script.")
        sys.exit(1)

    print(f"📂 Loading pairs from {JSONL_PATH}…")
    pairs = load_pairs(JSONL_PATH)
    print(f"   Found {len(pairs)} valid pairs.\n")

    print("🧠 Embedding user questions (text-embedding-3-small)…")
    questions = [p["user_question"] for p in pairs]
    embeddings = embed_texts(questions)
    print(f"   Embedded {len(embeddings)} questions.\n")

    print("☁️  Uploading to Supabase…")
    count = upload_to_supabase(pairs, embeddings)
    print(f"\n✅ Done! {count} pairs uploaded to hormozi_pairs table.")
    print("   Run this in Supabase SQL Editor to verify:")
    print("   select count(*) from hormozi_pairs;")
