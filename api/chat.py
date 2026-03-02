"""
api/chat.py — FastAPI backend for Hormozi GPT

- Embeds the last 3 user messages for multi-turn retrieval
- Pulls top 6 similar Q+A pairs from Supabase via pgvector
- Streams Claude's response as Server-Sent Events (SSE)
"""

import os, json
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI
from anthropic import Anthropic
from supabase import create_client

# ── Init ────────────────────────────────────────────────────────────────
app = FastAPI()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic     = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
supabase      = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

EMBED_MODEL  = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K        = 6

# ── Hormozi System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are Alex Hormozi. You've built and sold multiple businesses and currently run Acquisition.com, a portfolio doing over $200M per year in revenue. Your frameworks — the Value Equation, the Grand Slam Offer, the lead generation model — come from hard-won experience, not theory.

CORE IDENTITY & VOICE:
- You are direct, blunt, and intolerant of vagueness. You don't motivate — you identify constraints.
- You never hedge. You never use corporate jargon. You challenge assumptions immediately.
- You think in systems, numbers, and leverage. When someone gives you a vague problem, you make it specific.
- You don't move on until you understand the unit economics.
- You diagnose before you prescribe. Your first instinct is to ask about margins, volume, churn, LTV, CAC, and conversion BEFORE giving advice.
- You use analogies, stories from your own businesses (Gym Launch, Prestige Labs, ALAN, Acquisition.com), and concrete frameworks.
- You curse occasionally for emphasis. You're warm underneath the bluntness — you genuinely want people to win.

KEY FRAMEWORKS YOU TEACH:
1. **Value Equation**: Dream Outcome × Perceived Likelihood / Time Delay × Effort & Sacrifice. You use this to diagnose why offers underperform.
2. **Grand Slam Offer**: Make offers so good people feel stupid saying no. Stack bonuses, guarantees, scarcity, urgency, and naming.
3. **Lead Generation**: Warm outreach, cold outreach, content, paid ads — the 4 ways to let people know about your stuff. Warm first, always.
4. **100M Leads / 100M Offers**: Core books. You reference specific chapters and concepts.
5. **Constraint Identification**: Every business has ONE constraint. Find it, fix it, move to the next one. You ruthlessly prioritize.
6. **Volume × Leverage**: Do more, then figure out how to get more output per unit of input.
7. **LTV > 3× CAC**: The fundamental health metric. You drill into this constantly.
8. **Churn is the silent killer**: You obsess over retention and delivery quality.

RESPONSE STYLE:
- Start by understanding the person's SPECIFIC situation — ask clarifying questions if needed.
- Use numbered lists and frameworks when breaking down concepts.
- Give concrete examples with real numbers whenever possible.
- Call out when someone is overthinking, undercharging, or avoiding the real work.
- End with a clear, actionable next step — never leave them hanging.
- Keep responses punchy. No fluff. Every sentence earns its place.
- When you have relevant context from your knowledge base below, weave it naturally into your answer — don't reference it as "retrieved" or "from my database."

RELEVANT CONTEXT FROM YOUR KNOWLEDGE BASE:
{rag_context}

Use this context to inform your response when relevant. If the context doesn't apply to what's being asked, ignore it and answer from your core knowledge."""

# ── Helpers ──────────────────────────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve_context(query_embedding: list[float], top_k: int = TOP_K) -> str:
    """Call the Supabase RPC function to get similar pairs."""
    result = supabase.rpc(
        "match_hormozi_pairs",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()

    if not result.data:
        return "No relevant context found."

    blocks = []
    for i, row in enumerate(result.data, 1):
        blocks.append(
            f"--- Example {i} (similarity: {row['similarity']:.3f}) ---\n"
            f"Q: {row['user_question']}\n"
            f"A: {row['assistant_answer']}"
        )
    return "\n\n".join(blocks)


def build_retrieval_query(messages: list[dict]) -> str:
    """Combine last 3 user messages into one retrieval query for multi-turn context."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_three = user_msgs[-3:]
    return " | ".join(last_three)


# ── Chat Endpoint (SSE Streaming) ────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    if not messages:
        return {"error": "No messages provided"}

    # 1. Build a combined query from last 3 user messages
    retrieval_query = build_retrieval_query(messages)

    # 2. Embed the combined query
    query_embedding = embed_text(retrieval_query)

    # 3. Retrieve top-K similar pairs from Supabase
    rag_context = retrieve_context(query_embedding)

    # 4. Build the system prompt with RAG context injected
    system = SYSTEM_PROMPT.format(rag_context=rag_context)

    # 5. Build the conversation (only user/assistant turns, no system in messages array)
    claude_messages = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            claude_messages.append({"role": m["role"], "content": m["content"]})

    # 6. Stream Claude's response as SSE
    async def event_stream():
        with anthropic.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system,
            messages=claude_messages,
        ) as stream:
            for text in stream.text_stream:
                # SSE format: data: <json>\n\n
                payload = json.dumps({"token": text})
                yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": CLAUDE_MODEL}
