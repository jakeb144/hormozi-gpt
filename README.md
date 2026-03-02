# 🔥 Hormozi GPT — Deploy Guide

A RAG-powered Alex Hormozi business mentor. Built with Supabase (pgvector), OpenAI embeddings, Claude Sonnet 4, and a FastAPI backend deployed on Vercel.

**What it does:** You ask a business question → the app finds the 6 most similar Q&A pairs from 3,647 Hormozi training examples → Claude answers in Hormozi's voice with that context.

---

## Step 1 · Get Your 4 API Keys

You need accounts at three services. All have free tiers.

### 1a. OpenAI (for embeddings)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Click **API Keys** in the left sidebar
4. Click **Create new secret key** → copy it
5. Save it somewhere — you'll need it twice (upload + Vercel)

### 1b. Anthropic (for Claude)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Go to **API Keys** → **Create Key** → copy it

### 1c. Supabase (for the vector database)
1. Go to [supabase.com](https://supabase.com) → **Start your project**
2. Create a new project (pick any name, set a database password, choose a region close to you)
3. Wait ~2 minutes for it to spin up
4. Go to **Settings** → **API** in the left sidebar
5. Copy two things:
   - **Project URL** (looks like `https://abcdefg.supabase.co`)
   - **service_role key** (the secret one, NOT the anon key)

---

## Step 2 · Set Up the Database

1. In your Supabase project, click **SQL Editor** in the left sidebar
2. Click **New query**
3. Open the `supabase_setup.sql` file from this project
4. Copy-paste the ENTIRE contents into the SQL editor
5. Click **Run** (the green play button)
6. You should see "Success. No rows returned" — that's correct

---

## Step 3 · Upload Your Training Data

This step runs on your computer. You need Python 3.9+ installed.

```bash
# 1. Open a terminal and cd into this project folder
cd hormozi-gpt

# 2. Install dependencies
pip install openai supabase python-dotenv

# 3. Create your .env file
cp .env.example .env

# 4. Open .env in any text editor and paste in your 4 keys:
#    OPENAI_API_KEY=sk-...
#    ANTHROPIC_API_KEY=sk-ant-...
#    SUPABASE_URL=https://your-project.supabase.co
#    SUPABASE_SERVICE_KEY=eyJ...

# 5. Make sure training_pairs.jsonl is in this folder

# 6. Run the upload script
python upload_pairs.py
```

This will:
- Parse all 3,647 Q&A pairs
- Embed each user question with OpenAI's text-embedding-3-small
- Upload everything to your Supabase `hormozi_pairs` table

It takes about 3–5 minutes. You'll see progress updates.

**Verify it worked:** Go to Supabase → SQL Editor → run: `select count(*) from hormozi_pairs;`
You should see `3647`.

---

## Step 4 · Push to GitHub

```bash
# 1. Create a new repo on github.com (DON'T add a README — we have one)

# 2. In your terminal, inside the hormozi-gpt folder:
git init
git add .
git commit -m "Hormozi GPT initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hormozi-gpt.git
git push -u origin main
```

**Important:** Make sure your `.env` file is NOT committed. Create a `.gitignore`:

```
.env
__pycache__/
*.pyc
```

---

## Step 5 · Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) → Sign up with GitHub
2. Click **Add New** → **Project**
3. Find and import your `hormozi-gpt` repo
4. **Before clicking Deploy**, expand **Environment Variables**
5. Add these 4 variables (copy-paste from your .env):

| Name | Value |
|------|-------|
| `OPENAI_API_KEY` | `sk-...` |
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `SUPABASE_URL` | `https://your-project.supabase.co` |
| `SUPABASE_SERVICE_KEY` | `eyJ...` |

6. Click **Deploy**
7. Wait 1–2 minutes. Vercel will give you a URL like `hormozi-gpt.vercel.app`
8. Open it. Type a business question. Watch Hormozi respond in real time. 🔥

---

## Project Structure

```
hormozi-gpt/
├── api/
│   └── chat.py              ← FastAPI backend (Claude + RAG)
├── public/
│   └── index.html            ← Chat UI (dark theme, streaming)
├── supabase_setup.sql        ← Run once in Supabase SQL Editor
├── upload_pairs.py           ← Run once to embed + upload data
├── vercel.json               ← Routes /api/* to Python, rest to static
├── requirements.txt          ← Python dependencies
├── .env.example              ← Template for your 4 keys
└── README.md                 ← You are here
```

---

## How It Works (Under the Hood)

1. **You type a question** in the chat UI
2. **Frontend** sends your full conversation history to `/api/chat`
3. **Backend** takes your last 3 user messages, combines them into one retrieval query
4. **OpenAI** embeds that query into a 1536-dim vector
5. **Supabase** runs cosine similarity search → returns the 6 closest training pairs
6. **Claude Sonnet 4** gets a detailed Hormozi system prompt + those 6 pairs as context + your full conversation
7. **Response streams back** word-by-word as Server-Sent Events
8. **UI renders** each token in real time with a blinking cursor

---

## Troubleshooting

**"Connection error" in the chat:**
- Check Vercel logs: go to your project → **Deployments** → latest → **Functions** tab
- Most common: wrong environment variable names or values

**Upload script fails:**
- Make sure `training_pairs.jsonl` is in the same folder as `upload_pairs.py`
- Check that your `SUPABASE_SERVICE_KEY` is the service_role key, not the anon key

**Empty responses from Claude:**
- Check your Anthropic API key has credits
- Verify the model name in `api/chat.py` matches an available model

**Supabase function not found:**
- Re-run the `supabase_setup.sql` script in the SQL editor
- Make sure you ran ALL of it, including the `create or replace function` block

---

## Optional: Switch to Claude Opus

If you want deeper, more nuanced responses (recommended for serious business mentoring), change this line in `api/chat.py`:

```python
CLAUDE_MODEL = "claude-opus-4-0-20250514"
```

Opus is slower and costs more per token, but gives noticeably richer strategic advice.
