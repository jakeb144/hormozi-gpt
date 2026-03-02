-- ============================================================
-- HORMOZI GPT — Supabase pgvector Setup
-- Run this ONCE in the Supabase SQL Editor (supabase.com → your project → SQL Editor)
-- ============================================================

-- 1. Enable the pgvector extension (gives us the "vector" column type)
create extension if not exists vector with schema extensions;

-- 2. Create the table that stores every training pair + its embedding
create table if not exists hormozi_pairs (
  id            bigserial primary key,
  user_question text      not null,          -- the raw user question
  assistant_answer text   not null,          -- Hormozi's full answer
  embedding     vector(1536) not null,       -- text-embedding-3-small outputs 1536 dims
  created_at    timestamptz default now()
);

-- 3. Create an HNSW index so similarity search is fast even at thousands of rows
create index if not exists hormozi_pairs_embedding_idx
  on hormozi_pairs
  using hnsw (embedding vector_cosine_ops);

-- 4. Similarity search function — called from the backend
--    Takes an embedding + a match count, returns the closest pairs
create or replace function match_hormozi_pairs(
  query_embedding vector(1536),
  match_count     int default 6
)
returns table (
  id               bigint,
  user_question    text,
  assistant_answer text,
  similarity       float
)
language plpgsql
as $$
begin
  return query
    select
      hp.id,
      hp.user_question,
      hp.assistant_answer,
      1 - (hp.embedding <=> query_embedding) as similarity
    from hormozi_pairs hp
    order by hp.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- 5. Quick sanity check — run after uploading data
-- select count(*) from hormozi_pairs;
