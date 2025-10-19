import json
import time
import asyncio
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import redis

import config
from chat_cache import SemanticCache

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )
index = pc.Index(INDEX_NAME)

# Redis
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    username=config.REDIS_USERNAME,
    password=config.REDIS_PASSWORD,
    decode_responses=True
)

# Removed flushall — keep cache between runs
# redis_client.flushall()

# Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    t1 = time.time()
    print(f"DEBUG: Pinecone top {top_k} results: {len(res['matches'])}")
    return res["matches"]


def fetch_graph_context_by_keyword(query_text: str):
    """
    Fetch relevant graph facts using full-text search in Neo4j.
    """
    facts = []
    with driver.session() as session:
        q = """
        CALL db.index.fulltext.queryNodes("entityFullTextIndex", $query) YIELD node, score
        MATCH (node)-[r]-(m:Entity)
        RETURN node.id AS source, type(r) AS rel, 
               m.id AS target_id, m.name AS target_name,
               m.type AS target_type, m.description AS target_desc
        LIMIT 30
        """
        recs = session.run(q, parameters={"query": query_text})
        for r in recs:
            facts.append({
                "source": r["source"],
                "rel": r["rel"],
                "target_id": r["target_id"],
                "target_name": r["target_name"],
                "target_desc": (r["target_desc"] or "")[:400],
                "target_type": r["target_type"]
            })
    t1 = time.time()
    print(f"DEBUG: Graph facts: {len(facts)}")
    return facts


def context_summary(pinecone_matches, graph_facts):
    """ummarize retrieved context before passing to LLM."""
    texts = []
    for m in pinecone_matches:
        meta = m.get("metadata", {})
        if "text" in meta:
            texts.append(meta["text"][:300])

    facts = [f"{f['source']} -[{f['rel']}]-> {f['target_name']}" for f in graph_facts]
    combined = "\n".join(texts + facts)
    return combined[:2000]  # truncate to avoid LLM prompt bloat


def build_prompt(user_query, pinecone_matches, graph_facts, context_summary):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful, knowledgeable travel assistant. "
        "Use both semantic summaries and graph relationships to craft accurate, practical, "
        "and concise travel recommendations. Mention node ids when referring to destinations. "
        "Think step-by-step internally but only include your final reasoning in the response."
    )

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts[:20]
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"User query: {user_query}\n\n"
            f"Semantic context (background facts):\n{context_summary}\n\n"
            f"Graph facts (relationships and connections):\n" + "\n".join(graph_context) + "\n\n"
            "Use the semantic context for factual information and the graph facts for relational insights. "
            "Respond in this format:\n"
            "1. **Overview** – Brief context summary (1–2 sentences)\n"
            "2. **Suggested Itinerary** – 3–4 short bullet points\n"
            "3. **References** – Mention node ids used.\n"
        }
    ]
    return prompt



def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content

# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():
    print("Hybrid travel assistant (Old Version - Timed). Type 'exit' to quit.")
    semantic_cache = SemanticCache(redis_client, embed_text)

    while True:
        query = input("\nEnter your travel question or 'exit': ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        # 1. Cache check
        cached_result = semantic_cache.get_from_cache(query)
        if cached_result:
            print(cached_result)
            continue

        # 2. Run Pinecone & Graph in parallel using asyncio.to_thread
        matches, graph_facts = await asyncio.gather(
            asyncio.to_thread(pinecone_query, query, TOP_K),
            asyncio.to_thread(fetch_graph_context_by_keyword, query)
        )

        # 3. Build context + prompt
        summary = context_summary(matches, graph_facts)
        prompt = build_prompt(query, matches, graph_facts, summary)

        # 4. LLM answer
        answer = await asyncio.to_thread(call_chat, prompt)

        # 5. Cache the result
        semantic_cache.add_to_cache(query, {"matches": matches, "graph_facts": graph_facts}, answer)

        # 6. Show result + timing breakdown
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    asyncio.run(interactive_chat())
    driver.close()
