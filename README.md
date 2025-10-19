# 🌍 RAG Travel Assistant (RAG + Knowledge Graph + Semantic Cache)

An intelligent **AI travel assistant** that combines **Vector Search (Pinecone)**, **Graph Reasoning (Neo4j)**, and **Semantic Caching (Redis)** to deliver context-aware, fast, and accurate travel recommendations.

---

## 🚀 Overview

This project implements a **hybrid retrieval pipeline** that merges **semantic similarity** from a vector database and **relational knowledge** from a graph database.  
It intelligently caches previous user interactions to **reduce cost and latency**, while ensuring **contextual reasoning** and **dynamic graph traversal** for accurate answers.

---

## 🧠 Key Features

- **Hybrid Retrieval (Vector + Graph):**  
  Retrieves semantically relevant documents from **Pinecone** and relationship-driven facts from **Neo4j**.

- **Semantic Caching Layer (Redis):**  
  Uses a semantic cache to avoid redundant vector or LLM queries for similar user inputs.

- **Asynchronous Execution:**  
  Both retrievals (Pinecone + Neo4j) are executed concurrently for better performance.

- **Context Summarization:**  
  Summarizes the combined vector and graph context before prompting the LLM.

- **Dynamic Prompt Construction:**  
  Builds structured, context-rich prompts for more relevant LLM answers.

---

## ⚙️ System Architecture

```text
                ┌────────────────────────────┐
                │        User Query           │
                └─────────────┬───────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │     Semantic Cache (Redis) │
                │  - Checks if similar query │
                │    exists in cache         │
                └─────────────┬───────────────┘
                              │ (cache miss)
                              ▼
        ┌────────────────────────────┬────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐         ┌──────────────────────────┐
│ Vector DB         │        │ Graph DB         │         │ Asynchronous Execution   │
│ (Pinecone)        │        │ (Neo4j)          │         │ (asyncio + to_thread)    │
│ Retrieves seman-  │        │ Fetches relation │         │ Runs both tasks in       │
│ tic matches       │        │ facts between    │         │ parallel threads         │
│                   │        │ nodes            │         │                          │
└──────────┬────────┘        └──────────┬───────┘         └──────────────────────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
               ┌────────────────────────────┐
               │   Context Summarization     │
               │  Merges vector + graph data │
               │  into a single summary      │
               └─────────────┬───────────────┘
                              ▼
               ┌────────────────────────────┐
               │     LLM Prompt Builder      │
               │  Constructs prompt with:    │
               │  - Semantic summary         │
               │  - Graph context            │
               └─────────────┬───────────────┘
                              ▼
               ┌────────────────────────────┐
               │         LLM (OpenAI)       │
               │  Generates answer based on │
               │  summarized hybrid context │
               └─────────────┬───────────────┘
                              ▼
               ┌────────────────────────────┐
               │   Cache Result in Redis    │
               │  for future retrievals     │
               └─────────────┬───────────────┘
                              ▼
               ┌────────────────────────────┐
               │        Final Answer        │
               └────────────────────────────┘
```

## 📦 Setup & Installation

1️⃣ Clone the Repository
```python
    git clone https://github.com/Md-Tauhid101/RAG-Travel-Assistant.git
    cd RAG-Travel-Assistant
```

## 2️⃣ Create a Virtual Environment
```python
    python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate
```

## 3️⃣ Install Dependencies
```python
    pip install -r requirements.txt
```

## 4️⃣ Environment Variables
- fill the credentials in the config:
```python
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=your_pinecone_env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    REDIS_URL=redis://localhost:6379
```

## 🧠 Usage
### Task 1 – Setup & Data Upload
```python
    python pinecone_upload.py
```

### Task 2 – Create Graph Database 
```python
    python load_to_neo4j.py
```

### Task 2 – Hybrid Chat
```python
    python hybrid_chat.py
```

## Enhancements

✅ Caching: Redis-based semantic cache to skip repetitive queries.

✅ Async Retrieval: Parallel vector + graph fetching via asyncio.gather().

✅ Context Summarization: Summarizes top nodes before prompt generation.

✅ Prompt Improvement: Designed concise, context-grounded LLM prompts.

## 🧠 Key Learnings

- Built and debugged hybrid retrieval between vector & graph databases.
- Implemented async design for parallel querying to minimize latency.
- Designed structured prompt templates for LLM-based itinerary generation.
- Learned failure modes (vector–graph desync, latency spikes, token overflow).

## 🧰 Future Improvements

- Add FastAPI-based web interface
- Integrate LangChain or LlamaIndex for modular retrieval pipeline
- Add feedback-based reranking
- Experiment with local LLMs for cost reduction
