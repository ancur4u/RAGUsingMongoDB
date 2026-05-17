# RAGUsingMongoDB
# 🍃 MongoDB Atlas RAG Platform

> **Production-grade Retrieval-Augmented Generation (RAG) using MongoDB Atlas Vector Search**
> Built by [Ankur Parashar](https://github.com/ankurparashar) — Senior Pre-Sales Solution Architect

[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-00684A?style=flat&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)
[![Voyage AI](https://img.shields.io/badge/Voyage%20AI-Native%20Embeddings-4A148C?style=flat)](https://www.mongodb.com/products/platform/ai-search-and-retrieval)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat)](https://ollama.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)

---

## 📋 Overview

Two production-ready RAG applications demonstrating MongoDB Atlas as a unified AI data platform:

| App | Embedding | Generation | Use Case |
|-----|-----------|------------|----------|
| **App 1** — Voyage AI RAG | `voyage-3` (MongoDB native, 1024-dim) | `llama3` (local) | Production-grade, state-of-art |

Both apps share the same architecture — only the embedding layer differs. This demonstrates the **MongoDB unified data platform** story:

```
PDF Upload
    ↓ Extract text (PyMuPDF)
    ↓ Chunk (~250 words, 50-word overlap)
    ↓ Embed (Voyage AI or nomic-embed-text)
    ↓ Store in MongoDB Atlas (content + embedding + metadata)

User Query
    ↓ Embed query (same model)
    ↓ $vectorSearch (HNSW, cosine similarity)
    ↓ Filter chunks with score ≥ 0.7
    ↓ Build RAG prompt with top chunks
    ↓ llama3 (local Ollama) → streamed answer
```

---

## 🏦 BFSI Use Case Mapping

This demo maps directly to a **production BFSI SEC Filing RAG platform**:

| Demo Component | Production BFSI Equivalent |
|---|---|
| MongoDB Q4 earnings PDF | SEC EDGAR 10-K / 8-K filings |
| nomic-embed-text / voyage-3 | Voyage AI voyage-3-large (2048-dim) |
| llama3 (local) | Claude / GPT-4o via API |
| Single PDF ingestion | Kafka → Atlas streaming pipeline |
| Manual upload | Automated CDC from document store |
| Free tier rate limits | Enterprise Atlas tier — no limits |

> *"An analyst types: 'What are Goldman Sachs key risk factors in Q4 2024?'
> $vectorSearch retrieves the 5 most semantically similar chunks from 50 SEC filings.
> Claude generates a structured risk summary in < 800ms.
> Zero ETL. Zero separate vector DB. One MongoDB Atlas platform."*

---

## 📁 Project Structure

```
mongodb_rag_platform/
│
├── app1_ollama/               # App 1 — Local Ollama embeddings
│   ├── rag_app.py             # Main Streamlit application
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment template
│
├── app2_voyage/               # App 2 — MongoDB Voyage AI embeddings
│   ├── voyage_rag.py          # Main Streamlit application
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment template
│
├── docs/
│   ├── ARCHITECTURE.md        # Detailed architecture docs
│   ├── SETUP.md               # Step-by-step setup guide
│   ├── DEMO_SCRIPT.md         # SA Challenge demo script
│   └── TROUBLESHOOTING.md     # Common issues and fixes
│
├── assets/
│   └── architecture.png       # Architecture diagram
│
├── .gitignore                 # Excludes .env, venv, __pycache__
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [MongoDB Atlas](https://www.mongodb.com/atlas) account (free tier works)
- [Ollama](https://ollama.ai) installed locally
- Git

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/mongodb_rag_platform.git
cd mongodb_rag_platform
```

### Step 2 — Choose your app

**App 1 (Ollama — fully free, offline):**
```bash
cd app1_ollama
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your MongoDB URI
streamlit run rag_app.py
```

**App 2 (Voyage AI — MongoDB native embeddings):**
```bash
cd app2_voyage
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your MongoDB URI + Voyage API key
streamlit run voyage_rag.py
```

### Step 3 — Pull Ollama models

```bash
ollama pull llama3
ollama pull nomic-embed-text   # App 1 only
ollama pull llava              # Optional — multimodal
```

### Step 4 — Upload a PDF and ask questions

Open `http://localhost:8501` → **Upload & Ingest PDF** tab → upload any PDF → ask questions.

---

## 🔧 Environment Variables

### App 1 (`.env`)

```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=multimodal_rag
COLLECTION_NAME=documents
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=llama3
VISION_MODEL=llava
```

### App 2 (`.env`)

```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=voyage_rag
COLLECTION_NAME=voyage_documents
VOYAGE_API_KEY=your_mongodb_atlas_model_api_key
OLLAMA_BASE_URL=http://localhost:11434
CHAT_MODEL=llama3
```

> **Get Voyage API key:** MongoDB Atlas UI → top nav → **AI Models** → Create model API key

---

## 📊 MongoDB Atlas Vector Index

Create this index in Atlas → Search & Vector Search → Vector Search tab:

**App 1 index** (collection: `documents`, index name: `vector_index`):
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "source_type" },
    { "type": "filter", "path": "file_name" }
  ]
}
```

**App 2 index** (collection: `voyage_documents`, index name: `voyage_vector_index`):
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "source_type" },
    { "type": "filter", "path": "file_name" }
  ]
}
```

---

## 🎯 App Features

Both apps include **5 tabs:**

| Tab | Description |
|-----|-------------|
| 💬 **Ask a Question** | Full RAG pipeline with streaming answer |
| 📤 **Upload & Ingest PDF** | Multi-file upload with progress tracking |
| 🔍 **Vector Search Explorer** | Raw $vectorSearch results without LLM |
| 📐 **Architecture** | Side-by-side architecture comparison |
| 📊 **Collection Stats** | Index status, chunk counts, document preview |

**Key capabilities:**
- ✅ Multi-file PDF upload (batch ingestion)
- ✅ Automatic duplicate detection via MD5 hash
- ✅ Score-filtered LLM context (chunks ≥ 0.7 only)
- ✅ 3 response modes: Q&A, Summary, BFSI Analyst
- ✅ Live token streaming from llama3
- ✅ $vectorSearch pipeline viewer
- ✅ Per-document delete capability
- ✅ Retry logic with exponential backoff on rate limits

---

## 🧠 Document Schema

```json
{
  "_id": "ObjectId",
  "file_name": "MongoDB_Q4_2025.pdf",
  "file_hash": "7a6bb1fc...",
  "source_type": "pdf",
  "chunk_index": 4,
  "total_chunks": 24,
  "content": "MongoDB Atlas Revenue up 24% Year-over-Year...",
  "embedding": [-0.029, 0.071, ...],
  "ingested_at": "2026-05-17T10:30:00Z",
  "metadata": {
    "char_count": 1240,
    "word_count": 250,
    "embed_model": "voyage-3",
    "embed_dims": 1024,
    "embed_api": "MongoDB Atlas Embedding API (Voyage AI)"
  }
}
```

---

## 📈 Performance Benchmarks

Tested on MacBook Pro M2 with free tier Atlas cluster:

| Operation | App 1 (Ollama) | App 2 (Voyage AI) |
|-----------|----------------|-------------------|
| Embed query | ~0.5s (local) | ~1-8s (API, rate limited) |
| $vectorSearch | ~0.3s | ~0.3s |
| llama3 generation | ~3-8s | ~3-8s |
| **Total per query** | **~4-10s** | **~5-15s** |
| Ingestion per chunk | ~0.5s | ~21s (free tier) |

> **Note:** Voyage AI free tier is limited to 3 RPM. Add payment method to Atlas for standard rate limits and much faster ingestion.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ankur Parashar**
Senior Pre-Sales Solution Architect | 17+ Years Enterprise Experience
Confluent · Oracle · IBM · Optum/UHG · Zimbra

- MongoDB Atlas RAG portfolio project
- Demonstrates: Vector Search, Voyage AI native embeddings, Atlas Stream Processing
- Stack: MongoDB Atlas · Voyage AI · Ollama · Streamlit · Python

---

## 🔗 Resources

- [MongoDB Atlas Vector Search Docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Voyage AI by MongoDB](https://www.mongodb.com/docs/voyageai/)
- [Atlas Embedding API](https://www.mongodb.com/docs/voyageai/api-reference/overview/)
- [Ollama](https://ollama.ai)
- [Streamlit Docs](https://docs.streamlit.io)
