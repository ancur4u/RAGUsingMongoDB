"""
MongoDB Atlas RAG Platform — Native Voyage AI Embeddings
MongoDB Atlas Embedding API (voyage-3) + Atlas Vector Search + Local Ollama (llama3)
Author: Ankur Parashar — SA Challenge Demo
"""

import os, json, time, requests, certifi, hashlib, math
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI    = os.getenv("MONGODB_URI")
DB_NAME        = os.getenv("DB_NAME", "voyage_rag")
COLLECTION     = os.getenv("COLLECTION_NAME", "voyage_documents")
VOYAGE_KEY     = os.getenv("VOYAGE_API_KEY")
VOYAGE_API     = "https://ai.mongodb.com/v1/embeddings"
EMBED_MODEL    = "voyage-3"
EMBED_DIMS     = 1024
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "llama3")
INDEX_NAME     = "voyage_vector_index"
CHUNK_SIZE     = 250
CHUNK_OVERLAP  = 50

# ── MongoDB ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_col():
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    return client[DB_NAME][COLLECTION]

# ── Voyage AI Embedding (MongoDB Native) ──────────────────────────────────────
def voyage_embed(texts: list, input_type: str = "document") -> list:
    resp = requests.post(
        VOYAGE_API,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VOYAGE_KEY}"
        },
        json={"input": texts, "model": EMBED_MODEL, "input_type": input_type},
        timeout=60
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

def voyage_embed_query(text: str) -> list:
    """Embed a single query string with retry on 429."""
    max_retries = 6
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                VOYAGE_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VOYAGE_KEY}"
                },
                json={"input": [text], "model": EMBED_MODEL,
                      "input_type": "query"},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    return data[0]["embedding"]
                time.sleep(21)
                continue
            if resp.status_code == 429:
                wait = 21 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Query embedding failed: {e}")
            time.sleep(21)
    raise RuntimeError("Query embedding failed after all retries")

@st.cache_data(ttl=3600)
def cached_embed_query(text: str) -> list:
    """Cache query embeddings for 1 hour."""
    return voyage_embed_query(text)

def truncate_to_tokens(text: str, max_words: int = 300) -> str:
    """Truncate text to max_words to stay within Voyage API token limit."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def voyage_embed_single(text: str, max_retries: int = 8) -> list:
    """Embed a single chunk — mirrors the standalone script that worked."""
    safe_text = truncate_to_tokens(text, max_words=250)

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                VOYAGE_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VOYAGE_KEY}"
                },
                json={
                    "input":      [safe_text],
                    "model":      EMBED_MODEL,
                    "input_type": "document"
                },
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    return data[0]["embedding"]
                # Empty data — wait and retry
                time.sleep(21)
                continue

            if resp.status_code == 429:
                # Rate limit — wait full minute before retry
                wait = 60 if attempt == 0 else 21 * (attempt + 1)
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                # Token limit — truncate and retry immediately
                safe_text = truncate_to_tokens(safe_text, max_words=100)
                time.sleep(2)
                continue

            # Any other error — wait and retry
            time.sleep(21)
            continue

        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(21)

    raise RuntimeError("Failed to embed chunk after all retries")

def voyage_embed_documents(texts: list) -> list:
    """
    Embed each chunk individually with retry.
    One-by-one is slower but guarantees no mismatch —
    every chunk gets exactly one embedding.
    """
    all_embeddings = []
    for i, text in enumerate(texts):
        embedding = voyage_embed_single(text)
        all_embeddings.append(embedding)
        # Polite pause every chunk to respect rate limits
        time.sleep(0.8)
    return all_embeddings

# ── Ollama ────────────────────────────────────────────────────────────────────
def ollama_generate(prompt: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": True},
        timeout=180, stream=True
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if not chunk.get("done"):
                yield chunk.get("response", "")

def ollama_health() -> dict:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── PDF Ingestion ─────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        pass
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        pass
    raise ImportError("Run: pip install pymupdf")

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks

def ingest_pdf(pdf_bytes: bytes, file_name: str,
               progress_bar, status_text) -> dict:
    col       = get_col()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if col.count_documents({"file_hash": file_hash}) > 0:
        return {"status": "duplicate"}

    status_text.text("📄 Extracting text from PDF...")
    text = extract_pdf_text(pdf_bytes)
    if not text.strip():
        return {"status": "error",
                "message": "No text extracted — PDF may be scanned/image-only. "
                           "Try: pip install pymupdf4llm or use a text-based PDF"}

    status_text.text("✂️ Chunking text...")
    chunks = chunk_text(text)
    total  = len(chunks)

    try:
        embeddings = []
        for ci, chunk in enumerate(chunks):
            status_text.text(
                f"🧠 Embedding chunk {ci+1}/{total} via "
                f"MongoDB Atlas Voyage AI ({EMBED_MODEL})..."
            )
            progress_bar.progress(0.2 + 0.5 * (ci / total))
            emb = voyage_embed_single(chunk)
            embeddings.append(emb)
            # Free tier: 3 RPM — wait 20s between chunks
            if ci < total - 1:
                for remaining in range(20, 0, -1):
                    status_text.text(
                        f"⏳ Rate limit pause: {remaining}s "
                        f"(chunk {ci+1}/{total} done) — "
                        f"Free tier: 3 req/min"
                    )
                    time.sleep(1)
    except Exception as e:
        return {"status": "error", "message": f"Embedding failed: {e}"}

    # Retry once if embedding count mismatches (usually a rate limit issue)
    if len(embeddings) != len(chunks) or not embeddings:
        status_text.text(f"⚠️ Embedding mismatch — retrying after 10s...")
        import time as _time
        _time.sleep(10)
        try:
            embeddings = voyage_embed_documents(chunks)
        except Exception as e:
            return {"status": "error", "message": f"Embedding retry failed: {e}"}

    if len(embeddings) != len(chunks) or not embeddings:
        return {"status": "error",
                "message": f"Embedding mismatch after retry: "
                           f"{len(chunks)} chunks, {len(embeddings)} embeddings. "
                           f"Try ingesting this file alone."}

    progress_bar.progress(0.7)
    status_text.text(f"💾 Saving {total} chunks to MongoDB Atlas...")

    docs = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        docs.append({
            "file_name":    file_name,
            "file_hash":    file_hash,
            "source_type":  "pdf",
            "chunk_index":  i,
            "total_chunks": total,
            "content":      chunk,
            "embedding":    embedding,
            "ingested_at":  datetime.utcnow(),
            "metadata": {
                "char_count":  len(chunk),
                "word_count":  len(chunk.split()),
                "embed_model": EMBED_MODEL,
                "embed_dims":  EMBED_DIMS,
                "embed_api":   "MongoDB Atlas Embedding API (Voyage AI)",
            }
        })

    if not docs:
        return {"status": "error",
                "message": "No chunks were embedded — PDF may be empty or image-only"}

    col.insert_many(docs)
    progress_bar.progress(1.0)

    return {
        "status":      "success",
        "chunks":      total,
        "words":       len(text.split()),
        "embed_model": EMBED_MODEL,
        "embed_dims":  EMBED_DIMS,
    }

# ── Vector Search ─────────────────────────────────────────────────────────────
def vector_search(vec: list, k: int = 5,
                  file_filter: str = "All files") -> list:
    filter_clause = ({"filter": {"file_name": file_filter}}
                     if file_filter != "All files" else {})
    pipeline = [
        {"$vectorSearch": {
            "index":         INDEX_NAME,
            "path":          "embedding",
            "queryVector":   vec,
            "numCandidates": k * 15,
            "limit":         k,
            **filter_clause
        }},
        {"$project": {
            "_id": 0, "content": 1, "file_name": 1,
            "chunk_index": 1, "total_chunks": 1, "metadata": 1,
            "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    return list(get_col().aggregate(pipeline))

# ── RAG Prompt ────────────────────────────────────────────────────────────────
def build_prompt(query: str, chunks: list, mode: str) -> str:
    context = "\n\n---\n\n".join([
        f"[Source {i+1} | {c.get('file_name','?')} | "
        f"Chunk {c.get('chunk_index',0)+1}/{c.get('total_chunks','?')} | "
        f"Score: {c['score']:.4f}]\n{c['content']}"
        for i, c in enumerate(chunks)
    ])
    if mode == "qa":
        return f"""You are an expert research assistant with deep knowledge
across AI, technology, and business domains.

Use the retrieved context below as your PRIMARY source. If the context
contains relevant information, use it. If the context is partial,
combine it with your knowledge to give a complete, useful answer.
Only say you cannot answer if the question is completely unrelated
to the documents.

RETRIEVED CONTEXT:
{context}

QUESTION: {query}

Provide a clear, helpful answer:
- Lead with the direct answer
- Reference specific chunks where relevant (e.g. "According to Source 1...")
- Add broader context from your knowledge where it adds value
- Keep it concise and practical"""
    elif mode == "summary":
        return f"""You are an expert document analyst.
Summarise the following retrieved document chunks into a structured overview.

DOCUMENT CHUNKS:
{context}

Provide a well-structured summary:
1. **Main Topic** — what is this document about?
2. **Key Concepts** — 5 most important ideas or terms
3. **Technical Details** — specific methods, numbers, or processes mentioned
4. **Practical Applications** — how can this be used?
5. **One-line Summary** — single sentence capturing the essence

Be specific and reference actual content from the chunks."""
    else:  # bfsi
        return f"""You are a senior BFSI research analyst at a top investment bank.
A client has asked the following question. Use the retrieved document context
as your primary source, supplemented by your financial expertise.

CLIENT QUESTION: {query}

RETRIEVED CONTEXT FROM RESEARCH DATABASE:
{context}

Provide a professional, structured analysis suitable for a client memo:

**Direct Answer**
Address the client question clearly and concisely.

**Evidence from Documents**
Reference specific findings from the retrieved chunks.

**Risk Considerations**
Key risks or caveats the client should be aware of.

**Regulatory Angle**
Any compliance, regulatory, or governance implications.

**Recommendation**
Actionable next steps or strategic advice.

**Confidence Level**: High / Medium / Low
Justify based on quality and coverage of retrieved context."""


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MongoDB RAG — Voyage AI Native",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.answer-box {
  background: #f0faf5; border: 2px solid #00684A;
  padding: 1.3rem; border-radius: 10px; line-height: 1.75;
}
.voyage-badge {
  background: #4A148C; color: white; padding: 0.2rem 0.6rem;
  border-radius: 12px; font-size: 0.78rem; font-weight: bold;
}
.atlas-badge {
  background: #00684A; color: white; padding: 0.2rem 0.6rem;
  border-radius: 12px; font-size: 0.78rem; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#00684A 0%,#00ED64 60%,#4A148C 100%);
     padding:1.3rem 2rem;border-radius:12px;margin-bottom:1rem;'>
  <h2 style='color:white;margin:0;'>
    🍃 MongoDB Atlas RAG — Native Voyage AI Embeddings
  </h2>
  <p style='color:#d0f5e8;margin:0.3rem 0 0 0;'>
    <span class='voyage-badge'>voyage-3 · 1024-dim</span>
    &nbsp;MongoDB Atlas Embedding API&nbsp;·&nbsp;
    $vectorSearch&nbsp;·&nbsp;llama3 (local)
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Status")

    try:
        test_vec = voyage_embed_query("test")
        st.markdown("**MongoDB Voyage API:** ✅ Connected")
        st.caption(f"Model: `{EMBED_MODEL}` · {EMBED_DIMS} dims")
    except Exception as e:
        st.markdown("**MongoDB Voyage API:** ❌ Error")
        st.caption(str(e))

    health = ollama_health()
    if health["ok"]:
        st.markdown("**Ollama:** ✅ Running")
        for m in health["models"]:
            st.markdown(f"&nbsp;&nbsp;`{m}`")
    else:
        st.markdown("**Ollama:** ❌ Not running")
        st.caption("Run: `ollama serve`")

    st.markdown("---")
    try:
        col   = get_col()
        total = col.count_documents({})
        files = col.distinct("file_name")
        st.markdown("**MongoDB Atlas:** ✅ Connected")
        st.metric("Total chunks", total)
        st.metric("Documents",    len(files))
        st.caption(f"`{DB_NAME}.{COLLECTION}`")
    except Exception as e:
        st.markdown("**MongoDB:** ❌ Error")
        st.caption(str(e))
        files = []

    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    k    = st.slider("Chunks to retrieve", 2, 10, 5)
    mode = st.selectbox("Response mode",
                        ["qa", "summary", "bfsi"],
                        format_func=lambda x: {
                            "qa":      "📖 Q&A",
                            "summary": "📋 Summarise",
                            "bfsi":    "🏦 BFSI Analyst"
                        }[x])
    file_filter = "All files"
    if files:
        file_filter = st.selectbox("Filter by document",
                                   ["All files"] + sorted(files))

    st.markdown("---")
    st.markdown("### 🏗️ Stack")
    st.markdown("""
- 🍃 MongoDB Atlas Vector Search
- 🧠 **MongoDB Voyage AI** (native)
- &nbsp;&nbsp;`voyage-3` · 1024-dim · cosine
- 🦙 llama3 (local generation)
- 📄 HNSW index
    """)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask a Question",
    "📤 Upload & Ingest PDF",
    "🔍 Vector Search Explorer",
    "📐 Architecture",
    "📊 Collection Stats",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask anything about your ingested documents")
    st.caption(
        "Query embedded via **MongoDB Atlas Voyage AI API** · "
        "Retrieved via **$vectorSearch** · "
        "Answer generated by **llama3** (local)"
    )

    suggestions = [
        "What is the attention mechanism?",
        "How does multi-head attention work?",
        "What is positional encoding?",
        "Compare self-attention with RNNs",
        "What are the key contributions of the Transformer?",
        "How does the encoder-decoder architecture work?",
    ]
    st.markdown("**💡 Suggested questions:**")
    s_cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if s_cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state["vq"] = s
            st.rerun()

    st.markdown("---")

    query = st.text_area(
        "Your question:",
        value=st.session_state.get("vq", ""),
        height=80,
        placeholder="Ask anything about your documents..."
    )

    c1, c2 = st.columns([5, 1])
    run = c1.button("🚀 Run RAG Pipeline", type="primary",
                    use_container_width=True, disabled=not query.strip())
    if c2.button("🗑️ Clear", use_container_width=True):
        st.session_state["vq"] = ""
        st.rerun()

    if run and query.strip():
        st.session_state["vq"] = query

        with st.spinner("🧠 Embedding query via MongoDB Atlas Voyage AI API..."):
            t0 = time.time()
            try:
                qvec  = cached_embed_query(query)
                t_emb = time.time() - t0
            except Exception as e:
                st.error(
                    f"❌ Voyage API error: {e}\n\n"
                    f"Free tier limit (3 RPM) may be active. "
                    f"Wait 60 seconds and try again."
                )
                st.stop()

        if not qvec:
            st.error("❌ Empty embedding returned. Wait 60s and retry.")
            st.stop()

        st.info(
            f"✅ **MongoDB Voyage AI** embedded query — "
            f"{len(qvec)} dims · `{EMBED_MODEL}` · {t_emb:.2f}s"
        )

        with st.spinner("🔍 $vectorSearch on MongoDB Atlas..."):
            t1      = time.time()
            results = vector_search(qvec, k, file_filter)
            t_srch  = time.time() - t1
        st.info(f"✅ Retrieved {len(results)} chunks · {t_srch:.3f}s")

        if not results:
            st.error(
                "No results. Create `voyage_vector_index` in Atlas "
                "Search & Vector Search tab first."
            )
            st.stop()

        # Filter to high-confidence chunks (score > 0.7)
        # Fall back to all results if none meet the threshold
        HIGH_SCORE_THRESHOLD = 0.7
        high_score_results = [r for r in results if r["score"] >= HIGH_SCORE_THRESHOLD]
        llm_chunks = high_score_results if high_score_results else results

        if high_score_results:
            st.success(
                f"✅ Using **{len(high_score_results)} high-confidence chunks** "
                f"(score ≥ {HIGH_SCORE_THRESHOLD}) for LLM generation. "
                f"{len(results) - len(high_score_results)} lower-score chunk(s) excluded."
            )
        else:
            st.warning(
                f"⚠️ No chunks above {HIGH_SCORE_THRESHOLD} threshold — "
                f"using all {len(results)} retrieved chunks."
            )

        st.markdown("### 📄 Retrieved Context Chunks")
        for i, r in enumerate(results):
            score = r["score"]
            color = ("#00684A" if score > 0.7
                     else "#FFB300" if score > 0.5 else "#E53935")
            icon  = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
            with st.expander(
                f"{icon} Chunk {i+1} | {r.get('file_name','?')} | "
                f"Part {r.get('chunk_index',0)+1}/{r.get('total_chunks','?')} | "
                f"Score: {score:.4f}",
                expanded=(i == 0)
            ):
                st.markdown(
                    f"<span style='color:{color};font-weight:bold;'>"
                    f"Similarity: {score:.4f}</span>",
                    unsafe_allow_html=True
                )
                st.write(r["content"])
                meta = r.get("metadata", {})
                if meta:
                    st.caption(
                        f"Embed model: `{meta.get('embed_model','?')}` · "
                        f"Dims: {meta.get('embed_dims','?')} · "
                        f"Words: {meta.get('word_count','?')}"
                    )

        st.markdown("### 🤖 AI Answer")
        st.caption(
            f"Generation: `{CHAT_MODEL}` (local Ollama) · "
            f"Mode: `{mode}` · Context: {len(results)} chunks"
        )

        prompt = build_prompt(query, llm_chunks, mode)
        ph     = st.empty()
        ans    = ""
        t2     = time.time()

        try:
            for token in ollama_generate(prompt):
                ans += token
                ph.markdown(
                    f"<div class='answer-box'>{ans}</div>",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Generation error: {e}")
            st.stop()

        t_gen = time.time() - t2

        st.markdown("---")
        mc = st.columns(5)
        mc[0].metric("Voyage Embed",  f"{t_emb:.2f}s")
        mc[1].metric("Vector Search", f"{t_srch:.3f}s")
        mc[2].metric("Generate",      f"{t_gen:.1f}s")
        mc[3].metric("Total",         f"{t_emb+t_srch+t_gen:.1f}s")
        mc[4].metric("Chunks used",
                     f"{len(llm_chunks)}/{len(results)}")

        with st.expander("🔧 $vectorSearch Pipeline Used"):
            flt = (f',\n      "filter": {{"file_name": "{file_filter}"}}'
                   if file_filter != "All files" else "")
            st.code(f"""
// Query embedded via MongoDB Atlas Embedding API (Voyage AI)
// POST https://ai.mongodb.com/v1/embeddings
// model: voyage-3 | input_type: query | dims: 1024

db.voyage_documents.aggregate([
  {{
    "$vectorSearch": {{
      "index": "{INDEX_NAME}",
      "path": "embedding",
      "queryVector": [/* 1024-dim voyage-3 vector */],
      "numCandidates": {k * 15},
      "limit": {k}{flt}
    }}
  }},
  {{
    "$project": {{
      "content": 1, "file_name": 1, "chunk_index": 1,
      "score": {{"$meta": "vectorSearchScore"}}
    }}
  }}
])
            """, language="javascript")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD & INGEST PDF (MULTI-FILE)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📤 Upload PDFs — Ingest with MongoDB Native Voyage AI")
    st.markdown("""
Upload one or more PDFs. Embedding is handled by **MongoDB Atlas Embedding API**
powered by Voyage AI — no separate embedding service needed.

**Pipeline:**
1. Extract text from each PDF
2. Chunk into ~500-word segments
3. **Batch embed via MongoDB Atlas API** (`voyage-3`, 1024-dim)
4. Store chunks + embeddings in Atlas — immediately searchable
    """)

    # PDF library check
    pdf_lib = None
    try:
        import fitz
        pdf_lib = "PyMuPDF"
    except ImportError:
        try:
            import pdfplumber
            pdf_lib = "pdfplumber"
        except ImportError:
            pass

    if pdf_lib:
        st.success(f"✅ PDF library: `{pdf_lib}`")
    else:
        st.error("❌ Run: `pip install pymupdf`")
        st.stop()

    # Vector index check
    try:
        indexes   = list(get_col().list_search_indexes())
        idx_names = [ix.get("name") for ix in indexes]
        if INDEX_NAME in idx_names:
            st.success(f"✅ Vector index `{INDEX_NAME}` is Active")
        else:
            st.warning(
                f"⚠️ Vector index `{INDEX_NAME}` not found. "
                f"Create it in Atlas after ingesting documents."
            )
            with st.expander("📋 Index definition to create in Atlas UI"):
                st.code(f"""
{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": {EMBED_DIMS},
      "similarity": "cosine"
    }},
    {{"type": "filter", "path": "source_type"}},
    {{"type": "filter", "path": "file_name"}}
  ]
}}
                """, language="json")
                st.markdown("""
**Steps:**
1. Atlas → Search & Vector Search → **Vector Search** tab
2. Click **Create Search Index**
3. Database: `voyage_rag` · Collection: `voyage_documents`
4. Index name: `voyage_vector_index`
5. Paste JSON above → Next → Create
                """)
    except Exception:
        pass

    st.markdown("---")

    with st.expander("⚙️ Ingestion Settings"):
        ic1, ic2 = st.columns(2)
        c_size = ic1.number_input("Chunk size (words)", 100, 1000, CHUNK_SIZE)
        c_ovl  = ic2.number_input("Overlap (words)", 0, 200, CHUNK_OVERLAP)
        st.caption(
            "Embedding is batched (4 chunks/batch) with automatic "
            "retry on rate limits."
        )

    # ── MULTI-FILE UPLOADER ───────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Hold Cmd (Mac) or Ctrl (Windows) to select multiple files"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")

        # Preview table
        col = get_col()
        preview_data = []
        for f in uploaded_files:
            raw       = f.read()
            file_hash = hashlib.md5(raw).hexdigest()
            existing  = col.count_documents({"file_hash": file_hash})
            f.seek(0)
            preview_data.append({
                "file":      f,
                "bytes":     raw,
                "file_hash": file_hash,
                "existing":  existing,
                "size_kb":   f"{len(raw)/1024:.1f} KB",
                "status":    "⚠️ Already ingested" if existing else "✅ New",
            })

        # Show preview
        for p in preview_data:
            pc = st.columns([4, 1, 2])
            pc[0].markdown(f"📄 `{p['file'].name}`")
            pc[1].markdown(p["size_kb"])
            pc[2].markdown(p["status"])

        new_files = [p for p in preview_data if not p["existing"]]
        skip_files = [p for p in preview_data if p["existing"]]

        if skip_files:
            st.info(
                f"ℹ️ {len(skip_files)} file(s) already ingested — "
                f"will be skipped. {len(new_files)} new file(s) to ingest."
            )

        st.markdown("---")

        # Delete existing option
        if skip_files:
            if st.button("🗑️ Delete all existing & re-ingest everything",
                         type="secondary"):
                for p in skip_files:
                    col.delete_many({"file_hash": p["file_hash"]})
                st.success(
                    f"Deleted {len(skip_files)} existing file(s). "
                    f"Click Ingest to re-process."
                )
                st.rerun()

        if not new_files and not skip_files:
            st.info("No files to process.")
        else:
            btn_label = (
                f"🚀 Ingest {len(new_files)} New File(s) with Voyage AI"
                if new_files else "✅ All files already ingested"
            )
            ingest_btn = st.button(
                btn_label,
                type="primary",
                use_container_width=True,
                disabled=(len(new_files) == 0)
            )

            if ingest_btn and new_files:
                overall_success = 0
                overall_chunks  = 0

                for idx, p in enumerate(new_files):
                    # Cooldown between files to avoid rate limiting
                    if idx > 0:
                        time.sleep(5)
                    st.markdown(
                        f"### [{idx+1}/{len(new_files)}] `{p['file'].name}`"
                    )
                    prog   = st.progress(0)
                    status = st.empty()

                    result = ingest_pdf(
                        p["bytes"], p["file"].name, prog, status
                    )

                    if result["status"] == "success":
                        status.empty()
                        st.success(
                            f"✅ `{p['file'].name}` — "
                            f"{result['chunks']} chunks · "
                            f"{result['words']:,} words · "
                            f"`{result['embed_model']}` · "
                            f"{result['embed_dims']} dims"
                        )
                        overall_success += 1
                        overall_chunks  += result["chunks"]
                    elif result["status"] == "duplicate":
                        status.empty()
                        st.warning(f"⚠️ `{p['file'].name}` already ingested — skipped")
                    else:
                        status.empty()
                        st.error(
                            f"❌ `{p['file'].name}` failed: "
                            f"{result.get('message','Unknown error')}"
                        )

                # Summary
                if overall_success > 0:
                    st.balloons()
                    st.markdown("---")
                    sm = st.columns(3)
                    sm[0].metric("Files ingested", overall_success)
                    sm[1].metric("Total chunks",   overall_chunks)
                    sm[2].metric("Embed model",    EMBED_MODEL)
                    st.info(
                        "💡 Go to **Ask a Question** tab to query "
                        "all your documents now!"
                    )

    st.markdown("---")
    st.markdown("### 🗂️ Ingested Documents")
    try:
        col   = get_col()
        files = col.distinct("file_name")
        if files:
            for fname in sorted(files):
                n = col.count_documents({"file_name": fname})
                w = sum(
                    d.get("metadata", {}).get("word_count", 0)
                    for d in col.find({"file_name": fname}, {"metadata": 1})
                )
                fc = st.columns([4, 1, 1, 1])
                fc[0].markdown(f"📄 `{fname}`")
                fc[1].metric("Chunks", n)
                fc[2].metric("~Words", f"{w:,}")
                if fc[3].button("🗑️", key=f"del_{fname}",
                                help=f"Delete {fname}"):
                    col.delete_many({"file_name": fname})
                    st.success(f"Deleted `{fname}`")
                    st.rerun()
                st.divider()
        else:
            st.info("No documents yet. Upload PDFs above.")
    except Exception as e:
        st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VECTOR SEARCH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Raw Vector Search — No LLM")
    st.caption(
        "Query embedded via MongoDB Atlas Voyage AI API · "
        "Results from $vectorSearch"
    )

    raw_q = st.text_input("Search query:",
                           placeholder="e.g. transformer attention heads")
    n_raw = st.slider("Results", 1, 20, 8, key="raw_n")

    if st.button("Search", type="primary", key="raw_btn"):
        if not raw_q.strip():
            st.warning("Enter a query.")
        else:
            with st.spinner("Embedding via MongoDB Voyage AI + searching Atlas..."):
                t0      = time.time()
                qvec    = voyage_embed_query(raw_q)
                raw_res = vector_search(qvec, n_raw, file_filter)
                elapsed = time.time() - t0

            st.success(
                f"{len(raw_res)} results · {elapsed:.3f}s · "
                f"{len(qvec)}-dim {EMBED_MODEL} vector"
            )

            for i, r in enumerate(raw_res):
                score = r["score"]
                bar   = int(score * 100)
                color = ("#00684A" if score > 0.7
                         else "#FFB300" if score > 0.5 else "#E53935")
                st.markdown(f"""
<div style='background:white;border:1px solid #ddd;border-radius:8px;
     padding:1rem;margin-bottom:0.7rem;'>
  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
    <strong>#{i+1} — {r.get("file_name","?")}</strong>
    <span style='color:{color};font-weight:bold;'>{score:.4f}</span>
  </div>
  <div style='background:#eee;border-radius:4px;height:6px;margin-bottom:8px;'>
    <div style='background:{color};width:{bar}%;height:6px;border-radius:4px;'></div>
  </div>
  <div style='font-size:0.87rem;color:#333;line-height:1.5;'>
    {r["content"][:400]}{"..." if len(r["content"])>400 else ""}
  </div>
  <div style='font-size:0.76rem;color:#888;margin-top:6px;'>
    Chunk {r.get("chunk_index",0)+1} of {r.get("total_chunks","?")}
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📐 Architecture — Native MongoDB Voyage AI vs External")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### 🔵 Previous App (nomic-embed-text)")
        st.code("""PDF
  ↓ PyMuPDF extract
  ↓ chunk (500 words)
  ↓ Ollama nomic-embed-text
     → 768-dim (LOCAL)
  ↓ MongoDB Atlas insert

Query
  ↓ Ollama nomic-embed-text
     → 768-dim (LOCAL)
  ↓ $vectorSearch
  ↓ llama3 → answer

Embedding: External local model
Limitation: Model must be
running locally always""", language="text")

    with a2:
        st.markdown("#### 🟢 This App (MongoDB Native Voyage AI)")
        st.code("""PDF
  ↓ PyMuPDF extract
  ↓ chunk (500 words)
  ↓ MongoDB Atlas Embedding API
     voyage-3 → 1024-dim (CLOUD)
  ↓ MongoDB Atlas insert

Query
  ↓ MongoDB Atlas Embedding API
     voyage-3 → 1024-dim (CLOUD)
  ↓ $vectorSearch
  ↓ llama3 → answer

Embedding: MongoDB native API
Advantage: Managed by MongoDB,
no local embedding infra needed""", language="text")

    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")

    rows = [
        ["Provider",          "Local Ollama",               "MongoDB Atlas (Voyage AI)"],
        ["Model",             "nomic-embed-text",           "voyage-3"],
        ["Dimensions",        "768",                        "1024"],
        ["Similarity",        "cosine",                     "cosine"],
        ["Runs where",        "Your Mac (CPU/GPU)",         "MongoDB Cloud (managed)"],
        ["Retrieval quality", "Good (open source)",         "State-of-art (MTEB leader)"],
        ["Cost",              "Free (local compute)",       "Usage-based (Atlas API)"],
        ["Requires Ollama",   "✅ Yes",                     "❌ No"],
        ["Production ready",  "⚠️ Depends on infra",       "✅ Fully managed"],
        ["BFSI demo story",   "External embedding service", "MongoDB unified platform"],
    ]

    hc = st.columns(3)
    for i, h in enumerate(rows[0]):
        hc[i].markdown(f"**{h}**")
    st.divider()

    for row in rows[1:]:
        rc = st.columns(3)
        for i, cell in enumerate(row):
            bg = "#f0faf5" if i == 2 else "#f8f9fa"
            rc[i].markdown(
                f"<div style='background:{bg};padding:0.4rem;"
                f"border-radius:4px;font-size:0.9rem;'>{cell}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("#### 🔑 Why This Matters for the SA Demo")
    st.markdown("""
> *"In the previous version, we used nomic-embed-text running locally —
> good for development but requires local GPU infrastructure in production.
> In this version, embedding is handled natively by MongoDB's own
> Voyage AI API — the same state-of-the-art model that powers Atlas
> Vector Search at enterprise scale. Your application doesn't need to
> manage any embedding infrastructure. MongoDB handles it.
> One platform. One API key. No synchronization tax."*
    """)

    st.markdown("---")
    st.markdown("#### 🗂️ Index + Document Schema")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.code(f"""{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": {EMBED_DIMS},
      "similarity": "cosine"
    }},
    {{"type":"filter","path":"source_type"}},
    {{"type":"filter","path":"file_name"}}
  ]
}}""", language="json")
    with sc2:
        st.code("""{
  "file_name":   "paper.pdf",
  "file_hash":   "abc123...",
  "chunk_index": 4,
  "content":     "The attention mechanism...",
  "embedding":   [0.029, -0.072, ...],  // 1024 floats
  "metadata": {
    "embed_model": "voyage-3",
    "embed_dims":  1024,
    "embed_api":   "MongoDB Atlas Embedding API"
  }
}""", language="json")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COLLECTION STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📊 MongoDB Atlas Collection Overview")
    try:
        col   = get_col()
        total = col.count_documents({})
        files = col.distinct("file_name")
        w_emb = col.count_documents({"embedding": {"$exists": True}})

        sc = st.columns(4)
        sc[0].metric("Total chunks",    total)
        sc[1].metric("Unique files",    len(files))
        sc[2].metric("With embeddings", w_emb)
        sc[3].metric("Embed dims",      EMBED_DIMS)

        st.markdown("---")
        st.markdown("### 🔍 Vector Index Status")
        try:
            indexes = list(col.list_search_indexes())
            if indexes:
                for idx in indexes:
                    status = idx.get("status", "?")
                    color  = "green" if status == "READY" else "orange"
                    st.markdown(
                        f"**`{idx.get('name')}`** — "
                        f"<span style='color:{color};'>{status}</span>",
                        unsafe_allow_html=True
                    )
                    st.json(idx.get("latestDefinition", {}))
            else:
                st.warning("No indexes found. Create `voyage_vector_index`.")
        except Exception as e:
            st.caption(f"Could not list indexes: {e}")

        st.markdown("---")
        st.markdown("### 📄 Files")
        if files:
            for fname in sorted(files):
                chunks  = list(col.find(
                    {"file_name": fname},
                    {"chunk_index": 1, "metadata": 1, "content": 1}
                ).sort("chunk_index", 1))
                total_w = sum(
                    c.get("metadata", {}).get("word_count", 0)
                    for c in chunks
                )
                with st.expander(
                    f"📄 {fname} — {len(chunks)} chunks · ~{total_w:,} words"
                ):
                    pc = st.columns(4)
                    pc[0].metric("Chunks", len(chunks))
                    pc[1].metric("~Words", f"{total_w:,}")
                    pc[2].metric("Avg",
                                 f"{total_w//max(len(chunks),1)} w/chunk")
                    pc[3].metric("Dims",   EMBED_DIMS)
                    if chunks:
                        st.markdown("**First chunk:**")
                        st.markdown(f"> {chunks[0]['content'][:400]}...")
        else:
            st.info("No documents yet. Go to **Upload & Ingest PDF** tab.")

        st.markdown("---")
        st.markdown("### 🔎 Sample Documents")
        for doc in col.find({}, {"embedding": 0}).limit(3):
            with st.expander(
                f"Chunk {doc.get('chunk_index','?')} — "
                f"{doc.get('file_name','?')}"
            ):
                st.write(doc.get("content", "")[:500])
                st.json({k: v for k, v in doc.items()
                         if k not in ("content", "_id", "embedding")})

    except Exception as e:
        st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:#888;font-size:0.82rem;'>
  🍃 MongoDB Atlas Vector Search &nbsp;·&nbsp;
  <span class='voyage-badge'>voyage-3 · 1024-dim</span>
  &nbsp;MongoDB Native Embedding API &nbsp;·&nbsp;
  🦙 llama3 (local) &nbsp;·&nbsp;
  Built by <strong>Ankur Parashar</strong> — SA Challenge Demo
</div>
""", unsafe_allow_html=True)