
import streamlit as st
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import faiss
import requests
from sentence_transformers import SentenceTransformer
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "VoltAI — EV Outlook Assistant"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "phi3:mini"

INDEX_FILE = Path("vector_db/iea_faiss.index")
META_FILE = Path("vector_db/iea_metadata.jsonl")

# ----------------------------
# LOAD RESOURCES (cached)
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_vector_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
    return faiss.read_index(str(INDEX_FILE))

@st.cache_resource
def load_metadata():
    if not META_FILE.exists():
        raise FileNotFoundError(f"Metadata not found: {META_FILE}")
    meta = []
    with META_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

embedder = load_embedder()
index = load_vector_index()
metadata = load_metadata()

# ----------------------------
# OLLAMA CLIENT
# ----------------------------
def ollama_generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    max_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    context_window: int = 2048,
    timeout_sec: int = 60
) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": context_window
        }
    }
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ----------------------------
# RETRIEVER
# ----------------------------
def retrieve(query: str, k: int = 5) -> List[Dict]:
    qv = embedder.encode(query, normalize_embeddings=True).astype("float32")
    D, I = index.search(np.array([qv]), k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results

# ----------------------------
# PROMPT BUILDER
# ----------------------------
SYSTEM_PROMPT = """
You are VoltAI, a domain-specific assistant for electric vehicle market trends, charging infrastructure, battery ecosystem, and EV policy.
You must answer ONLY using the provided EVIDENCE from the IEA Global EV Outlook knowledge base.

Rules:
1) Use ONLY information present in EVIDENCE.
2) If the answer is not found in EVIDENCE, respond exactly with: "Insufficient data in knowledge base."
3) Be factual, concise, and avoid assumptions.
4) When relevant, include numbers and year references.
5) Use bullet points.
6) Keep answer under 140 words.
7) End with a Sources section listing sources used.
""".strip()

def format_evidence(chunks: List[Dict], max_chars_per_chunk: int = 650) -> str:
    formatted = []
    for i, ch in enumerate(chunks, 1):
        txt = ch["text"].replace("\n", " ").strip()[:max_chars_per_chunk]
        formatted.append(
            f"[EVIDENCE {i}] (SOURCE={ch['source']}, YEAR={ch['year']}, CHUNK_ID={ch['chunk_id']}, SCORE={ch.get('score',0):.4f})\n"
            f"{txt}"
        )
    return "\n\n".join(formatted)

def format_chat_history(chat_history: List[Dict], max_turns: int = 6) -> str:
    if not chat_history:
        return "None"
    recent = chat_history[-max_turns:]
    lines = []
    for msg in recent:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content'].strip()}")
    return "\n".join(lines)

def build_prompt(user_query: str, retrieved_chunks: List[Dict], chat_history: List[Dict]) -> str:
    return f"""
{SYSTEM_PROMPT}

CHAT HISTORY:
{format_chat_history(chat_history)}

EVIDENCE:
{format_evidence(retrieved_chunks)}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Bullet points only
- Under 140 words
- End with Sources: ...
""".strip()

def voltai_answer(user_query: str, chat_history: List[Dict], top_k: int = 5) -> Dict:
    retrieved = retrieve(user_query, k=top_k)
    prompt = build_prompt(user_query, retrieved, chat_history)

    answer = ollama_generate(prompt)

    sources = [{
        "source": ch["source"],
        "year": ch["year"],
        "chunk_id": ch["chunk_id"],
        "score": ch["score"]
    } for ch in retrieved]

    new_history = chat_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer}
    ]

    return {
        "answer": answer,
        "sources": sources,
        "evidence": retrieved,
        "chat_history": new_history
    }

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="VoltAI", layout="wide")
st.title(APP_TITLE)
st.caption("Contextual Conversational RAG demo (IEA Global EV Outlook 2022–2024)")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    top_k = st.slider("Top-k evidence chunks", 3, 10, 5, 1)
    max_tokens = st.slider("Max output tokens", 120, 350, 220, 10)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    show_evidence = st.checkbox("Show retrieved evidence", value=True)
    show_sources = st.checkbox("Show sources", value=True)

    st.divider()
    st.subheader("Ollama Status")
    if check_ollama():
        st.success("Ollama server running ✅")
    else:
        st.error("Ollama not reachable ❌")
        st.info("Run in terminal: `ollama serve`")

    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

# Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_query = st.chat_input("Ask VoltAI about EV trends, charging, batteries, policies...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("VoltAI is thinking..."):
            # apply runtime sliders to generator options
            def ollama_generate_ui(prompt: str) -> str:
                return ollama_generate(
                    prompt=prompt,
                    model=OLLAMA_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    context_window=2048,
                    timeout_sec=90
                )

            # override generation function for UI
            retrieved = retrieve(user_query, k=top_k)
            prompt = build_prompt(user_query, retrieved, st.session_state.chat_history)

            try:
                answer = ollama_generate_ui(prompt)
            except Exception as e:
                answer = f"⚠️ Generation failed: {e}"

            st.markdown(answer)

            # Show sources
            if show_sources:
                with st.expander("Sources"):
                    for ch in retrieved:
                        st.write(f"- {ch['source']} ({ch['year']}) | {ch['chunk_id']} | score={ch['score']:.3f}")

            # Show evidence
            if show_evidence:
                with st.expander("Retrieved Evidence"):
                    for i, ch in enumerate(retrieved, 1):
                        st.markdown(f"**Evidence {i}** — `{ch['chunk_id']}` | {ch['source']} ({ch['year']}) | score={ch['score']:.3f}")
                        st.write(ch["text"][:1200] + "...")
                        st.divider()

    # Update memory
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history = st.session_state.chat_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer}
    ]
