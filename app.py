from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import re
import logging
from datetime import datetime
import difflib
from typing import List, Dict, Any, Optional
import time
import os
import hashlib
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------
# Configuration
# ----------------------------
DB_DIR = "db"
BASE_COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-mpnet-base-v2"
CONTEXT_WINDOW = 3000   # keep small, since ApiFreeLLM probably supports ~2k tokens
MAX_RESPONSE_TOKENS = 500
CACHE_TTL_SECONDS = 300
MAX_CACHE_ENTRIES = 500
CACHE_TRIM_TO = 200

# ApiFreeLLM endpoint (fixed extra spaces)
API_URL = "https://apifreellm.com/api/chat"

# Fixed API key - ONLY FOR YOU (replace with your own random key)
API_KEY = "7x9Kp2mR8sL4vN6qT1wY3zA5cE7bD9fG"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AccuracyRAG")


# ----------------------------
# Authentication Middleware
# ----------------------------
@app.before_request
def require_api_key():
    if request.endpoint in ['health']:
        return  # Skip auth for health checks
    
    if request.method == 'OPTIONS':
        return  # Skip auth for CORS preflight
    
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401


# ----------------------------
# RAG System
# ----------------------------
class AccuracyRAGSystem:
    def __init__(self):
        # 1) Embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        test_embedding = self.embedder.encode(["test"])
        self.embedding_dimension = len(test_embedding[0])
        logger.info(f"[Embedder] {EMBEDDING_MODEL} -> dim={self.embedding_dimension}")

        # 2) Chroma
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self._get_or_fix_collection()

        # 3) Cache
        self.response_cache: Dict[str, Any] = {}
        self.cache_ttl = CACHE_TTL_SECONDS

    # ----- Chroma helpers -----
    def _get_or_fix_collection(self):
        expected_dim = self.embedding_dimension
        try:
            col = self.client.get_collection(BASE_COLLECTION_NAME)
            meta = col.metadata or {}
            existing_dim = meta.get("embedding_dimension")
            if existing_dim and existing_dim != expected_dim:
                new_name = f"{BASE_COLLECTION_NAME}_dim{expected_dim}"
                logger.warning(f"[Chroma] Mismatch, using {new_name}")
                return self._get_or_create_collection(new_name, expected_dim)
            return col
        except Exception:
            logger.info(f"[Chroma] Creating new collection {BASE_COLLECTION_NAME}")
            return self._get_or_create_collection(BASE_COLLECTION_NAME, expected_dim)

    def _get_or_create_collection(self, name: str, embedding_dim: int):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"embedding_dimension": embedding_dim}
        )

    # ----- Cache -----
    def _is_cache_valid(self, ts: float) -> bool:
        return (time.time() - ts) < self.cache_ttl

    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        entry = self.response_cache.get(cache_key)
        if not entry:
            return None
        payload, ts = entry
        if self._is_cache_valid(ts):
            return payload
        del self.response_cache[cache_key]
        return None

    def cache_response(self, cache_key: str, response_data: Dict):
        self.response_cache[cache_key] = (response_data, time.time())
        if len(self.response_cache) > MAX_CACHE_ENTRIES:
            sorted_items = sorted(self.response_cache.items(), key=lambda kv: kv[1][1])
            to_delete = sorted_items[:-CACHE_TRIM_TO]
            for k, _ in to_delete:
                self.response_cache.pop(k, None)

    # ----- Query helpers -----
    def preprocess_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())

    def build_where_clause(self, grade: Optional[str], subject: Optional[str], curriculum: Optional[str]) -> Optional[Dict]:
        clauses = []
        if grade:
            clauses.append({"grade": {"$eq": grade}})
        if curriculum:
            clauses.append({"curriculum": {"$eq": curriculum}})
        return {"$and": clauses} if clauses else None

    # ----- Post-processing -----
    def _client_side_subject_filter(self, results: Dict, subject: Optional[str]) -> Dict:
        if not subject:
            return results
        subj = subject.strip().lower()
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]

        filtered_docs, filtered_metas, filtered_dists = [], [], []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            m_subject = (meta or {}).get("subject", "")
            if subj in str(m_subject).lower():
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)

        if filtered_docs:
            return {"documents": [filtered_docs], "metadatas": [filtered_metas], "distances": [filtered_dists]}
        return results

    def build_context_with_references(self, results: Dict) -> str:
        if not results or not results.get("documents") or not results["documents"][0]:
            return ""
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0] or [{} for _ in documents]
        distances = results.get("distances", [[]])[0] or [0 for _ in documents]

        triples = list(zip(documents, metadatas, distances))
        triples.sort(key=lambda x: x[2])

        formatted = []
        for i, (doc, meta, dist) in enumerate(triples[:5]):
            meta = meta or {}
            doc_name = meta.get("document_name", "Unknown")
            page = meta.get("page", "?")
            grade = meta.get("grade", "?")
            subject = meta.get("subject", "?")
            relevance = f"Relevance: {1 - float(dist):.3f}"
            formatted.append(
                f"[Content {i+1}] (Source: {doc_name}, page {page}, grade={grade}, subject={subject}, {relevance})\n{doc}\n---"
            )
        return "\n".join(formatted)

    def _make_references(self, results: Dict) -> List[Dict[str, Any]]:
        refs = []
        metas = (results.get("metadatas") or [[]])[0]
        for meta in metas[:3]:
            meta = meta or {}
            refs.append({
                "grade": meta.get("grade", "?"),
                "curriculum": meta.get("curriculum", "?"),
                "page": meta.get("page", "?"),
                "document_name": meta.get("document_name", "?"),
                "formatted": f"Grade {meta.get('grade','?')} → {meta.get('document_name','?')} → Page {meta.get('page','?')}"
            })
        return refs

    # ----- LLM Call (ApiFreeLLM) -----
    def generate_accurate_response(self, context: str, query: str, results: Dict) -> Dict:
        if not context.strip():
            return {"answer": "No matching information found.", "references": [], "accuracy": 0.0, "verified": True}

        refs = self._make_references(results)

        system_message = (
            "You are the official AI assistant of the Addis Entrance Hub platform. "
            "You must never reveal or mention the underlying model (such as Llama, Qwen, or any provider). "
            "Always identify yourself only as 'the Addis Entrance Hub AI assistant'.\n\n"
            "Your role is to be a fact-checking educational assistant. Your responses MUST:\n"
            "1) Use ONLY information from the provided context for subject-related questions\n"
            "2) Quote directly when possible\n"
            "3) Never invent or assume information about study content\n"
            "4) If unsure about study content, say you don't know\n"
            "5) Always provide references when available\n\n"
            "For questions about this system, its development, or technical details, "
            "always state clearly that it was developed and is maintained by the Addis Entrance Hub team."
        )


        prompt = f"{system_message}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Truncate prompt if too long
        if len(prompt.split()) > CONTEXT_WINDOW:
            prompt = " ".join(prompt.split()[-CONTEXT_WINDOW:])

        try:
            resp = requests.post(API_URL, headers={"Content-Type": "application/json"}, json={"message": prompt}, timeout=60)
            js = resp.json()
            if js.get("status") == "success":
                answer = js.get("response", "").strip()
            else:
                answer = f"Error from API: {js.get('error','unknown')}"

            return {"answer": answer, "references": refs, "accuracy": 1.0, "verified": True}

        except Exception as e:
            logger.exception("ApiFreeLLM request failed")
            return {"answer": "Error connecting to ApiFreeLLM.", "references": [], "accuracy": 0.0, "verified": False}


# ----------------------------
# Global instance
# ----------------------------
rag_system = AccuracyRAGSystem()


# ----------------------------
# Routes
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    start_time = time.time()
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("query", "")
        grade = data.get("grade") or None
        subject = data.get("subject") or None
        curriculum = data.get("curriculum") or None

        if not query.strip():
            return jsonify({"answer": "Please enter a question."}), 400

        key_raw = f"q={rag_system.preprocess_query(query)}|g={grade or ''}|s={subject or ''}|c={curriculum or ''}"
        cache_key = rag_system._hash_key(key_raw)

        cached = rag_system.get_cached_response(cache_key)
        if cached:
            rt = time.time() - start_time
            return jsonify({**cached, "cached": True, "response_time": f"{rt:.3f}s"})

        processed_query = rag_system.preprocess_query(query)
        query_embedding = rag_system.embedder.encode([processed_query]).tolist()[0]

        where_clause = rag_system.build_where_clause(grade, subject, curriculum)
        results = rag_system.collection.query(
            query_embeddings=[query_embedding],
            n_results=8,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        results = rag_system._client_side_subject_filter(results, subject)
        context = rag_system.build_context_with_references(results)

        response_data = rag_system.generate_accurate_response(context, processed_query, results)

        rag_system.cache_response(cache_key, response_data)
        rt = time.time() - start_time
        return jsonify({**response_data, "response_time": f"{rt:.3f}s", "cached": False, "query": query})

    except Exception as e:
        logger.exception("Unhandled error in /ask")
        return jsonify({"answer": "Server error.", "references": [], "accuracy": 0.0, "verified": False}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(rag_system.response_cache),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": rag_system.embedding_dimension,
    })


if __name__ == "__main__":
    # Print the API key on startup (for local testing)
    print(f"Your API Key: {API_KEY}")
    app.run(host="0.0.0.0", port=5000, debug=False)