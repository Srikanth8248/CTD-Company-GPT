import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from auth import can_access_document

logger = logging.getLogger(__name__)

VECTOR_DIR    = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)
INDEX_FILE    = VECTOR_DIR / "faiss.index"
METADATA_FILE = VECTOR_DIR / "metadata.json"


class VectorStore:

    def __init__(self):
        self._chunks: List[str]      = []
        self._meta:   List[Dict]     = []
        self._embeddings: Optional[np.ndarray] = None
        self._load()

    def add_document(self, doc_id: str, filename: str, chunks: List[str], access_level: str = "public") -> None:
        logger.info(f"Embedding {len(chunks)} chunks for '{filename}' (access: {access_level})")
        embeddings = self._embed_texts(chunks)
        start_idx  = len(self._chunks)
        self._chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            self._meta.append({
                "doc_id":       doc_id,
                "filename":     filename,
                "access_level": access_level,
                "chunk_idx":    i,
                "global_idx":   start_idx + i,
            })
        self._embeddings = embeddings if self._embeddings is None else np.vstack([self._embeddings, embeddings])
        self._save()

    def search(self, query: str, role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._chunks or self._embeddings is None:
            return []
        q_emb  = self._embed_texts([query])[0]
        scores = self._cosine_sim(q_emb, self._embeddings)
        top_idx = np.argsort(scores)[::-1]
        results = []
        for idx in top_idx:
            if len(results) >= top_k:
                break
            if scores[idx] < 0.1:
                continue
            meta = self._meta[idx]
            # Role-based filter
            if not can_access_document(role, meta["access_level"]):
                continue
            results.append({
                "chunk":        self._chunks[idx],
                "filename":     meta["filename"],
                "doc_id":       meta["doc_id"],
                "access_level": meta["access_level"],
                "score":        float(scores[idx]),
            })
        return results

    def delete_document(self, doc_id: str) -> bool:
        keep = [i for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
        if len(keep) == len(self._meta):
            return False
        self._chunks     = [self._chunks[i] for i in keep]
        self._meta       = [self._meta[i]   for i in keep]
        self._embeddings = self._embeddings[keep] if (self._embeddings is not None and keep) else None
        for new_i, m in enumerate(self._meta):
            m["global_idx"] = new_i
        self._save()
        return True

    def total_chunks(self) -> int:
        return len(self._chunks)

    def reset(self):
        self._chunks     = []
        self._meta       = []
        self._embeddings = None
        INDEX_FILE.unlink(missing_ok=True)
        METADATA_FILE.unlink(missing_ok=True)

    # ── Embedding ──────────────────────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        groq_key = os.getenv("GROQ_API_KEY", "")
        # Groq doesn't do embeddings — use TF-IDF hash fallback (fast & free)
        return self._tfidf_embed(texts)

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        DIM = 768
        vectors = []
        for text in texts:
            vec  = np.zeros(DIM, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                idx = hash(word) % DIM
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(len(matrix))
        m_norms = np.linalg.norm(matrix, axis=1)
        m_norms[m_norms == 0] = 1e-9
        return (matrix @ query) / (m_norms * q_norm)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self):
        try:
            METADATA_FILE.write_text(
                json.dumps({"chunks": self._chunks, "meta": self._meta}, ensure_ascii=False)
            )
            if self._embeddings is not None:
                with open(INDEX_FILE, "wb") as f:
                    pickle.dump(self._embeddings, f)
        except Exception as exc:
            logger.error(f"Save failed: {exc}")

    def _load(self):
        try:
            if METADATA_FILE.exists():
                data = json.loads(METADATA_FILE.read_text())
                self._chunks = data.get("chunks", [])
                self._meta   = data.get("meta",   [])
                logger.info(f"Loaded {len(self._chunks)} chunks from disk")
            if INDEX_FILE.exists():
                with open(INDEX_FILE, "rb") as f:
                    self._embeddings = pickle.load(f)
        except Exception as exc:
            logger.warning(f"Load failed ({exc}), starting fresh")
            self._chunks     = []
            self._meta       = []
            self._embeddings = None
