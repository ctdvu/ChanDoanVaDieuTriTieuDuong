"""
rag_pipeline.py
RAG Pipeline: PDF → Embedding → ChromaDB → Retrieval → Response
Dùng sentence-transformers (local, không cần API key) + ChromaDB.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

# Sentence transformers cho embedding
from sentence_transformers import SentenceTransformer

# ChromaDB
import chromadb
from chromadb.config import Settings

from pdf_processor import process_pdf


# ─── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Hỗ trợ tiếng Việt
COLLECTION_NAME = "diabetes_treatment"
TOP_K = 5


class RAGPipeline:
    """
    Pipeline RAG cho phác đồ điều trị tiểu đường.
    
    Usage:
        rag = RAGPipeline()
        rag.build_index("path/to/HD-chan-doan.pdf")  # 1 lần duy nhất
        result = rag.query("Phác đồ điều trị tiểu đường type 2?")
    """

    def __init__(self, vector_store_dir: Optional[str] = None):
        self.vector_store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RAG] Đang tải embedding model: {EMBED_MODEL_NAME}")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
        )
        self.collection = None
        self._load_or_create_collection()

    def _load_or_create_collection(self):
        """Load collection đã có hoặc tạo mới."""
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            count = self.collection.count()
            print(f"[RAG] Loaded collection '{COLLECTION_NAME}' với {count} documents")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[RAG] Tạo mới collection '{COLLECTION_NAME}'")

    def build_index(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Build vector index từ PDF.
        Xóa collection cũ nếu đã có, tạo lại từ đầu.
        """
        # Xóa collection cũ
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            print("[RAG] Đã xóa collection cũ")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Process PDF
        chunks = process_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            print("[RAG] CẢNH BÁO: Không có chunk nào được tạo!")
            return

        # Embed và add vào ChromaDB theo batch
        BATCH_SIZE = 100
        total = len(chunks)
        print(f"[RAG] Đang embed {total} chunks...")

        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            ids = [f"chunk_{c['chunk_id']}" for c in batch]
            metadatas = [{"source": c["source"], "char_start": c["char_start"]} for c in batch]

            embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            print(f"[RAG]   Đã add {min(i + BATCH_SIZE, total)}/{total} chunks")

        print(f"[RAG] ✅ Build index hoàn thành! Tổng: {self.collection.count()} documents")

    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        Truy vấn RAG: tìm chunks liên quan nhất, trả về response có nguồn trích dẫn.
        
        Returns:
            {
                "question": str,
                "answer": str,          # Câu trả lời tổng hợp
                "sources": List[str],   # Các đoạn trích từ tài liệu
                "num_retrieved": int,
            }
        """
        if self.collection is None or self.collection.count() == 0:
            return {
                "question": question,
                "answer": "Chưa có dữ liệu phác đồ điều trị. Vui lòng build index từ PDF trước.",
                "sources": [],
                "num_retrieved": 0,
            }

        # Embed câu hỏi
        query_embedding = self.embed_model.encode([question])[0].tolist()

        # Tìm top-k chunks gần nhất
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Lọc kết quả có cosine distance < 0.8 (similarity > 0.2)
        relevant = [
            (doc, dist, meta)
            for doc, dist, meta in zip(documents, distances, metadatas)
            if dist < 0.8
        ]

        if not relevant:
            return {
                "question": question,
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu phác đồ điều trị.",
                "sources": [],
                "num_retrieved": 0,
            }

        # Tổng hợp answer từ các chunks retrieved
        context_parts = []
        sources = []
        for idx, (doc, dist, meta) in enumerate(relevant):
            context_parts.append(f"[{idx+1}] {doc}")
            sources.append({
                "text": doc[:200] + "..." if len(doc) > 200 else doc,
                "source": meta.get("source", ""),
                "relevance_score": round(1 - dist, 3),
            })

        context = "\n\n".join(context_parts)

        # Template response (không cần LLM API)
        answer = self._format_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_retrieved": len(relevant),
        }

    def _format_answer(self, question: str, context: str) -> str:
        """
        Format câu trả lời từ context retrieved.
        Dùng template đơn giản, không cần LLM.
        """
        # Trích xuất các câu quan trọng từ context
        lines = [line.strip() for line in context.split("\n") if line.strip() and len(line.strip()) > 20]
        
        # Bỏ các dòng là heading số như [1], [2]
        content_lines = [l for l in lines if not l.startswith("[") or len(l) > 5]
        
        # Lấy tối đa 10 câu
        selected = content_lines[:10]
        
        answer = (
            f"📋 **Thông tin từ Hướng dẫn Chẩn đoán và Điều trị Đái tháo đường:**\n\n"
            + "\n".join(f"• {line}" for line in selected)
            + "\n\n⚠️ *Đây là thông tin tham khảo từ tài liệu y tế. "
            "Vui lòng tham khảo ý kiến bác sĩ để được tư vấn cụ thể.*"
        )
        return answer

    def is_ready(self) -> bool:
        """Kiểm tra xem index đã được build chưa."""
        return self.collection is not None and self.collection.count() > 0


# ─── Singleton instance ──────────────────────────────────────────────────────
_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Trả về singleton RAGPipeline."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


if __name__ == "__main__":
    import sys

    rag = RAGPipeline()

    # Build index nếu chưa có
    if not rag.is_ready():
        pdf_file = sys.argv[1] if len(sys.argv) > 1 else "../HD-chan-doan-dieu-tri-DTD-2017_07_19-Approved.pdf"
        rag.build_index(pdf_file)

    # Demo queries
    demo_queries = [
        "Phác đồ điều trị tiểu đường type 2 ở bệnh nhân béo phì?",
        "Liều dùng Metformin khởi đầu?",
        "Thay đổi lối sống cho bệnh nhân tiền tiểu đường?",
        "Mục tiêu kiểm soát HbA1c?",
    ]

    for q in demo_queries:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        result = rag.query(q)
        print(f"✅ {result['answer'][:500]}")
        print(f"📚 Retrieved: {result['num_retrieved']} chunks")
