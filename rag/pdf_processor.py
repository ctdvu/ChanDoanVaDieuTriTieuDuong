"""
pdf_processor.py
Xử lý PDF hướng dẫn chẩn đoán điều trị tiểu đường.
Dùng PyMuPDF (fitz) để extract text, sau đó chunk bằng RecursiveCharacterTextSplitter.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract toàn bộ text từ PDF."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Trang {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(full_text)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Chia text thành các chunk nhỏ với overlap.
    Trả về list dict: {"chunk_id": int, "text": str, "char_start": int}
    """
    chunks = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Cố gắng cắt tại dấu xuống dòng hoặc dấu chấm để không cắt giữa câu
        if end < text_len:
            # Tìm dấu xuống dòng gần nhất phía trước end
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 1
            else:
                # Tìm dấu chấm gần nhất
                dot_pos = text.rfind(".", start, end)
                if dot_pos > start + chunk_size // 2:
                    end = dot_pos + 1

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "char_start": start,
            })
            chunk_id += 1

        start = end - chunk_overlap if end < text_len else text_len

    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Pipeline đầy đủ: PDF → text → chunks.
    Trả về list chunks với metadata.
    """
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    print(f"[PDF Processor] Đang extract text từ: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"[PDF Processor] Đã extract {len(text):,} ký tự")

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[PDF Processor] Đã tạo {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # Thêm source metadata vào mỗi chunk
    pdf_name = Path(pdf_path).name
    for chunk in chunks:
        chunk["source"] = pdf_name

    return chunks


if __name__ == "__main__":
    # Test nhanh
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "HD-chan-doan-dieu-tri-DTD-2017_07_19-Approved.pdf"
    chunks = process_pdf(pdf_file)
    print(f"\nSample chunk #0:")
    print(chunks[0]["text"][:300] if chunks else "Không có chunk nào")
