import glob
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Vietnamese
sys.stdout.reconfigure(encoding='utf-8')

from src.chunking import RecursiveChunker, SemanticChunker
from src.embeddings import _mock_embed, LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore

# Use local embedding if possible to get real similarity scores!
try:
    from sentence_transformers import SentenceTransformer
    embedder = LocalEmbedder()
except Exception:
    print("WARNING: Using mock embedder. Vector search won't be contextually accurate. Please 'pip install sentence-transformers' for real embeddings.")
    embedder = _mock_embed

def main():
    print("=== Bước 1: Load và Chunk tài liệu Quy Chế ===")
    data_dir = Path("data/Quy_che")
    md_files = list(data_dir.glob("*.md"))
    
    if not md_files:
        print(f"Không tìm thấy file .md nào trong {data_dir}")
        return

    # Commented out RecursiveChunker as requested
    # chunker = RecursiveChunker(chunk_size=400)
    
    # Use Semantic Chunker
    print("Khởi tạo SemanticChunker (Ngưỡng similarity = 0.45)...")
    chunker = SemanticChunker(embedding_fn=embedder, threshold=0.45)
    all_chunks = []
    
    for file_path in md_files:
        content = file_path.read_text(encoding="utf-8")
        # Chia text thành các chunks nhỏ
        chunks = chunker.chunk(content)
        print(f"Loaded {file_path.name}: {len(chunks)} chunks")
        
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                id=f"{file_path.stem}_chunk_{i}",
                content=chunk_text,
                metadata={"category": "quy_che", "file_source": file_path.name}
            )
            all_chunks.append(doc)
            
    print(f"\nTổng số chunks tạo ra: {len(all_chunks)}")

    print("\n=== Bước 2: Nạp vào Vector Store ===")
    store = EmbeddingStore(collection_name="quy_che_store", embedding_fn=embedder)
    store.add_documents(all_chunks)
    print(f"Đã lưu {store.get_collection_size()} chunks vào EmbeddingStore.")

    print("\n=== Bước 3: Đặt câu hỏi hỏi đáp (Tương tác) ===")
    print("Hệ thống đã sẵn sàng! Mời bạn đặt câu hỏi về Quy chế (Gõ 'q' hoặc 'exit' để thoát).")
    
    while True:
        try:
            query = input("\n> Câu hỏi của bạn: ").strip()
            if not query:
                continue
            if query.lower() in ['q', 'quit', 'exit']:
                print("Đóng hệ thống. Tạm biệt!")
                break
                
            results = store.search(query, top_k=3)
            
            print(f"\n--- Kết quả trích xuất ---")
            if not results:
                print("Không tìm thấy thông tin nào phù hợp.")
            else:
                for r_idx, r in enumerate(results, 1):
                    score = r.get('score', 0)
                    source = r.get('metadata', {}).get('file_source', 'Unknown')
                    content_preview = r['content'].replace('\n', ' ')[:200]
                    print(f"\n[Top {r_idx}] (Mức độ phù hợp: {score:.4f}) - Trích từ tài liệu: {source}")
                    print(f"Nội dung: {content_preview}...")
                    
        except KeyboardInterrupt:
            print("\nĐóng hệ thống. Tạm biệt!")
            break

if __name__ == "__main__":
    main()
