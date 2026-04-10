# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Duy Hưng
**Nhóm:** Nhóm A3
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có góc giữa chúng rất nhỏ (gần 0), đồng nghĩa với việc hai đoạn văn bản mà chúng đại diện có nội dung và ngữ nghĩa cực kỳ sát nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Sinh viên phải nộp học phí đúng hạn."
- Sentence B: "Thời hạn đóng tiền học phải được sinh viên tuân thủ nghiêm ngặt."
- Tại sao tương đồng: Khác biệt hoàn toàn về mặt từ vựng (nộp/đóng, học phí/tiền học), nhưng mang chung một ý nghĩa căn bản về nghĩa vụ tài chính.

**Ví dụ LOW similarity:**
- Sentence A: "Hôm nay trời mưa rất to ở Hà Nội."
- Sentence B: "Quy chế bảo lưu học tập dành cho sinh viên năm nhất."
- Tại sao khác: Hai câu bàn về hai chủ đề hoàn toàn không liên quan (thời tiết vs quy chế đào tạo).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine đo lường *góc* (hướng) của vector thay vì *độ lớn*. Nhờ vậy, một câu ngắn và một đoạn văn dài có cùng ý nghĩa vẫn sẽ có độ tương đồng cao (do vector chỉ cùng hướng) mặc dù khoảng cách Euclid giữa chúng rất xa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Kích thước hiệu dụng của mỗi chunk (trừ chunk đầu) là 500 - 50 = 450 ký tự. Số chunk = ceil(10,000 / 450) = ceil(22.2)
> *Đáp án:* Khoảng 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Chunk count sẽ tăng lên (Khoảng ceil(10000 / 400) = 25 chunks). Ta muốn overlap nhiều hơn để tránh trường hợp một khái niệm quan trọng bị cắt vụn đúng ở ranh giới giữa hai chunk, việc overlap giúp gối đầu thông tin đảm bảo context không bị đứt gãy.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quy chế học vụ và quản lý sinh viên (University Regulations).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì đây là bộ quy tắc cực kỳ quan trọng đối với sinh viên, chứa nhiều con số, điều kiện và quy trình phức tạp (đăng ký học phần, điểm số, tốt nghiệp). Việc xây dựng hệ thống RAG cho domain này sẽ hỗ trợ sinh viên tra cứu quy chế nhanh chóng và chính xác.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | dang_ky_hoc_phan.md | Phòng Đào tạo | 2,599 | type: hoc_vu, urgency: high |
| 2 | diem_va_xep_loai.md | Sổ tay sinh viên | 2,642 | type: hoc_vu, urgency: medium |
| 3 | ho_tro_sinh_vien.md | Phòng CT&CTSV | 2,825 | type: chinh_sach, urgency: low |
| 4 | hoc_phi_hoc_bong.md | Phòng Kế hoạch Tài chính | 2,783 | type: tai_chinh, urgency: medium |
| 5 | ky_luat_chuyen_can.md | Sổ tay sinh viên | 2,908 | type: ky_luat, urgency: low |
| 6 | tot_nghiep.md | Quy định tốt nghiệp | 3,012 | type: tot_nghiep, urgency: high |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| type | string | hoc_vu, tai_chinh | Cho phép sinh viên lọc câu hỏi theo mảng kiến thức (ví dụ: chỉ tìm trong mảng Học phí). |
| urgency | string | high, medium | Giúp hệ thống ưu tiên hiển thị các quy định có tính thời hạn hoặc mức độ quan trọng cao. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Quy chế | FixedSizeChunker (`fixed_size`) | ~70 | 200 | Kém, rất hay cắt ngang giữa câu (gây gãy nghĩa). |
| Quy chế | SentenceChunker (`by_sentences`) | ~65 | 220 | Tốt hơn, nhưng đôi khi cụm 3 câu lại quá rời rạc (câu 1 của đoạn này, câu 2 của đoạn trước). |
| Quy chế | RecursiveChunker (`recursive`) | ~55 | 270 | Khá tốt, do dựa vào cấu trúc xuống dòng `\n\n` để gộp đoạn văn. |

### Strategy Của Tôi

**Loại:** Custom Strategy - `SemanticChunker` (Chunking dựa trên độ tương đồng ngữ nghĩa).

**Mô tả cách hoạt động:**
> Module gỡ văn bản thành từng câu lẻ bằng Regex (`.!?`). Sau đó dùng một Embedding model (`LocalEmbedder`) để mã hóa từng câu. Nó đi theo sequence từ câu 1 đến câu N, nếu Cosine Similarity giữa câu hiện tại và câu trước đó lớn hơn một ngưỡng (threshold = 0.45) thì nó gộp lại thành 1 Chunk. Nếu độ tương đồng tụt sập, chứng tỏ đang chuyển chủ đề -> tách chunk mới.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Quy chế học vụ rất dài và chứa các "điều khoản" có độ dài ngắn khác nhau. Việc cắt ngữ nghĩa giúp đảm bảo "1 điều khoản luật" hoặc "mô tả của 1 quy trình" được giữ trọn vẹn trong một khối thông tin vì nghĩa của chúng liên kết chặt chẽ.

**Code snippet (nếu custom):**
```python
        for i in range(1, len(sentences)):
            sim = compute_similarity(embeddings[i-1], embeddings[i])
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length (chars) | Retrieval Score (/10) | Nhận xét |
|-----------|----------|-------------|--------------------|-----------------------|-----------|
| Quy chế chung | **FixedSizeChunker** (size=200) | ~70 | 200 | 4 / 10 | Cắt ngang câu, gây mất context ở đoạn giữa điều khoản. Chunk ngắn nên vector không đủ thông tin ngữ nghĩa. |
| Quy chế chung | **SentenceChunker** (3 câu) | ~65 | 220 | 6 / 10 | Giữ trọn câu nhưng 3 câu đôi khi thuộc 2 đoạn ý khác nhau, retrieval vẫn bị nhiễu. |
| Quy chế chung | **RecursiveChunker** | ~55 | 270 | 7 / 10 | Bám theo cấu trúc `\n\n` của Markdown, giữ nguyên đoạn ý. Baseline tốt nhất trong 3 baseline. |
| Quy chế chung | **SemanticChunker** (threshold=0.45) *(của tôi)* | ~45 | 320 | 4 / 10 | Chunk ngữ nghĩa tập trung hơn nhưng mô hình embedding nhỏ (`all-MiniLM-L6-v2`) chưa đủ mạnh để phân biệt ranh giới chủ đề tiếng Việt chuyên ngành. Cần fine-tuned model mới phát huy hết tiềm năng. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Semantic | 4 / 10 | Giữ trọn vẹn ngữ cảnh của đoạn văn, gom chung nhóm ý tưởng rất tốt để làm context cho LLM. | Tốn nhiều bước tính toán (chạy nhúng từng câu), nếu xác định ngưỡng threshold sai thì có thể thu nhầm cả cụm đoạn không liên quan. |
| Nguyễn Văn Bách | Recursive | 9/10 | Giữ trọn vẹn ngữ cảnh Điều/Khoản | Cấu trúc đệ quy phức tạp |
| Nguyễn Đức Duy | SentenceChunker (3 câu) | 8/10 | Giữ nguyên ý trọn vẹn, phù hợp văn bản quy chế | Chunk dài hơn, avg ~300 chars |
| Trần Trọng Giang | MarkdownHeader | 8/10 | Tối ưu tuyệt đối cho cấu trúc Markdown | Chỉ hiệu quả với file có Header rõ ràng | 
| Nguyễn Xuân Hoàng | FixedSize (Size 200, Overlap 50) | 4/10 | Triển khai nhanh, dễ tính toán số lượng | Hay cắt ngang từ, phân mảnh câu |
| Nguyễn Tuấn Kiệt | Recursive | 8/10 | Lấy được điều khoản đúng, giữ được context | Score match khá thấp dù top 3 đúng, 1 chunk = 1 câu có khả năng mất info dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Dựa trên kết quả thực nghiệm của cả nhóm, **RecursiveChunker** và **MarkdownHeaderChunker** cho retrieval score cao nhất (8–9/10) với bộ dữ liệu Quy chế dạng Markdown này. Lý do cốt lõi là các file quy chế đã có sẵn cấu trúc tiêu đề (##) và đoạn văn rõ ràng — đây chính là tín hiệu chia cắt tự nhiên mà RecursiveChunker khai thác rất hiệu quả. SemanticChunker về lý thuyết mạnh hơn vì không phụ thuộc cấu trúc văn bản, nhưng trong thực tế bài lab, hiệu quả còn bị giới hạn bởi mô hình embedding đa ngôn ngữ nhỏ chưa hiểu tốt thuật ngữ quy chế tiếng Việt. Nếu có embedding model chuyên ngành giáo dục Việt Nam, SemanticChunker sẽ là lựa chọn số một.


---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex phân tách bằng các kí tự tận cùng mốc câu `(?<=[.!?])\s+` để xẻ nhỏ chuỗi. Sau đó tích lũy mảng tạm `current_chunk`, đếm đủ `max_sentences_per_chunk` thì join lại cắt ra.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Cơ chế đệ quy: lặp qua các dấu phân tách (từ lỏng nhất như `\n\n` đến chặt nhất như khoảng trắng). Nếu chẻ theo dấu hiện tại mà chunk vẫn lớn hơn `chunk_size`, hàm sẽ lấy cái chunk đó gọi ngược lại hàm _split với các mốc phân tách sâu hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Sử dụng ChromaDB làm lõi để xử lý (hoặc List cho Fallback). Lúc Add, gán UUID và đẩy trọn ID/Metadata/Vector vào db. Lúc Search, lấy text nhập vào đi Embed, lấy Vector xuất ra đi gọi thẳng hàm `.query` của Chroma tìm N kết quả gần nhất.

**`search_with_filter` + `delete_document`** — approach:
> Truyền tham số `where=metadata_filter` thẳng vào API query của ChromaDB. Đối với delete, dùng query `where` theo metadata `doc_id` của Document để xác định và loại bỏ sạch sẽ các Chunk thuộc chung ID đó.

### KnowledgeBaseAgent

**`answer`** — approach:
> Sau khi có List kết quả từ `search()`, Agent xây dựng một chuỗi Context gồm `[Chunk N]: content...`. Hệ thống ném chuỗi Context đó cùng với Question gốc vào rập khuôn Prompt "Bạn là AI trả lời bằng văn cảnh..." và gọi hàm `llm_fn(prompt)`.

### Test Results

```
test_solution.py::test_fixed_size_chunker PASSED
test_solution.py::test_sentence_chunker PASSED
...
test_solution.py::test_knowledge_base_agent PASSED
===================== 42 passed in 1.15s =====================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Sinh viên phải nộp học phí" | "Học viên đóng tiền học tập" | high | 0.82 | Đúng |
| 2 | "Điều kiện cấp bằng tốt nghiệp" | "Quy trình đăng ký tín chỉ" | low | 0.15 | Đúng |
| 3 | "Đuổi học nếu vi phạm" | "Kỷ luật buộc thôi học" | high | 0.91 | Đúng |
| 4 | "Xin bảo lưu 1 năm" | "Làm thế nào để nghỉ học tạm thời" | high | 0.78 | Đúng |
| 5 | "Thư viện mở đến mấy giờ" | "Xe bus đưa rước sinh viên" | low | 0.08 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là "Đuổi học" và "Buộc thôi học" (Pair 3) có độ tương đồng cực kỳ cao (>0.9) mặc dù từ vựng khác hẳn nhau. Điều này chứng minh Embedding model đã vượt qua được giới hạn Keyword Matching để học được không gian Ngữ Nghĩa sâu bên dưới chữ viết!

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp? | Dao động từ 120 đến 150 tín chỉ tùy theo ngành đào tạo. |
| 2 | GPA bao nhiêu thì được làm luận văn tốt nghiệp? | Sinh viên có GPA tích lũy từ 2.8 trở lên. |
| 3 | Những đối tượng nào được miễn giảm học phí? | Sinh viên thuộc hộ nghèo, cận nghèo, con thương binh liệt sĩ, khuyết tật. |
| 4 | Khi nào sinh viên bị cảnh cáo học vụ? | Khi GPA học kỳ thấp dưới 1.0 hoặc GPA tích lũy dưới 1.2 (năm 1). |
| 5 | Chuẩn đầu ra ngoại ngữ để tốt nghiệp là gì? | Chứng chỉ B1 quốc tế hoặc tương đương theo khung 6 bậc Việt Nam. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên cần bao nhiêu tín ch... | "# Điều kiện xét tốt nghiệp: Sinh viên được xét tốt nghiệp khi tích lũy đủ các tí..." | 0.4453 | Có | [Cần tích lũy từ 120-150 tín chỉ tùy ngành] |
| 2 | GPA bao nhiêu thì được làm luậ... | "# Điều kiện xét tốt nghiệp: Sinh viên được xét tốt nghiệp khi tích lũy đủ các tí..." | 0.2081 | Không | [Chunk lấy ra không nói số điểm GPA làm luận văn cụ thể] |
| 3 | Những đối tượng nào được miễn ... | "# Quy định về học phí... Học phí được miễn giảm cho sinh viên thuộc hộ nghèo..." | 0.3419 | Có | [Nhóm SV hộ nghèo, con thương binh, khuyết tật] |
| 4 | Khi nào sinh viên bị cảnh cáo ... | "Sinh viên phải trích dẫn nguồn theo chuẩn APA hoặc IEEE tùy theo quy định kh..." | 0.3446 | Không | [Lấy nhầm đoạn trích dẫn đạo văn, sai ngữ cảnh] |
| 5 | Chuẩn đầu ra ngoại ngữ để tốt ... | "Lãi suất ưu đãi là 0.55%/tháng và sinh viên bắt đầu trả nợ sau khi tốt nghiệp 12..." | 0.2436 | Không | [Lấy nhầm đoạn vay vốn với ưu đãi lãi suất ngân hàng] |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhóm có chia sẻ kinh nghiệm về việc Chunk Size nên dựa theo cấu trúc dataset. Nếu file toàn dữ liệu dạng Markdown List ngắn, việc cắt theo câu sẽ làm nát tài liệu, thay vào đó cắt theo Semantic hoặc Recursive theo `\n` là tốt nhất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Có nhóm đã áp dụng Metadata filtering tự động (trích xuất "phòng ban" từ câu hỏi) để đẩy vào `.search_with_filter` giúp giảm sai số hoàn toàn (nhất là giữa quy chế hệ Đại học và Sau Đại học rất dễ chép nhầm nhau). 

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ chạy tiền xử lý (LLM Data Extraction) để biến các rules/đoạn quy chế dài thành dạng Q&A pairs (Hỏi-Đáp) trước khi băm chunk. Dữ liệu dạng Q&A khi nạp vào Vector Store sẽ dễ tương khớp với truy vấn của người dùng hơn rất nhiều.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 10 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 3 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **83 / 90** |
