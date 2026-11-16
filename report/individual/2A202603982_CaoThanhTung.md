# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: *Principles of Marketing* (Chapters 1 & 2), xuất bản bởi OpenStax / Rice University (©2026), giấy phép **CC BY-NC-SA 4.0** (hợp lệ cho mục đích học tập phi thương mại, không chứa dữ liệu mật/cá nhân). Tổng `104` trang PDF vật lý, trong đó **`75` trang** nội dung hữu ích (`25–58` thuộc Chapter 1 và `61–101` thuộc Chapter 2). Sử dụng **`351` chunks** từ chiến lược **C** của Day 6 (`by_paragraph`, `max_size=900`, kèm metadata `chapter` và `content_type`), xuất qua [export_day6.py](../../src/export_day6.py) và chuẩn hóa qua [convert_day6.py](../../starter/convert_day6.py) tại `starter/data/converted/chunks.json`.
- Vector store, embedding model, phiên bản và số chiều: Vector store **PostgreSQL 16.16 + pgvector 0.8.7** (Docker Compose Day 6 `day6_student_starter-postgres-1`, cổng `127.0.0.1:55433`, exact cosine distance `<=>`). Embedding model `BAAI/bge-small-en-v1.5` (`sentence-transformers==6.3.0`, **384 chiều**, chuẩn hóa L2; query dùng prefix `"Represent this sentence for searching relevant passages: "`).
- Agent model, cách chạy và phân chia dev/test: Model `deepseek-flash` (DeepSeek Chat Completions API với tool calling, `temperature=0.0`, `max_tokens=1500`). Cách chạy: khởi động container Day 6 (`docker compose -p day6_student_starter up -d --wait`), chạy `src/export_day6.py` + `starter/convert_day6.py`, chạy `src/eval_retrieval.py --split dev|test`, chạy `src/run_agent_eval.py --split dev|test` và tổng hợp bằng `starter/agent_metrics.py`. Bộ benchmark ([agent_cases.json](../../starter/data/converted/agent_cases.json)) gồm **24 câu**: **`dev` (17 câu)** gồm `15` câu có đáp án (`q01`–`q15`, có 3 câu filter `q13`–`q15`) + `2` câu thiếu bằng chứng (`q21`, `q22`); **`test` (7 câu)** gồm `5` câu có đáp án (`q16`–`q20`, có 1 câu filter `q20`) + `2` câu thiếu bằng chứng (`q23`, `q24`).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: Câu có đáp án giữ `gold` (`source_id`, `page`, `evidence`) từ Day 6 (chuẩn hóa `\n\n` ở `q07`, `q09`, `q13` để khớp nguyên văn chunk). Câu thiếu evidence chọn các chủ đề gần ngữ cảnh marketing (*NPS*, *Maslow*, *price skimming vs. penetration pricing*, *Porter's Five Forces*) và đã kiểm chứng vắng mặt hoàn toàn trong 351 chunks Chapters 1–2. Tiêu chí chấm: Retrieval dùng `Hit@1`, `Evidence Hit@3`, `Context Evidence Rate` (chỉ tính câu có đáp án); Context `test` chấm `0/1/2`; Agent chấm `Tool use`, `Grounded answer` (`0/1/2` và trung bình), `Citation hợp lệ` (có trong context và hỗ trợ câu trả lời), `Abstention` đúng và `Từ chối nhầm`. Tiêu chí chốt trên `dev`: ưu tiên `Grounded/Answered` (`0` từ chối nhầm), `Abstention` `2/2`, `Context Evidence Rate` cao, `0` lỗi hệ thống, sau đó xét đánh đổi token/latency.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF (`starter/baseline.py`), `top-k=3`, không filter, không rerank, `budget=800` (`fifo`) | Mốc lexical offline mặc định | dev | 8/15 (53.3%) | 11/15 (73.3%) | 3/15 (20.0%) | `1.65 ms` median / `\$0.00` |
| Phương pháp 2 | BM25 (`k1=1.5, b=0.75`), `top-k=3`, có filter, không rerank, `budget=2400` | Tăng `budget=2400` tránh rơi chunk `max_size=900`; BM25 chuẩn hóa độ dài | dev | 9/15 (60.0%) | 11/15 (73.3%) | 11/15 (73.3%) | `1.49 ms` median / `\$0.00` |
| Phương pháp 3 | Dense (`pgvector` + `bge-small-en-v1.5`), `top-k=3`, SQL filter, không rerank, `budget=2400` | Embedding ngữ nghĩa xử lý tốt câu hỏi diễn đạt lại | dev | 11/15 (73.3%) | 13/15 (86.7%) | 12/15 (80.0%) | `49.93 ms` median / `\$0.00` |
| Phương pháp 4 (Hybrid) | Hybrid RRF (`Dense` + `BM25`, `candidate_k=20, rrf_k=60`), `top-k=3`, có filter, `budget=2400` | Kết hợp khớp từ khóa chính xác (BM25) và ngữ nghĩa (Dense) | dev | 11/15 (73.3%) | 13/15 (86.7%) | 13/15 (86.7%) | `51.60 ms` median / `\$0.00` |
| Agent Baseline | Agent `deepseek-flash` + `Dense`, `top-k=3`, `max_searches=3`, `budget=2400` (`fifo`) | Mốc agentic dùng Dense và đóng gói context `fifo` | dev | — | — | 11/15 (73.3%) *(13/15 answered, 2 lỗi)* | `6,839.7 ms` median / `55,316` tokens (`155.0s`) |
| Cải thiện 1 | Agent + `Hybrid RRF`, `top-k=3`, `budget=2400`, **`context_mode=ranked`** + retry | Thay `fifo` bằng `ranked` để chunk điểm cao ở lượt 2 không bị lượt 1 chặn | dev | — | — | 15/15 (100%) *(13/15 answered, 2 false abstain)* | `6,212.0 ms` median / `74,271` tokens (`128.9s`) |
| Cải thiện 2 | Agent + `Hybrid RRF`, **`top-k=5`, `budget=3600`, `context_mode=ranked`** + retry | Tăng `top-k=5` & `budget=3600` giảm rơi chunk phụ trong câu nhiều ý | dev | — | — | 14/15 (93.3%) *(15/15 answered, 0 false abstain)* | `7,316.7 ms` median / `93,122` tokens (`335.0s`) |
| Cấu hình đã chọn | Agent + `Hybrid RRF`, `top-k=5`, `budget=3600`, `context_mode=ranked` | Chạy cấu hình đã chốt trên `test` (retrieval thuần đạt `Hit@1=3/5`, `Hit@3=5/5`) | test | — | — | 4/5 (80.0%) | `8,122.2 ms` median / `44,245` tokens (`87.6s`) |
| Baseline | Agent `Dense`, `top-k=3`, `budget=2400`, `fifo` | Đối chứng trên `test` (retrieval thuần đạt `Hit@1=3/5`, `Hit@3=5/5`) | test | — | — | 4/5 (80.0%) | `5,711.9 ms` median / `32,735` tokens (`55.0s`) |

Hai lỗi `dev`, nguyên nhân, cách thử và lý do chọn cấu hình:
1. Lỗi `q09` (*gap analysis*) và `q10` (*SWOT Opportunities*) ở Agent Baseline: Agent tìm được chunk gold (`p65:C:0`, `p74:C:1`) ở lượt search thứ 2, nhưng `fifo` + `budget=2400` giữ 3 chunk lượt 1 làm đầy budget nên đẩy chunk gold khỏi `final_ctx`. Chuyển sang `hybrid_rrf` + `ranked` ở Cải thiện 1 & 2 đã khắc phục hoàn toàn cả hai câu.
2. Lỗi mạng `WinError 10060` (`q07`, `q14`) ở Agent Baseline và từ chối nhầm (`q05`, `q07`) ở Cải thiện 1: Đã thêm retry/backoff (giảm lỗi về `0`). Ở Cải thiện 1, `budget=2400` giữ chưa đủ các đoạn liệt kê nên model chọn `ABSTAIN:` dù `ctx_gold=True`. Chốt **Cải thiện 2** vì đạt `15/15` câu có đáp án được trả lời (`0` từ chối nhầm), `2/2` abstention đúng, `0` lỗi hệ thống trên `dev`.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: Dùng tool `search_chunks(query, chapter?, content_type?)` gọi tới `Dense` hoặc `Hybrid RRF`, giới hạn `max_searches = 3` lượt gọi; dừng sớm khi model trả lời hoặc ép trả lời/từ chối (`tools=None`) khi chạm giới hạn 3 lượt. Chọn context bằng deduplication theo `chunk_id`, sắp xếp theo điểm retrieval giảm dần (`ranked`) hoặc thứ tự xuất hiện (`fifo`), đóng gói vừa `context_budget`. Xử lý lỗi bằng retry tối đa 3 lần (exponential backoff) cho HTTP 429/5xx và lỗi mạng; nếu thất bại vẫn ghi `status="error"` riêng.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent Baseline / `test` | 7/7 (100%) | 1 lượt: 1, 2 lượt: 1, 3 lượt: 5 | `0 / 1 / 4` — **1.80 / 2** | 7/7 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |
| C2 đã chọn / `test` | 7/7 (100%) | 1 lượt: 1, 2 lượt: 3, 3 lượt: 3 | `0 / 0 / 5` — **2.00 / 2** | 12/12 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |

*(Chấm context 5 câu test có đáp án: cả Baseline và C2 đều đạt `0 / 1 / 4` — trung bình `1.80 / 2`, cùng bị mức `1` ở `q18`.)*

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 2 lượt search; `ctx_gold=True` (`p49:C:0`, `p49:C:4`) | Nêu đúng nguyên tắc Lapaas Digital: *"the privacy of the end user is most important"* `[p49:C:0]`, bổ sung `[p49:C:4]`. | `Context=2`, `Answer=2`, `Cites=2/2` — Khớp nguyên văn trang 49. |
| `q17` | Có đáp án | 3 lượt search; `ctx_gold=True` (`p78:C:1`, `p78:C:2`, `p82:C:3`, `p78:C:0`) | Giải thích đầy đủ mục đích marketing plan khi không gọi vốn (thiết lập/đạt mục tiêu, phân công hành động). | `Context=2`, `Answer=2`, `Cites=4/4` — Đầy đủ, dẫn chứng trang 78, 82. |
| `q18` | Có đáp án | 2 lượt search; `ctx_gold=False` (`p72:C:1`, `p71:C:1`, `p71:C:2` trong `final_ctx`; `p72:C:2` rơi khỏi `final_ctx`) | Giải thích đúng iPhone là star (thị phần cao, tăng trưởng) và iPad/MacBook là cash cow (thị phần cao, thị trường ít tăng trưởng). | `Context=1` (thiếu `p72:C:2` trong `final_ctx`), `Answer=2`, `Cites=3/3` — Trả lời đúng đủ nhưng `p72:C:2` bị tràn budget khỏi `final_ctx`. |
| `q19` | Có đáp án | 2 lượt search; `ctx_gold=True` (`p61:C:1`, `p61:C:2`) | Nêu đúng Frito-Lay crowdsourcing qua *"Do Us a Flavor"* và bổ sung *"Turn Up the Flavor"*. | `Context=2`, `Answer=2`, `Cites=2/2` — Chính xác, đầy đủ. |
| `q20` | Có đáp án (filter) | 1 lượt search với filter `chapter=2, content_type=case_study`; `ctx_gold=True` (`p100:C:0`) | Liệt kê đủ 5 Blue Zones: Okinawa, Sardinia, Nicoya, Ikaria, Loma Linda `[p100:C:0]`. | `Context=2`, `Answer=2`, `Cites=1/1` — Chính xác ngay lượt đầu. |
| `q23` | Thiếu bằng chứng | 3 lượt search (pricing, market penetration) | `ABSTAIN:` Nêu rõ Chapters 1–2 không định nghĩa *price skimming* hay *penetration pricing*, phân biệt với *market penetration strategy*. | `Abstention=True`, `Cites=0/0` — Từ chối chuẩn, không suy diễn. |
| `q24` | Thiếu bằng chứng | 3 lượt search (Porter, five forces, competition) | `ABSTAIN:` Nêu rõ Chapter 2 chỉ có SWOT, BCG, SBU và không đề cập Porter's Five Forces. | `Abstention=True`, `Cites=0/0` — Từ chối chuẩn, không suy diễn. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: Trace & summary lưu tại `starter/results/retrieval-dev/`, `starter/results/retrieval-test/`, `starter/results/agent-dev-baseline/`, `starter/results/agent-dev-c1/`, `starter/results/agent-dev-c2/`, `starter/results/agent-test-baseline/`, `starter/results/agent-test-c2/`. Nhãn audit `test` tại `reviews.json` và tổng hợp bằng `python starter/agent_metrics.py ... --split test` tại `agent_metrics_summary.json`. Thời gian/token trên `test`: Baseline `55.0s` (`32,735` tokens), C2 `87.6s` (`44,245` tokens).
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn: Trên `dev`, C2 giảm lỗi mạng `2 → 0`, tăng `answerable_answered` `13/15 → 15/15` (`0` từ chối nhầm) và `Context Evidence Hit` `11/15 → 14/15`. Trên `test`, C2 nâng `Grounded answer` trung bình từ `1.80/2` (Baseline thiếu ý ở `q18`) lên `2.00/2`. Đánh đổi: C2 tốn thêm ~35% token và latency cao hơn (~`8.1s` vs `5.7s` trên `test`). Giới hạn: tập `test` nhỏ (`7` câu); `final_ctx` đóng gói lại từ `all_candidates` vẫn có thể làm rơi một chunk phụ như `q18` dù LLM đã đọc chunk đó trong lượt tool call trước; nhãn audit do tác giả tự chấm.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện phần thử thách nâng cao.

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: Không áp dụng.
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý: Không áp dụng.
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev: Không áp dụng.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | — | — | Không áp dụng |
| Ảnh, sơ đồ | — | — | Không áp dụng |
| Biểu đồ | — | — | Không áp dụng |
| Multi-hop | — | — | Không áp dụng |
| Text | — | — | Không áp dụng |
| Thiếu bằng chứng | — | — | Không áp dụng |
| **Toàn bộ** | — | — | Không áp dụng |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống: Không áp dụng.
- Giới hạn, cách tái lập và đường dẫn benchmark: Không áp dụng.
