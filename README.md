# lotte — thống kê & thử dự đoán loto XSMB

Crawl lịch sử quay số XSMB (xoso.com.vn), thử nhiều chiến lược/model để dự đoán
số về, và backtest nghiêm túc từng chiến lược để trả lời câu hỏi: **có cách
nào dự đoán tốt hơn ngẫu nhiên không?**

Kết luận sau tất cả thử nghiệm: **không.** Xem [Kết luận](#kết-luận) cuối file.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dùng nhanh — 2 lệnh chính

```bash
python3 run.py       # crawl mới nhất -> chấm điểm heuristic -> in số chốt + backtest
python3 run_ml.py    # crawl mới nhất -> retrain 2 model ML -> predict + đánh giá
```

Cả hai đều tự cập nhật `mb_history_long.csv` (chỉ fetch phần dữ liệu còn thiếu),
không cần chạy script crawl riêng trước.

## Các file

### Dữ liệu

| File | Nội dung |
|---|---|
| `mb_history_long.csv` | Dataset chính: `date,number,count` — mỗi kỳ x mỗi số 00-99. Nguồn cho mọi script khác. |
| `ml_panel_w365.csv` | Cache feature panel cho ML (`train_ml.py`/`run_ml.py` tự rebuild khi data đổi). |

Site chỉ có dữ liệu XSMB từ **01/01/2014** — không crawl được xa hơn dù xin
`--days` bao nhiêu.

### Crawl

- **`crawl_loto.py`** — gọi thẳng endpoint AJAX nội bộ của xoso.com.vn
  (`/ThongKe/AjaxTanSuatLo`), tự chia nhỏ request nếu `--days` > 1000 (giới hạn
  mỗi lần gọi của site).
  ```bash
  python3 crawl_loto.py --days 2000 --output mb_2000d
  ```

### Model heuristic (z-score 4 tín hiệu)

- **`predict_loto.py`** — chấm điểm mỗi số 00-99 bằng 4 tín hiệu z-score
  (tần suất dài hạn, đang nóng 30 ngày, lô gan, tần suất theo thứ), có backtest
  tích hợp.
  ```bash
  python3 predict_loto.py --input mb_history_long.csv --target-date 05/08/2026
  ```
- **`run.py`** — bản end-to-end: tự crawl cập nhật rồi chạy `predict_loto.py`,
  kèm giải thích vì sao số top-1 đứng đầu, backtest 500 kỳ, và hiệu suất
  top-1 trong 30 kỳ gần nhất.

### Backtest các chiến lược "dân gian"

- **`backtest_streak.py`** — backtest "đánh top-1 mỗi ngày": tỉ lệ thắng tổng,
  chuỗi thắng/thua dài nhất, chuỗi thua hiện tại, hiệu suất N kỳ gần nhất.
- **`backtest_wait_k.py`** — test giả thuyết "đợi thua k trận rồi mới vào"
  (gambler's fallacy) với nhiều giá trị k.
- **`backtest_khan.py`** — chiến lược "loại lô khan + chọn đầu/đuôi lạnh
  nhất", có thể giới hạn chỉ chơi 1 ngày/tuần (`--weekday T2`..`T6`).

### Model Machine Learning thật

- **`train_ml.py`** — build feature panel (freq dài hạn, heat 7/14/30/60 ngày,
  độ gan, tần suất theo thứ, hàng chục/đơn vị, thứ trong tuần), train
  Logistic Regression + Histogram Gradient Boosting, tách train/test theo
  thời gian (không shuffle), đánh giá AUC-ROC / log-loss / Brier / calibration
  / backtest top-1.
  ```bash
  python3 train_ml.py --test-days 500 --rebuild-panel
  ```
- **`run_ml.py`** — bản end-to-end: crawl cập nhật → rebuild panel → train +
  đánh giá trên 500 kỳ gần nhất → refit trên 100% dữ liệu → predict kỳ tiếp
  theo, so sánh xem 2 model có đồng thuận hay không.

## Kết luận

Đã thử: heuristic 4-tín-hiệu, "đợi thua k trận", loại lô khan theo đầu/đuôi,
giới hạn chơi 1 ngày/tuần, Logistic Regression, Gradient Boosting — **không
chiến lược nào vượt ngưỡng ý nghĩa thống kê (±2σ)** so với baseline ngẫu nhiên
(~23.8%), và AUC-ROC của cả hai model ML đều ~0.50 (ngang tung đồng xu).

Đây đúng với bản chất của xổ số: **các kỳ quay độc lập nhau**, không có cấu
trúc nào trong lịch sử để khai thác. Toàn bộ công cụ trong repo này nên được
xem như dự án học/thử nghiệm thống kê, **không phải công cụ để đặt cược thật**.

## Dọn dẹp (tùy chọn)

`mb_1000d_*.csv`, `mb_10000d_*.csv`, `mb_history_summary.csv` là output từ các
lần crawl thử nghiệm ban đầu, không còn được script nào dùng — có thể xoá an
toàn nếu muốn gọn thư mục.
