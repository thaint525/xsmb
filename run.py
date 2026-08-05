#!/usr/bin/env python3
"""One-shot: pull the latest XSMB loto draws, update the local dataset, and
print today's/next draw's pick.

Incrementally updates DATA_FILE with any new draws since the last run (or
does a full crawl from scratch on first use), then feeds the combined
history straight into predict_loto. Reuses crawl_loto/predict_loto directly
instead of duplicating their logic.

Usage:
    python3 run.py
"""
import csv
import math
from bisect import bisect_left
from datetime import datetime, timedelta

import requests

import backtest_streak as streaks
import crawl_loto as crawler
import predict_loto as predictor

AVG_HITS_PER_NUMBER = 27 / 100  # 27 lô rơi vào 100 số mỗi kỳ, nếu chia đều

DATA_FILE = "mb_history_long.csv"
FULL_HISTORY_DAYS = 5000  # comfortably covers the site's data (starts 01/01/2014)
BACKTEST_DAYS = 500
RECENT_DAYS = 30


def load_existing(path):
    by_date = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_date.setdefault(row["date"], {})[row["number"]] = int(row["count"])
    except FileNotFoundError:
        pass
    return by_date


def save(path, by_date):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "number", "count"])
        for date_str in sorted(by_date, key=lambda d: datetime.strptime(d, crawler.DATE_FMT)):
            for number, count in sorted(by_date[date_str].items()):
                writer.writerow([date_str, number, count])


def update_dataset(today):
    existing = load_existing(DATA_FILE)
    if existing:
        last_date = max(existing, key=lambda d: datetime.strptime(d, crawler.DATE_FMT))
        last_dt = datetime.strptime(last_date, crawler.DATE_FMT)
        days_needed = (today - last_dt).days + 2  # small overlap, cheap & safe
        print(f"Đã có dữ liệu tới {last_date}, cập nhật thêm {days_needed} ngày gần nhất...")
    else:
        days_needed = FULL_HISTORY_DAYS
        print(f"Chưa có dữ liệu cục bộ, crawl toàn bộ lịch sử (tối đa {days_needed} ngày)...")

    session = requests.Session()
    new_data = crawler.crawl(session, today, days_needed, delay=1.0, log=print)
    existing.update(new_data)
    save(DATA_FILE, existing)
    print(f"Đã lưu {len(existing)} kỳ vào {DATA_FILE}\n")
    return existing


def print_ranking_table(ranked):
    print(f"{'#':<3}{'Lô':<5}{'Điểm':>8}{'TS dài':>9}{'Nóng 30':>9}{'Gan/CK':>8}{'Theo thứ':>10}")
    print("-" * 52)
    for rank, (n, s) in enumerate(ranked[:10], 1):
        print(
            f"{rank:<3}{n:<5}{s['score']:>8.2f}{s['freq_long']:>9.3f}"
            f"{s['heat']:>9.3f}{s['overdue']:>8.2f}{s['weekday']:>10.3f}"
        )


def print_backtest(hist):
    bt = predictor.backtest(hist, BACKTEST_DAYS)
    print(f"\nBacktest {bt['days']} kỳ gần nhất:")
    max_sigma = 0.0
    for k, r in bt["results"].items():
        edge = r["hit_rate"] - r["base_rate"]
        se = math.sqrt(r["base_rate"] * (1 - r["base_rate"]) / bt["days"])
        sigma = edge / se if se else 0.0
        max_sigma = max(max_sigma, abs(sigma))
        print(
            f"  top-{k}: model trúng {r['hit_rate']:.1%} | đoán bừa {r['base_rate']:.1%} "
            f"| chênh {edge:+.1%} ({sigma:+.2f} sigma)"
        )
    verdict = (
        "chênh lệch không phân biệt được với nhiễu ngẫu nhiên (<2 sigma)"
        if max_sigma < 2
        else "chênh lệch có ý nghĩa thống kê (>=2 sigma), nhưng vẫn nên kiểm tra lại trước khi tin"
    )
    print(f"  => {verdict}. Các số về XSMB coi như độc lập — đừng đặt cược thật theo số chốt này.")


def explain_top_pick(hist, target, ranked):
    """Human-readable breakdown of why the #1 pick scored highest."""
    n, s = ranked[0]
    t = hist.cutoff(target)

    by_freq = sorted(ranked, key=lambda kv: -kv[1]["freq_long"])
    freq_rank = next(i for i, (num, _) in enumerate(by_freq, 1) if num == n)

    hits_idx = hist.hit_idx[n]
    j = bisect_left(hits_idx, t)

    lines = [f"\n=== SỐ CHỐT HÔM NAY ({target:%d/%m/%Y}): {n} ===", f"Vì sao {n} đứng đầu:"]
    lines.append(
        f"  - Tần suất dài hạn : {s['freq_long']:.3f} lô/kỳ (top {freq_rank}/100 toàn thời gian)"
    )
    lines.append(
        f"  - Đang nóng (30 kỳ): {s['heat']:.3f} lô/kỳ "
        f"({'trên' if s['heat'] > AVG_HITS_PER_NUMBER else 'dưới'} trung bình {AVG_HITS_PER_NUMBER:.2f})"
    )
    if j:
        last_hit_date = hist.dates[hits_idx[j - 1]]
        streak = t - 1 - hits_idx[j - 1]
        mean_gap = (hits_idx[j - 1] - hits_idx[0]) / (j - 1) if j > 1 else float(t)
        lines.append(
            f"  - Lô gan           : gan {streak} kỳ (về gần nhất {last_hit_date:%d/%m/%Y}), "
            f"chu kỳ TB {mean_gap:.2f} kỳ -> gấp {s['overdue']:.2f}x"
        )
    else:
        lines.append("  - Lô gan           : chưa từng về trong lịch sử đã crawl")
    lines.append(f"  - Theo {predictor.WEEKDAYS_VI[target.weekday()]}      : {s['weekday']:.3f} lô/kỳ")

    top5_next = ", ".join(num for num, _ in ranked[1:5])
    lines.append(f"Top 5 kế tiếp nếu muốn dàn rộng: {top5_next}")
    return "\n".join(lines)


def main():
    today = datetime.now()
    by_date = update_dataset(today)

    hist = predictor.History(
        [(datetime.strptime(d, predictor.DATE_FMT), c) for d, c in by_date.items()]
    )
    target = hist.dates[-1] + timedelta(days=1)

    print(f"Dữ liệu: {hist.n_days} kỳ, {hist.dates[0]:%d/%m/%Y} → {hist.dates[-1]:%d/%m/%Y}")
    print(f"Dự đoán cho: {target:%d/%m/%Y} ({predictor.WEEKDAYS_VI[target.weekday()]})\n")

    ranked = sorted(predictor.score(hist, target).items(), key=lambda kv: -kv[1]["score"])
    print(explain_top_pick(hist, target, ranked))

    print("\nBảng xếp hạng top 10:")
    print_ranking_table(ranked)
    print_backtest(hist)

    recent_results = streaks.run_strategy(hist, warmup=hist.n_days - RECENT_DAYS)
    streaks.print_recent(recent_results, RECENT_DAYS)

    print(f"\n>>> SỐ CHỐT: {ranked[0][0]} <<<")


if __name__ == "__main__":
    main()
