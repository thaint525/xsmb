#!/usr/bin/env python3
"""Backtest the "đánh top-1 mỗi ngày" strategy: every day, bet on the single
highest-scored number; find the longest losing streak.

Reuses History/score from predict_loto.py — each day's pick only uses draws
strictly before it, so there's no future leakage.

Usage:
    python3 backtest_streak.py --input mb_history_long.csv
    python3 backtest_streak.py --input mb_history_long.csv --warmup 365
"""
import argparse

import predict_loto as predictor


def run_strategy(hist, warmup):
    """Return [(date, pick, hit_bool)] for every day after the warmup period."""
    results = []
    for i in range(warmup, hist.n_days):
        target_date = hist.dates[i]
        ranked = sorted(
            predictor.score(hist, target_date).items(), key=lambda kv: -kv[1]["score"]
        )
        pick = ranked[0][0]
        results.append((target_date, pick, pick in hist.actual(i)))
    return results


def longest_streak(results, want_hit):
    """Longest run of consecutive days matching want_hit; returns (length, start_date, end_date)."""
    best_len, best_start, best_end = 0, None, None
    cur_len, cur_start = 0, None
    for date, _pick, hit in results:
        if hit != want_hit:
            cur_len = 0
            continue
        if cur_len == 0:
            cur_start = date
        cur_len += 1
        if cur_len > best_len:
            best_len, best_start, best_end = cur_len, cur_start, date
    return best_len, best_start, best_end


def current_streak(results):
    """Length and start date of the losing streak still active at the last day."""
    length, start = 0, None
    for date, _pick, hit in reversed(results):
        if hit:
            break
        length += 1
        start = date
    return length, start


def print_recent(results, n):
    recent = results[-n:]
    wins = sum(1 for _, _, hit in recent if hit)
    print(f"\nHiệu suất top-1 trong {len(recent)} kỳ gần nhất: thắng {wins}/{len(recent)}")
    for date, pick, hit in recent:
        print(f"  {date:%d/%m/%Y}  lô {pick}  {'Thắng' if hit else 'Thua'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV from crawl_loto.py")
    parser.add_argument(
        "--warmup", type=int, default=365,
        help="Số kỳ đầu bỏ qua trước khi bắt đầu tính điểm, cho model đủ dữ liệu (default: 365)",
    )
    parser.add_argument("--recent", type=int, default=7, help="Số kỳ gần nhất để liệt kê Thắng/Thua (default: 7)")
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    if args.warmup >= hist.n_days:
        raise SystemExit(f"--warmup ({args.warmup}) >= tổng số kỳ có sẵn ({hist.n_days})")

    results = run_strategy(hist, args.warmup)
    wins = sum(1 for _, _, hit in results if hit)

    print(
        f"Chiến lược: đánh top-1 mỗi ngày, {len(results)} kỳ "
        f"({results[0][0]:%d/%m/%Y} -> {results[-1][0]:%d/%m/%Y})"
    )
    print(f"Thắng {wins}/{len(results)} kỳ ({wins / len(results):.1%})")

    lose_len, lose_start, lose_end = longest_streak(results, want_hit=False)
    print(f"\nChuỗi thua dài nhất: {lose_len} kỳ liên tiếp")
    print(f"  Từ {lose_start:%d/%m/%Y} -> {lose_end:%d/%m/%Y}")

    win_len, win_start, win_end = longest_streak(results, want_hit=True)
    print(f"\nChuỗi thắng dài nhất: {win_len} kỳ liên tiếp")
    print(f"  Từ {win_start:%d/%m/%Y} -> {win_end:%d/%m/%Y}")

    cur_len, cur_start = current_streak(results)
    if cur_len:
        print(f"\nChuỗi thua hiện tại (tính đến kỳ cuối cùng): {cur_len} kỳ, bắt đầu từ {cur_start:%d/%m/%Y}")
    else:
        print("\nKỳ gần nhất đang THẮNG (chuỗi thua hiện tại = 0)")

    print_recent(results, args.recent)


if __name__ == "__main__":
    main()
