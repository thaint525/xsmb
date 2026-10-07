#!/usr/bin/env python3
"""Backtest "đợi thua rồi mới vào" for the top-1 pick, under two entry rules:

    streak  — bet once the top-1 pick has missed k times IN A ROW, then reset
              the counter regardless of that bet's outcome.
    window  — bet on any day where the top-1 pick missed at least k of the
              last WINDOW draws (default 7 — the same "hiệu năng 7 kỳ gần
              nhất" figure run.py prints). k=0 means "always play".

Both rules only look at draws strictly before the day being bet on, so there
is no future leakage. Stake/payout come from backtest_wait_k_top2 so the
money column matches the rest of the project.

Usage:
    python3 backtest_wait_k.py
    python3 backtest_wait_k.py --ks 0,1,2,3,4,5,6,7 --window 7
"""
import argparse
import csv
import math
import os
import sys
from datetime import datetime

import predict_loto as predictor
from backtest_wait_k_top2 import PAYOUT, STAKE

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CACHE_TMPL = "top1_picks_w{warmup}.csv"


def run_strategy_top1(hist, warmup, cache=True):
    """[(date, pick, count)] per draw after warmup.

    `count` is how many times the pick actually landed that day (0 = miss,
    2+ = "nháy"), so payouts can scale with nháy instead of just win/lose.

    Scoring every draw is the slow part of this script, and the picks only
    change when the dataset does, so results are cached keyed by warmup and
    reused whenever the row count still matches.
    """
    path = CACHE_TMPL.format(warmup=warmup)
    expected = hist.n_days - warmup
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [
                (datetime.strptime(r["date"], predictor.DATE_FMT), r["pick"], int(r["count"]))
                for r in csv.DictReader(f)
            ]
        if len(rows) == expected and rows[-1][0] == hist.dates[-1]:
            return rows

    results = []
    for i in range(warmup, hist.n_days):
        target_date = hist.dates[i]
        ranked = sorted(predictor.score(hist, target_date).items(), key=lambda kv: -kv[1]["score"])
        pick = ranked[0][0]
        count = hist.cum[pick][i + 1] - hist.cum[pick][i]
        results.append((target_date, pick, count))

    if cache:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "pick", "count"])
            for date, pick, count in results:
                w.writerow([date.strftime(predictor.DATE_FMT), pick, count])
    return results


def simulate_streak(results, k):
    """Bet one day once the pick has missed k times running, then reset."""
    bets = []
    streak = 0
    for row in results:
        if streak >= k:
            bets.append(row)
            streak = 0
            continue
        _date, _pick, count = row
        streak = 0 if count > 0 else streak + 1
    return bets


def simulate_window(results, k, window):
    """Bet on any day where the pick won exactly k of the previous `window`
    draws — i.e. bucket every draw by its trailing win count, so each k row
    answers "what happened next, historically, after a week that looked like
    this?". Losses are just window - k. Buckets are disjoint, so the n column
    sums to the number of draws scored.
    """
    bets = []
    for idx, row in enumerate(results):
        prior = results[max(0, idx - window) : idx]
        if len(prior) < window:
            continue  # not enough history yet to judge the window
        wins = sum(1 for _d, _p, count in prior if count > 0)
        if wins == k:
            bets.append(row)
    return bets


def summarize(bets, baseline_rate, total_days):
    if not bets:
        return None
    n = len(bets)
    wins = sum(1 for _d, _p, count in bets if count > 0)
    rate = wins / n
    se = math.sqrt(baseline_rate * (1 - baseline_rate) / n)
    sigma = (rate - baseline_rate) / se if se else 0.0
    nets = [PAYOUT * count - STAKE for _d, _p, count in bets]
    return {
        "n": n,
        "per_week": n / total_days * 7,
        "wins": wins,
        "rate": rate,
        "edge": rate - baseline_rate,
        "sigma": sigma,
        "avg_net": sum(nets) / n,
        "total_net": sum(nets),
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'k':>3} {'Số lần vào':>11} {'Lần/tuần':>9} {'Thắng':>7} {'Tỉ lệ':>8} {'Chênh':>8} {'Sigma':>8} {'Lãi TB/lần':>13} {'Tổng lãi':>16}")
    print("-" * 92)
    for k, s in rows:
        if s is None:
            print(f"{k:>3}   (không có lần vào nào)")
            continue
        print(
            f"{k:>3} {s['n']:>11} {s['per_week']:>9.2f} {s['wins']:>7} {s['rate']:>8.1%} "
            f"{s['edge']:>+8.1%} {s['sigma']:>+7.2f}σ {s['avg_net']:>+12,.0f}đ {s['total_net']:>+15,.0f}đ"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV từ crawl_loto.py")
    parser.add_argument("--warmup", type=int, default=365, help="Số kỳ đầu bỏ qua (default: 365)")
    parser.add_argument("--ks", default="0,1,2,3,4,5,6,7", help="Các mức k, cách nhau bằng dấu phẩy")
    parser.add_argument("--window", type=int, default=7, help="Cửa sổ cho luật 'thua k/N kỳ gần nhất' (default: 7)")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    hist = predictor.load_history(args.input)
    results = run_strategy_top1(hist, args.warmup)
    total_days = len(results)
    baseline_wins = sum(1 for _d, _p, count in results if count > 0)
    baseline_rate = baseline_wins / total_days
    baseline_net = sum(PAYOUT * count - STAKE for _d, _p, count in results)

    print(
        f"Dữ liệu: {total_days} kỳ ({results[0][0]:%d/%m/%Y} -> {results[-1][0]:%d/%m/%Y}), "
        f"bỏ {args.warmup} kỳ warmup"
    )
    print(f"Tiền cược: {STAKE:,}đ/lần, mỗi nháy trúng ăn {PAYOUT:,}đ")
    print(
        f"Baseline (đánh top-1 MỖI ngày): {baseline_wins}/{total_days} = {baseline_rate:.1%}, "
        f"lãi TB {baseline_net / total_days:+,.0f}đ/lần, tổng {baseline_net:+,.0f}đ"
    )

    print_table(
        "[A] Luật CHUỖI LIÊN TIẾP — vào sau khi top-1 trượt k kỳ liên tiếp:",
        [(k, summarize(simulate_streak(results, k), baseline_rate, total_days)) for k in ks],
    )
    print_table(
        f"[B] Luật CỬA SỔ {args.window} KỲ — k = số trận THẮNG của top-1 trong {args.window} kỳ gần nhất "
        f"(tức thua {args.window}-k lần); chỉ vào đúng những ngày có trạng thái đó:",
        [(k, summarize(simulate_window(results, k, args.window), baseline_rate, total_days)) for k in ks],
    )

    print(
        "\nGhi chú: k=0 ở cả hai luật = đánh mỗi ngày (chính là baseline). |sigma| < 2 nghĩa là "
        "chênh lệch không phân biệt được với nhiễu ngẫu nhiên."
    )


if __name__ == "__main__":
    main()
