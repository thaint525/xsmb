#!/usr/bin/env python3
"""Backtest the "đợi thua k trận rồi mới vào" strategy: only bet the top-1
pick once the model's daily pick has missed k times in a row. Each trigger
buys exactly ONE bet (win or lose, doesn't matter) — then it's back to
watching from a clean slate for the next k-loss streak.

This tests the gambler's-fallacy premise that a pick becomes "due" after a
losing streak. Reuses the (date, pick, hit) sequence from backtest_streak.py
— same top-1 scoring, no future leakage — so only the entry rule changes.

Usage:
    python3 backtest_wait_k.py
    python3 backtest_wait_k.py --ks 1,2,3,5,8,10,15
"""
import argparse
import math

import backtest_streak as streaks
import predict_loto as predictor


def simulate_wait_k_losses(results, k):
    """Bet exactly one day once the underlying top-1 pick has lost k times
    running, then reset the streak to zero regardless of that bet's outcome
    and resume watching from scratch. Returns [(date, pick, hit)] for the
    days actually bet on.
    """
    bets = []
    streak = 0
    for date, pick, hit in results:
        if streak >= k:
            bets.append((date, pick, hit))
            streak = 0
            continue
        streak = 0 if hit else streak + 1
    return bets


def summarize(bets, baseline_rate):
    if not bets:
        return None
    wins = sum(1 for _, _, hit in bets if hit)
    rate = wins / len(bets)
    se = math.sqrt(baseline_rate * (1 - baseline_rate) / len(bets))
    sigma = (rate - baseline_rate) / se if se else 0.0
    return {"n": len(bets), "wins": wins, "rate": rate, "edge": rate - baseline_rate, "sigma": sigma}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV từ crawl_loto.py")
    parser.add_argument("--warmup", type=int, default=365, help="Số kỳ đầu bỏ qua (default: 365)")
    parser.add_argument(
        "--ks", default="1,2,3,5,8,10,15,20",
        help="Danh sách số trận thua liên tiếp cần chờ trước khi vào, cách nhau bằng dấu phẩy",
    )
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    results = streaks.run_strategy(hist, args.warmup)
    baseline_wins = sum(1 for _, _, hit in results if hit)
    baseline_rate = baseline_wins / len(results)

    print(
        f"Baseline (đánh top-1 mỗi ngày, không điều kiện): "
        f"{baseline_wins}/{len(results)} = {baseline_rate:.1%}\n"
    )
    print(f"{'k':>3} {'Số lần vào':>11} {'Thắng':>7} {'Tỉ lệ':>8} {'Chênh vs baseline':>18} {'Sigma':>8}")
    print("-" * 65)
    for k_str in args.ks.split(","):
        k = int(k_str)
        bets = simulate_wait_k_losses(results, k)
        s = summarize(bets, baseline_rate)
        if s is None:
            print(f"{k:>3}   (không có lần nào đủ {k} trận thua liên tiếp để vào)")
            continue
        print(
            f"{k:>3} {s['n']:>11} {s['wins']:>7} {s['rate']:>8.1%} "
            f"{s['edge']:>+17.1%} {s['sigma']:>+7.2f}σ"
        )

    print(
        "\nGhi chú: nếu tỉ lệ thắng sau khi 'chờ đủ k thua' không cao hơn baseline một cách có ý nghĩa "
        "(|sigma| < ~2), thì việc chờ không có tác dụng — đúng với giả định các kỳ độc lập với nhau."
    )


if __name__ == "__main__":
    main()
