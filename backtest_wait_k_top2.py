#!/usr/bin/env python3
"""Backtest the "đợi thua k ngày rồi chơi cả top-1 lẫn top-2 của ngày đó"
strategy: watch the model's top-1 pick until it has lost k times in a row,
then bet BOTH the top-1 and top-2 numbers for that single next day. Whether
either (or both) hits, the entry resets and it's back to watching from a
clean slate for the next k-loss streak.

The losing streak that triggers entry is still defined on the top-1 pick
alone (same convention as backtest_wait_k.py) -- only the bet itself now
covers 2 numbers instead of 1.

Reuses History/score from predict_loto.py -- every day only uses draws
strictly before it, so there's no future leakage.

Usage:
    python3 backtest_wait_k_top2.py
    python3 backtest_wait_k_top2.py --ks 1,2,3,4,5
"""
import argparse
import math
import sys

import predict_loto as predictor

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

STAKE = 200_000
PAYOUT = 733_000


def run_strategy_top2(hist, warmup):
    """[(date, day_index, pick1, pick2, count1, count2)] for every day after
    warmup. count = number of times that pick actually hit that day (0 if it
    missed, 2+ if it "nháy" - hit more than once), so payout can be scaled by
    nháy count, not just win/lose.
    """
    results = []
    for i in range(warmup, hist.n_days):
        target_date = hist.dates[i]
        ranked = sorted(predictor.score(hist, target_date).items(), key=lambda kv: -kv[1]["score"])
        pick1, pick2 = ranked[0][0], ranked[1][0]
        count1 = hist.cum[pick1][i + 1] - hist.cum[pick1][i]
        count2 = hist.cum[pick2][i + 1] - hist.cum[pick2][i]
        results.append((target_date, i, pick1, pick2, count1, count2))
    return results


def simulate_wait_k_top2(results, k, streak_on="top1"):
    """Bet top-1 + top-2 on the day right after the losing streak has reached
    k, then reset regardless of outcome. `streak_on` picks what counts as a
    "loss" for the wait counter: "top1" (only top-1 misses, ignores top-2) or
    "both" (streak only grows when BOTH top-1 and top-2 miss that day; either
    one hitting resets it). Returns the entered rows.
    """
    bets = []
    streak = 0
    for row in results:
        _date, _i, _p1, _p2, count1, count2 = row
        if streak >= k:
            bets.append(row)
            streak = 0
            continue
        is_loss = (count1 == 0) if streak_on == "top1" else (count1 == 0 and count2 == 0)
        streak = streak + 1 if is_loss else 0
    return bets


def summarize(bets, hist, baseline_rate):
    if not bets:
        return None
    n = len(bets)
    hit1_wins = sum(1 for *_, c1, _c2 in bets if c1 > 0)
    hit2_wins = sum(1 for *_, _c1, c2 in bets if c2 > 0)
    either_wins = sum(1 for *_, c1, c2 in bets if c1 > 0 or c2 > 0)
    both_wins = sum(1 for *_, c1, c2 in bets if c1 > 0 and c2 > 0)

    either_rate = either_wins / n
    baseline_either = 1 - (1 - baseline_rate) ** 2
    se = math.sqrt(baseline_either * (1 - baseline_either) / n) if baseline_either else 0.0
    sigma = (either_rate - baseline_either) / se if se else 0.0

    nets = []
    for *_, c1, c2 in bets:
        payout = PAYOUT * (c1 + c2)  # ăn tuyến tính theo số nháy
        nets.append(payout - 2 * STAKE)
    avg_net = sum(nets) / n

    return {
        "n": n,
        "hit1_rate": hit1_wins / n,
        "hit2_rate": hit2_wins / n,
        "either_rate": either_rate,
        "both_rate": both_wins / n,
        "baseline_either": baseline_either,
        "sigma": sigma,
        "avg_net": avg_net,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV từ crawl_loto.py")
    parser.add_argument("--warmup", type=int, default=365, help="Số kỳ đầu bỏ qua (default: 365)")
    parser.add_argument(
        "--ks", default="0,1,2,3,4,5,8,10,15,20",
        help="Danh sách số trận thua liên tiếp cần chờ trước khi vào, cách nhau bằng dấu phẩy",
    )
    parser.add_argument(
        "--streak-on", choices=["top1", "both"], default="top1",
        help="Điều kiện tính 1 kỳ là 'thua' cho chuỗi chờ: 'top1' chỉ xét top-1, "
        "'both' chỉ tính thua khi CẢ top-1 và top-2 cùng trật (default: top1)",
    )
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    results = run_strategy_top2(hist, args.warmup)
    baseline_wins = sum(1 for *_, c1, _c2 in results if c1 > 0)
    baseline_rate = baseline_wins / len(results)
    n_days_span = (results[-1][0] - results[0][0]).days
    months_span = n_days_span / 30.44

    print(f"Baseline top-1 riêng lẻ: {baseline_wins}/{len(results)} = {baseline_rate:.1%}")
    print(f"Tiền cược: {STAKE:,}đ/số/ngày x 2 số = {2*STAKE:,}đ/lần, ăn {PAYOUT:,}đ cho mỗi số trúng.")
    print(f"Giai đoạn test: {results[0][0]:%d/%m/%Y} -> {results[-1][0]:%d/%m/%Y} (~{months_span:.1f} tháng)\n")

    header = (
        f"{'k':>3} {'Số lần vào':>11} {'Lần/tháng':>10} {'Tỉ lệ top1':>11} {'Tỉ lệ top2':>11} "
        f"{'Tỉ lệ>=1 thắng':>15} {'Cả 2 trúng':>11} {'Baseline lý thuyết':>19} {'Sigma':>8} {'Lãi TB/lần':>13}"
    )
    print(header)
    print("-" * len(header))

    best = None
    for k_str in args.ks.split(","):
        k = int(k_str)
        bets = simulate_wait_k_top2(results, k, streak_on=args.streak_on)
        s = summarize(bets, hist, baseline_rate)
        if s is None:
            print(f"{k:>3}   (không có lần nào đủ {k} trận thua liên tiếp để vào)")
            continue
        per_month = s["n"] / months_span
        print(
            f"{k:>3} {s['n']:>11} {per_month:>10.2f} {s['hit1_rate']:>11.1%} {s['hit2_rate']:>11.1%} "
            f"{s['either_rate']:>15.1%} {s['both_rate']:>11.1%} {s['baseline_either']:>19.1%} "
            f"{s['sigma']:>+7.2f}σ {s['avg_net']:>+12,.0f}đ"
        )
        if best is None or s["avg_net"] > best[1]["avg_net"]:
            best = (k, s)

    if best:
        k, s = best
        print(
            f"\n=> k={k} có lãi kỳ vọng TB cao nhất mỗi lần vào: {s['avg_net']:+,.0f}đ "
            f"(tỉ lệ >=1 thắng: {s['either_rate']:.1%} vs baseline lý thuyết {s['baseline_either']:.1%}, {s['sigma']:+.2f}σ)."
        )

    print(
        "\nGhi chú: 'Baseline lý thuyết' = xác suất ít nhất 1 trong 2 số ngẫu nhiên trúng cùng ngày, "
        f"1-(1-p)^2 với p={baseline_rate:.1%} là tỉ lệ nền top-1. Nếu sigma không vượt ~2 và lãi TB không dương rõ ràng "
        "thì đánh thêm top-2 cũng không có edge thật -- chỉ đơn giản là tăng gấp đôi tiền cược mỗi lần."
    )


if __name__ == "__main__":
    main()
