#!/usr/bin/env python3
"""Backtest a "loại khan, chọn đầu-đuôi lạnh" strategy:

1. Find the 10 coldest numbers ("lô khan") over the last WINDOW days.
2. Collect their tens digits and units digits into two exclude sets.
3. Rank the 10 possible tens digits and 10 possible units digits by their
   own total hit count over the same window, coldest first.
4. Final pick = coldest tens digit that isn't itself in the tens exclude
   set, paired with coldest units digit that isn't itself in the units
   exclude set (each picked independently, then combined).

Reuses History from predict_loto.py — every day only looks at draws
strictly before it, so there's no future leakage.

Usage:
    python3 backtest_khan.py
    python3 backtest_khan.py --window 40 --warmup 365
"""
import argparse
import math

import predict_loto as predictor

NUMBERS = predictor.NUMBERS
# Python's date.weekday(): Monday=0 .. Sunday=6
WEEKDAY_CHOICES = {"T2": 0, "T3": 1, "T4": 2, "T5": 3, "T6": 4}
WEEKDAY_LABELS = {"T2": "Thứ 2", "T3": "Thứ 3", "T4": "Thứ 4", "T5": "Thứ 5", "T6": "Thứ 6"}


def window_freq(hist, t, window):
    """{number: hit count over the `window` days before day index t}."""
    start = max(0, t - window)
    return {n: hist.cum[n][t] - hist.cum[n][start] for n in NUMBERS}


def digit_freq(freq, digit_index):
    """Aggregate per-number freq into {digit 0-9: total hits} by tens(0)/units(1) position."""
    totals = {d: 0 for d in range(10)}
    for n, count in freq.items():
        totals[int(n[digit_index])] += count
    return totals


def pick_for_day(hist, t, window):
    """Return (pick, detail) for day index t; pick is None if a digit position
    has no non-excluded value left (every tens or every units digit excluded).
    """
    freq = window_freq(hist, t, window)

    khan10 = sorted(NUMBERS, key=lambda n: (freq[n], n))[:10]
    exclude_tens = {int(n[0]) for n in khan10}
    exclude_units = {int(n[1]) for n in khan10}

    tens_freq = digit_freq(freq, 0)
    units_freq = digit_freq(freq, 1)
    ordered_tens = sorted(range(10), key=lambda d: (tens_freq[d], d))
    ordered_units = sorted(range(10), key=lambda d: (units_freq[d], d))

    # Coldest tens digit not itself excluded, coldest units digit not itself
    # excluded — picked independently, then combined.
    valid_tens = [d for d in ordered_tens if d not in exclude_tens]
    valid_units = [d for d in ordered_units if d not in exclude_units]
    pick = f"{valid_tens[0]}{valid_units[0]}" if valid_tens and valid_units else None

    detail = {
        "khan10": khan10,
        "exclude_tens": sorted(exclude_tens),
        "exclude_units": sorted(exclude_units),
        "ordered_tens": ordered_tens,
        "ordered_units": ordered_units,
    }
    return pick, detail


def run_strategy(hist, warmup, window):
    """Return [(date, day_index, pick_or_None, hit_or_None)]."""
    results = []
    for i in range(warmup, hist.n_days):
        pick, _detail = pick_for_day(hist, i, window)
        hit = (pick in hist.actual(i)) if pick else None
        results.append((hist.dates[i], i, pick, hit))
    return results


def summarize_played(played, hist):
    """{n, wins, rate, base_rate, sigma} for a set of played (date, i, pick, hit) rows."""
    if not played:
        return None
    wins = sum(1 for _, _, _, h in played if h)
    rate = wins / len(played)
    base_rate = sum(len(hist.actual(i)) / 100 for _, i, _, _ in played) / len(played)
    se = math.sqrt(base_rate * (1 - base_rate) / len(played)) if base_rate else 0.0
    sigma = (rate - base_rate) / se if se else 0.0
    return {"n": len(played), "wins": wins, "rate": rate, "base_rate": base_rate, "sigma": sigma}


def print_weekly_comparison(played, hist, weekday_names):
    print(f"\n--- Chỉ chơi 1 ngày/tuần (Thứ 2 - Thứ 6) ---")
    print(f"{'Ngày':<8}{'Số lần chơi':>13}{'Thắng':>8}{'Tỉ lệ':>9}{'Baseline':>10}{'Chênh':>9}{'Sigma':>9}")
    print("-" * 66)
    for name in weekday_names:
        wd = WEEKDAY_CHOICES[name]
        weekly = [row for row in played if row[0].weekday() == wd]
        s = summarize_played(weekly, hist)
        label = WEEKDAY_LABELS[name]
        if s is None:
            print(f"{label:<8}   (không có kỳ nào chơi được)")
            continue
        edge = s["rate"] - s["base_rate"]
        print(
            f"{label:<8}{s['n']:>13}{s['wins']:>8}{s['rate']:>9.1%}"
            f"{s['base_rate']:>10.1%}{edge:>+9.1%}{s['sigma']:>+8.2f}σ"
        )


def explain_latest(hist, window):
    """Human-readable breakdown of the pick for the day right after the dataset."""
    t = hist.n_days
    pick, detail = pick_for_day(hist, t, window)
    print(f"\n10 lô khan nhất ({window} ngày gần nhất): {', '.join(detail['khan10'])}")
    print(f"Exclude hàng chục: {detail['exclude_tens']}")
    print(f"Exclude hàng đơn vị: {detail['exclude_units']}")
    print(f"Thứ tự đầu số lạnh -> nóng: {detail['ordered_tens']}")
    print(f"Thứ tự đuôi số lạnh -> nóng: {detail['ordered_units']}")
    print(f"=> Số chọn: {pick if pick else '(không có tổ hợp hợp lệ, bỏ qua kỳ này)'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV từ crawl_loto.py")
    parser.add_argument("--warmup", type=int, default=365, help="Số kỳ đầu bỏ qua (default: 365)")
    parser.add_argument("--window", type=int, default=40, help="Số ngày tính lô khan + tần suất đầu/đuôi (default: 40)")
    parser.add_argument(
        "--weekday", choices=list(WEEKDAY_CHOICES), default=None,
        help="Chỉ chơi 1 ngày/tuần (T2..T6). Không truyền thì so sánh cả 5 ngày.",
    )
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    results = run_strategy(hist, args.warmup, args.window)

    played = [(d, i, p, h) for d, i, p, h in results if p is not None]
    skipped = len(results) - len(played)
    wins = sum(1 for _, _, _, h in played if h)

    print(
        f"Chiến lược: loại khan + chọn đầu/đuôi lạnh nhất, cửa sổ {args.window} ngày, "
        f"{len(results)} kỳ ({results[0][0]:%d/%m/%Y} -> {results[-1][0]:%d/%m/%Y})"
    )
    if skipped:
        print(f"Bỏ qua {skipped} kỳ không có tổ hợp hợp lệ (toàn bộ đầu/đuôi bị loại)")

    if played:
        rate = wins / len(played)
        base_rate = sum(len(hist.actual(i)) / 100 for _, i, _, _ in played) / len(played)
        se = math.sqrt(base_rate * (1 - base_rate) / len(played))
        sigma = (rate - base_rate) / se if se else 0.0
        print(f"Thắng {wins}/{len(played)} kỳ chơi được ({rate:.1%})")
        print(f"Baseline (đoán bừa 1 số, đo trên đúng {len(played)} kỳ đã chơi): {base_rate:.1%}")
        print(f"Chênh: {rate - base_rate:+.1%} ({sigma:+.2f}σ)")
    else:
        print("Không có kỳ nào chơi được với cửa sổ này.")

    weekday_names = [args.weekday] if args.weekday else list(WEEKDAY_CHOICES)
    print_weekly_comparison(played, hist, weekday_names)

    explain_latest(hist, args.window)


if __name__ == "__main__":
    main()
