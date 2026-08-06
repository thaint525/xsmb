#!/usr/bin/env python3
"""Score all 100 loto numbers for the next XSMB draw from crawled history.

Reads the *_long.csv produced by crawl_loto.py and combines four standardised
signals into one score, then backtests that score against the base rate so the
output can be read honestly.

Signals (each z-scored across the 100 numbers, then weighted):
    freq_long  long-run hit rate over the whole window
    heat       hit rate over the last HEAT_WINDOW days (momentum)
    overdue    current dry streak / that number's own mean gap ("lô gan")
    weekday    hit rate restricted to the target draw's weekday

All signals are read off prefix sums built once up front, so scoring any day is
O(100) and the backtest stays fast even on ~10k draws.

Examples:
    python3 predict_loto.py --input mb_history_long.csv
    python3 predict_loto.py --input mb_history_long.csv --backtest-days 500
"""
import argparse
import csv
import statistics
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta

DATE_FMT = "%d/%m/%Y"
NUMBERS = [f"{i:02d}" for i in range(100)]
HEAT_WINDOW = 30

# Weights are a judgement call, not a fitted result — see the backtest output.
WEIGHTS = {"freq_long": 0.3, "heat": 0.8, "overdue": 0.6, "weekday": 0.7}

WEEKDAYS_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]


class History:
    """Draw history indexed for O(1) lookup of every signal at any cut-off."""

    def __init__(self, rows):
        """rows: [(datetime, {number: count})] in any order."""
        rows = sorted(rows, key=lambda r: r[0])
        self.dates = [d for d, _ in rows]
        self.n_days = len(rows)

        # cum[n][i] = total hits of n over days [0, i)
        self.cum = {}
        # hit_idx[n] = ascending day indices where n hit at least once
        self.hit_idx = {}
        # wd_days[w] = ascending day indices falling on weekday w
        self.wd_days = defaultdict(list)
        # wd_cum[n][w][j] = total hits of n over the first j weekday-w days
        self.wd_cum = {n: defaultdict(lambda: [0]) for n in NUMBERS}

        for i, (date, _) in enumerate(rows):
            self.wd_days[date.weekday()].append(i)

        for n in NUMBERS:
            cum = [0] * (self.n_days + 1)
            hits = []
            for i, (_, counts) in enumerate(rows):
                c = counts.get(n, 0)
                cum[i + 1] = cum[i] + c
                if c > 0:
                    hits.append(i)
            self.cum[n] = cum
            self.hit_idx[n] = hits

        for w, day_idxs in self.wd_days.items():
            for n in NUMBERS:
                running = 0
                col = [0]
                for i in day_idxs:
                    running += self.cum[n][i + 1] - self.cum[n][i]
                    col.append(running)
                self.wd_cum[n][w] = col

    def cutoff(self, target_date):
        """Number of draws strictly before target_date."""
        return bisect_left(self.dates, target_date)

    def actual(self, day_index):
        """Set of numbers that hit on a given day index."""
        return {n for n in NUMBERS if self.cum[n][day_index + 1] > self.cum[n][day_index]}


def zscore(values):
    """Standardise a {key: value} dict; all-equal input maps to all zeros."""
    nums = list(values.values())
    sd = statistics.pstdev(nums)
    if sd == 0:
        return {k: 0.0 for k in values}
    mean = statistics.fmean(nums)
    return {k: (v - mean) / sd for k, v in values.items()}


def score(hist, target_date):
    """Score every number for target_date using only draws before it.

    Returns {number: {"score": float, **raw signal values}}.
    """
    t = hist.cutoff(target_date)
    if t == 0:
        raise ValueError("no history before target date")

    heat_start = max(0, t - HEAT_WINDOW)
    heat_len = t - heat_start
    w = target_date.weekday()
    wd_days = hist.wd_days.get(w, [])
    wd_before = bisect_left(wd_days, t)

    freq_long, heat, weekday_rate, overdue = {}, {}, {}, {}
    for n in NUMBERS:
        cum = hist.cum[n]
        freq_long[n] = cum[t] / t
        heat[n] = (cum[t] - cum[heat_start]) / heat_len
        weekday_rate[n] = hist.wd_cum[n][w][wd_before] / wd_before if wd_before else 0.0

        # Dry streak vs this number's own mean gap between appearances.
        hits = hist.hit_idx[n]
        j = bisect_left(hits, t)
        if j == 0:
            overdue[n] = float(t)  # never seen: maximally overdue
            continue
        streak = t - 1 - hits[j - 1]
        # Mean of consecutive gaps telescopes to (last - first) / (count - 1).
        mean_gap = (hits[j - 1] - hits[0]) / (j - 1) if j > 1 else float(t)
        overdue[n] = streak / mean_gap if mean_gap else 0.0

    raw = {"freq_long": freq_long, "heat": heat, "overdue": overdue, "weekday": weekday_rate}
    z = {name: zscore(vals) for name, vals in raw.items()}

    return {
        n: {
            "score": sum(WEIGHTS[s] * z[s][n] for s in WEIGHTS),
            **{s: raw[s][n] for s in raw},
        }
        for n in NUMBERS
    }


def backtest(hist, days, top_ks=(1, 5)):
    """Replay the last `days` draws: how often is a top-k pick an actual hit?

    Compared against the base rate — the chance k blind guesses land on a day's
    ~27 drawn numbers — which is what guessing achieves.
    """
    start = max(1, hist.n_days - days)
    tested = range(start, hist.n_days)
    hits = {k: 0 for k in top_ks}
    base = {k: [] for k in top_ks}

    for i in tested:
        ranked = sorted(score(hist, hist.dates[i]).items(), key=lambda kv: -kv[1]["score"])
        actual = hist.actual(i)
        for k in top_ks:
            if any(n in actual for n, _ in ranked[:k]):
                hits[k] += 1
            base[k].append(1 - (1 - len(actual) / 100) ** k)

    n_tested = len(tested)
    return {
        "days": n_tested,
        "results": {
            k: {"hit_rate": hits[k] / n_tested, "base_rate": statistics.fmean(base[k])}
            for k in top_ks
        },
    }


def load_history(path):
    by_date = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_date[row["date"]][row["number"]] = int(row["count"])
    return History([(datetime.strptime(d, DATE_FMT), c) for d, c in by_date.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV from crawl_loto.py")
    parser.add_argument("--target-date", default=None, help="dd/mm/yyyy to predict (default: day after last draw)")
    parser.add_argument("--top", type=int, default=10, help="How many candidates to list (default: 10)")
    parser.add_argument("--backtest-days", type=int, default=500, help="Draws to replay, 0 to skip (default: 500)")
    args = parser.parse_args()

    hist = load_history(args.input)
    last_date = hist.dates[-1]
    target = (
        datetime.strptime(args.target_date, DATE_FMT)
        if args.target_date
        else last_date + timedelta(days=1)
    )

    print(f"Dữ liệu: {hist.n_days} kỳ, {hist.dates[0]:%d/%m/%Y} → {last_date:%d/%m/%Y}")
    print(f"Dự đoán cho: {target:%d/%m/%Y} ({WEEKDAYS_VI[target.weekday()]})\n")

    ranked = sorted(score(hist, target).items(), key=lambda kv: -kv[1]["score"])

    print(f"{'#':<3}{'Lô':<5}{'Điểm':>8}{'TS dài':>9}{'Nóng 30':>9}{'Gan/CK':>8}{'Theo thứ':>10}")
    print("-" * 52)
    for rank, (n, s) in enumerate(ranked[: args.top], 1):
        print(
            f"{rank:<3}{n:<5}{s['score']:>8.2f}{s['freq_long']:>9.3f}"
            f"{s['heat']:>9.3f}{s['overdue']:>8.2f}{s['weekday']:>10.3f}"
        )

    if args.backtest_days:
        bt = backtest(hist, args.backtest_days)
        print(f"\nBacktest {bt['days']} kỳ gần nhất:")
        for k, r in bt["results"].items():
            edge = r["hit_rate"] - r["base_rate"]
            print(
                f"  top-{k}: model trúng {r['hit_rate']:.1%} | đoán bừa {r['base_rate']:.1%} "
                f"| chênh {edge:+.1%}"
            )

    print(f"\n>>> SỐ CHỐT: {ranked[0][0]} <<<")


if __name__ == "__main__":
    main()
