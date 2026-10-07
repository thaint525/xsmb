#!/usr/bin/env python3
"""Grid-search predict_loto's signal weights, without fooling ourselves.

Searching hundreds of weight combos on the full history would always find
one that looks great by luck. So the history is split chronologically:
weights are chosen on the TRAIN period only, then judged on a TEST period
they never saw. Only the test number says anything about the future.

Objective: mean nháy count of the daily top-1 pick (EV per bet is linear in
it: count * PAYOUT - STAKE). Break-even at 200k/733k is 0.2729.

Speed: the four z-scored signals are computed once per draw into an array
(draws x signals x numbers); each weight combo is then just a weighted sum
and an argmax, so the whole grid runs in seconds.

Usage:
    python3 search_weights.py
    python3 search_weights.py --test-frac 0.3 --step 0.25
"""
import argparse
import itertools
import math
import sys

import numpy as np

import predict_loto as predictor
from backtest_wait_k_top2 import PAYOUT, STAKE

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SIGNALS = list(predictor.WEIGHTS)  # freq_long, heat, overdue, weekday
NUMBERS = predictor.NUMBERS


def build_tensor(hist, warmup):
    """Z-scored signals (draws, signals, 100) and nháy counts (draws, 100)."""
    n_draws = hist.n_days - warmup
    z = np.zeros((n_draws, len(SIGNALS), len(NUMBERS)))
    counts = np.zeros((n_draws, len(NUMBERS)))
    for row, i in enumerate(range(warmup, hist.n_days)):
        scored = predictor.score(hist, hist.dates[i])
        for s_idx, sig in enumerate(SIGNALS):
            raw = np.array([scored[n][sig] for n in NUMBERS])
            sd = raw.std()
            z[row, s_idx] = (raw - raw.mean()) / sd if sd else 0.0
        counts[row] = [hist.cum[n][i + 1] - hist.cum[n][i] for n in NUMBERS]
    return z, counts


def evaluate(weights, z, counts):
    """Mean nháy of the top-1 pick, and its standard error."""
    scores = np.tensordot(z, weights, axes=([1], [0]))  # (draws, 100)
    picks = scores.argmax(axis=1)
    hit = counts[np.arange(len(picks)), picks]
    return hit.mean(), hit.std(ddof=1) / math.sqrt(len(hit))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="mb_history_long.csv")
    parser.add_argument("--warmup", type=int, default=365)
    parser.add_argument("--test-frac", type=float, default=0.3, help="Phần cuối lịch sử giữ làm test (default 0.3)")
    parser.add_argument("--step", type=float, default=0.25, help="Bước lưới trọng số trong [0,1] (default 0.25)")
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    print("Tính trước z-score 4 tín hiệu cho mọi kỳ...")
    z, counts = build_tensor(hist, args.warmup)
    dates = hist.dates[args.warmup :]
    cut = int(len(dates) * (1 - args.test_frac))
    z_tr, c_tr, z_te, c_te = z[:cut], counts[:cut], z[cut:], counts[cut:]
    be = STAKE / PAYOUT
    print(f"Train: {cut} kỳ ({dates[0]:%d/%m/%Y} -> {dates[cut-1]:%d/%m/%Y})")
    print(f"Test : {len(dates)-cut} kỳ ({dates[cut]:%d/%m/%Y} -> {dates[-1]:%d/%m/%Y})")
    print(f"Hoà vốn: {be:.4f} nháy/kỳ | chọn bừa: 0.2700\n")

    grid = np.arange(0, 1 + 1e-9, args.step)
    results = []
    for w in itertools.product(grid, repeat=len(SIGNALS)):
        if not any(w):
            continue
        w = np.array(w)
        m_tr, _ = evaluate(w, z_tr, c_tr)
        m_te, se_te = evaluate(w, z_te, c_te)
        results.append((m_tr, m_te, se_te, w))

    results.sort(key=lambda r: -r[0])
    best_tr, best_te, best_se, best_w = results[0]
    fmt = lambda w: ", ".join(f"{s}={v:.2f}" for s, v in zip(SIGNALS, w))

    print(f"Đã thử {len(results)} tổ hợp. Top 5 theo TRAIN (kèm điểm TEST thật):")
    for m_tr, m_te, se_te, w in results[:5]:
        print(f"  [{fmt(w)}]  train {m_tr:.4f} | test {m_te:.4f} ({(m_te-be)/se_te:+.2f}σ vs hoà vốn)")

    print("\nSo sánh trên TEST:")
    for label, w in [
        ("Tối ưu từ train", best_w),
        ("Trọng số cũ (0.3/0.8/0.6/0.7)", np.array([0.3, 0.8, 0.6, 0.7])),
        ("Trọng số hiện tại", np.array([predictor.WEIGHTS[s] for s in SIGNALS])),
        ("Chỉ heat", np.array([0, 1, 0, 0])),
    ]:
        m, se = evaluate(w, z_te, c_te)
        print(
            f"  {label:<32} nháy {m:.4f}  z {(m-be)/se:+.2f}σ  lãi TB {m*PAYOUT-STAKE:+,.0f}đ/lần"
        )

    te_scores = np.array([r[1] for r in results])
    rank = (te_scores > best_te).sum() + 1
    print(
        f"\nKiểm tra overfit: tổ hợp tốt nhất trên train đứng hạng {rank}/{len(results)} trên test. "
        f"Test của mọi tổ hợp dao động {te_scores.min():.4f} .. {te_scores.max():.4f} "
        f"(trung vị {np.median(te_scores):.4f})."
    )
    corr = np.corrcoef([r[0] for r in results], te_scores)[0, 1]
    print(f"Tương quan điểm train vs test giữa các tổ hợp: {corr:+.2f} "
          "(gần 0 = thứ hạng trên train không dự báo được test -> chỉ là nhiễu)")


if __name__ == "__main__":
    main()
