#!/usr/bin/env python3
"""Train an actual ML model to predict which XSMB loto number hits each day,
and evaluate it honestly against the same baseline used throughout this
project.

Pipeline:
    1. Features — one row per (day, number), built only from draws strictly
       before that day (no leakage). Reuses predict_loto.score()'s raw
       signals (freq_long/heat_30/overdue/weekday_rate) plus extra
       multi-window momentum features and the number's own digits.
    2. Split — chronological: the last --test-days days are held out, the
       rest is training data. No shuffling, so test is strictly "the future"
       relative to train.
    3. Train — logistic regression (linear, interpretable) and histogram
       gradient boosting (non-linear, can pick up interactions).
    4. Evaluate — row-level AUC / log-loss / Brier score + calibration, and
       the same top-1-per-day backtest (with baseline + sigma) used by
       predict_loto.py / backtest_streak.py, so results are comparable.

Usage:
    python3 train_ml.py
    python3 train_ml.py --test-days 2000 --warmup 365 --rebuild-panel
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import predict_loto as predictor

NUMBERS = predictor.NUMBERS
EXTRA_HEAT_WINDOWS = (7, 14, 60)  # predictor.score() already gives a 30-day heat signal
PANEL_FILE_TMPL = "ml_panel_w{warmup}.csv"

NUMERIC_FEATURES = ["freq_long", "heat_30", "heat_7", "heat_14", "heat_60", "overdue", "weekday_rate"]
CATEGORICAL_FEATURES = ["weekday", "tens", "units"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def extra_heat_features(hist, t, windows):
    """{number: {'heat_<w>': rate}} for each window, at day index t."""
    out = {n: {} for n in NUMBERS}
    for w in windows:
        start = max(0, t - w)
        span = max(t - start, 1)
        for n in NUMBERS:
            out[n][f"heat_{w}"] = (hist.cum[n][t] - hist.cum[n][start]) / span
    return out


def build_panel(hist, warmup):
    """One row per (day, number): leakage-safe features + label."""
    rows = []
    for t in range(warmup, hist.n_days):
        date = hist.dates[t]
        base = predictor.score(hist, date)
        extra = extra_heat_features(hist, t, EXTRA_HEAT_WINDOWS)
        actual = hist.actual(t)
        for n in NUMBERS:
            rows.append(
                {
                    "date": date,
                    "number": n,
                    "tens": int(n[0]),
                    "units": int(n[1]),
                    "weekday": date.weekday(),
                    "freq_long": base[n]["freq_long"],
                    "heat_30": base[n]["heat"],
                    "overdue": base[n]["overdue"],
                    "weekday_rate": base[n]["weekday"],
                    **extra[n],
                    "label": int(n in actual),
                }
            )
    return pd.DataFrame(rows)


def load_or_build_panel(hist, warmup, rebuild):
    path = PANEL_FILE_TMPL.format(warmup=warmup)
    if not rebuild and os.path.exists(path):
        print(f"Nạp panel đã cache: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        if len(df) == (hist.n_days - warmup) * 100:
            return df
        print("Panel cache lệch số dòng so với dữ liệu hiện tại, build lại...")
    print(f"Build panel đặc trưng ({hist.n_days - warmup} kỳ x 100 số)...")
    df = build_panel(hist, warmup)
    df.to_csv(path, index=False)
    print(f"Đã lưu panel -> {path}\n")
    return df


def make_pipeline(clf):
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def top1_backtest(dates, y_true, y_prob):
    """Group by date, pick the argmax-probability number, compare to baseline."""
    df = pd.DataFrame({"date": dates, "y": y_true, "p": y_prob})
    wins, base_rates = 0, []
    for _date, group in df.groupby("date"):
        pick_idx = group["p"].idxmax()
        wins += int(group.loc[pick_idx, "y"])
        base_rates.append(group["y"].mean())
    n_days = df["date"].nunique()
    rate = wins / n_days
    base_rate = float(np.mean(base_rates))
    se = math.sqrt(base_rate * (1 - base_rate) / n_days) if base_rate else 0.0
    sigma = (rate - base_rate) / se if se else 0.0
    return {"n_days": n_days, "wins": wins, "rate": rate, "base_rate": base_rate, "sigma": sigma}


def calibration_table(y_true, y_prob, bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bucket"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bucket", observed=True).agg(mean_pred=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"))


def evaluate_model(name, pipe, X_test, y_test, test_dates):
    prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    ll = log_loss(y_test, prob)
    brier = brier_score_loss(y_test, prob)
    bt = top1_backtest(test_dates, y_test, prob)

    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc:.4f} (0.5 = không phân biệt được gì, 1.0 = hoàn hảo)")
    print(f"Log-loss: {ll:.4f} | Brier score: {brier:.4f}")
    print("\nHiệu chỉnh xác suất (calibration) theo 10 nhóm phân vị:")
    print(calibration_table(y_test, prob).to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nBacktest top-1/ngày trên {bt['n_days']} kỳ test: "
        f"thắng {bt['wins']}/{bt['n_days']} ({bt['rate']:.1%}) "
        f"| baseline {bt['base_rate']:.1%} | chênh {bt['rate'] - bt['base_rate']:+.1%} ({bt['sigma']:+.2f}σ)"
    )
    return {"auc": auc, "log_loss": ll, "brier": brier, **bt}


def print_logreg_coefficients(pipe, top_n=10):
    clf = pipe.named_steps["clf"]
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(clf.coef_[0], index=feature_names).sort_values()
    print("\nHệ số logistic regression (âm = giảm xác suất về, dương = tăng):")
    print(pd.concat([coefs.head(top_n), coefs.tail(top_n)]).to_string(float_format=lambda x: f"{x:+.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="mb_history_long.csv", help="Long CSV từ crawl_loto.py")
    parser.add_argument("--warmup", type=int, default=365, help="Số kỳ đầu bỏ qua khi build panel (default: 365)")
    parser.add_argument("--test-days", type=int, default=2000, help="Số kỳ gần nhất giữ làm test set (default: 2000)")
    parser.add_argument("--rebuild-panel", action="store_true", help="Bỏ qua cache, build lại panel đặc trưng")
    args = parser.parse_args()

    hist = predictor.load_history(args.input)
    df = load_or_build_panel(hist, args.warmup, args.rebuild_panel)

    unique_dates = sorted(df["date"].unique())
    if args.test_days >= len(unique_dates):
        raise SystemExit(f"--test-days ({args.test_days}) >= tổng số kỳ có sẵn ({len(unique_dates)})")
    cutoff = unique_dates[-args.test_days]

    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    print(
        f"Train: {train['date'].nunique()} kỳ ({train['date'].min():%d/%m/%Y} -> {train['date'].max():%d/%m/%Y}), "
        f"{len(train)} dòng"
    )
    print(
        f"Test:  {test['date'].nunique()} kỳ ({test['date'].min():%d/%m/%Y} -> {test['date'].max():%d/%m/%Y}), "
        f"{len(test)} dòng"
    )

    X_train, y_train = train[FEATURES], train["label"]
    X_test, y_test = test[FEATURES], test["label"]

    models = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
        "Gradient Boosting": make_pipeline(HistGradientBoostingClassifier(max_depth=4, random_state=0)),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(name, pipe, X_test, y_test, test["date"])

    print_logreg_coefficients(models["Logistic Regression"])

    print("\n=== So sánh tổng kết (cùng kỳ test) ===")
    print(f"{'Model':<22}{'AUC':>7}{'LogLoss':>10}{'Brier':>8}{'Top-1':>9}{'Baseline':>10}{'Sigma':>8}")
    for name, r in results.items():
        print(
            f"{name:<22}{r['auc']:>7.4f}{r['log_loss']:>10.4f}{r['brier']:>8.4f}"
            f"{r['rate']:>9.1%}{r['base_rate']:>10.1%}{r['sigma']:>+7.2f}σ"
        )
    print(
        "\nGhi chú: AUC ~0.5 và |sigma| < 2 nghĩa là model không tìm được tín hiệu thật nào — "
        "khớp với các backtest heuristic trước đó, đúng với giả định các kỳ độc lập nhau."
    )


if __name__ == "__main__":
    main()
