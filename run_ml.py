#!/usr/bin/env python3
"""One-shot: crawl the latest XSMB draws, retrain the ML models, and predict
today's/next draw's top pick.

Pipeline:
    1. Crawl/update the local dataset (reuses run.py's update_dataset, so
       only the missing days get fetched).
    2. Rebuild the feature panel so it includes the freshest draws (reuses
       train_ml.py's caching — it rebuilds automatically once the row count
       no longer matches the updated dataset).
    3. Train + evaluate on a held-out window (last EVAL_DAYS days) for an
       honest performance readout, exactly like train_ml.py.
    4. Refit on the FULL dataset (today's draw included) and predict the
       very next, not-yet-played day.

Usage:
    python3 run_ml.py
"""
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import predict_loto as predictor
import run as daily_run
import train_ml as tml

EVAL_DAYS = 500
WARMUP = 365

MODEL_SPECS = [
    ("Logistic Regression", lambda: LogisticRegression(max_iter=1000)),
    ("Gradient Boosting", lambda: HistGradientBoostingClassifier(max_depth=4, random_state=0)),
]


def evaluate_on_holdout(df, eval_days):
    """Same train/test split + metrics as train_ml.py — an honest performance check."""
    unique_dates = sorted(df["date"].unique())
    cutoff = unique_dates[-eval_days]
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    print(
        f"Đánh giá trên {test['date'].nunique()} kỳ gần nhất "
        f"({test['date'].min():%d/%m/%Y} -> {test['date'].max():%d/%m/%Y}), "
        f"train {train['date'].nunique()} kỳ trước đó:"
    )
    for name, make_clf in MODEL_SPECS:
        pipe = tml.make_pipeline(make_clf())
        pipe.fit(train[tml.FEATURES], train["label"])
        tml.evaluate_model(name, pipe, test[tml.FEATURES], test["label"], test["date"])


def features_for_day(hist, target_date):
    """Build the FEATURES row for every number on a day not yet in the dataset."""
    base = predictor.score(hist, target_date)
    t = hist.cutoff(target_date)
    extra = tml.extra_heat_features(hist, t, tml.EXTRA_HEAT_WINDOWS)
    rows = [
        {
            "number": n,
            "tens": int(n[0]),
            "units": int(n[1]),
            "weekday": target_date.weekday(),
            "freq_long": base[n]["freq_long"],
            "heat_30": base[n]["heat"],
            "overdue": base[n]["overdue"],
            "weekday_rate": base[n]["weekday"],
            **extra[n],
        }
        for n in tml.NUMBERS
    ]
    return pd.DataFrame(rows)


def predict_today(df, hist, target_date):
    """Refit each model on 100% of the data and rank every number for target_date."""
    X_target = features_for_day(hist, target_date)
    predictions = {}
    for name, make_clf in MODEL_SPECS:
        pipe = tml.make_pipeline(make_clf())
        pipe.fit(df[tml.FEATURES], df["label"])
        prob = pipe.predict_proba(X_target[tml.FEATURES])[:, 1]
        predictions[name] = sorted(zip(X_target["number"], prob), key=lambda kv: -kv[1])
    return predictions


def print_predictions(predictions, target_date, top_n=5):
    print(f"\n=== Dự đoán cho {target_date:%d/%m/%Y} ({predictor.WEEKDAYS_VI[target_date.weekday()]}) ===")
    for name, ranked in predictions.items():
        top = ", ".join(f"{n} ({p:.3f})" for n, p in ranked[:top_n])
        print(f"{name:<22}: {top}")

    picks = {name: ranked[0][0] for name, ranked in predictions.items()}
    agree = len(set(picks.values())) == 1
    print(
        f"\n{'2 model đồng thuận' if agree else '2 model KHÔNG đồng thuận'}: "
        + ", ".join(f"{name} -> {pick}" for name, pick in picks.items())
    )
    print(
        "Lưu ý: các backtest trước đó cho thấy cả hai model đều không có edge thật (AUC ~0.5) "
        "— đây là số có xác suất dự đoán cao nhất theo model, không phải số 'chắc về'."
    )


def main():
    today = datetime.now()
    by_date = daily_run.update_dataset(today)

    hist = predictor.History(
        [(datetime.strptime(d, predictor.DATE_FMT), c) for d, c in by_date.items()]
    )
    target = hist.dates[-1] + timedelta(days=1)

    df = tml.load_or_build_panel(hist, WARMUP, rebuild=False)

    evaluate_on_holdout(df, EVAL_DAYS)

    predictions = predict_today(df, hist, target)
    print_predictions(predictions, target)


if __name__ == "__main__":
    main()
