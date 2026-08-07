#!/usr/bin/env python3
"""One-shot: pull the latest XSMB loto draws, update the local dataset, and
print today's/next draw's pick.

Incrementally updates DATA_FILE with any new draws since the last run (or
does a full crawl from scratch on first use), then feeds the combined
history straight into predict_loto. Reuses crawl_loto/predict_loto directly
instead of duplicating their logic.

Usage:
    python3 run.py               # Vietnamese output (default)
    python3 run.py --lang en     # English output (used by the Telegram workflow)
"""
import argparse
import csv
import math
from bisect import bisect_left
from datetime import datetime, timedelta

import requests

import backtest_streak as streaks
import backtest_wait_k as waitk
import crawl_loto as crawler
import predict_loto as predictor

AVG_HITS_PER_NUMBER = 27 / 100  # 27 lô rơi vào 100 số mỗi kỳ, nếu chia đều

DATA_FILE = "mb_history_long.csv"
FULL_HISTORY_DAYS = 5000  # comfortably covers the site's data (starts 01/01/2014)
BACKTEST_DAYS = 500
RECENT_DAYS = 7
STREAK_WARMUP = 365  # bỏ qua 1 năm đầu khi tính chuỗi thắng/thua + gợi ý k
WAIT_K_RANGE = range(1, 16)
WAIT_K_MIN_SAMPLE = 200  # bỏ qua k nào có quá ít lần vào để tránh gợi ý dựa trên mẫu nhỏ

STRINGS = {
    "vi": {
        "update_existing": "Đã có dữ liệu tới {last_date}, cập nhật thêm {days} ngày gần nhất...",
        "update_fresh": "Chưa có dữ liệu cục bộ, crawl toàn bộ lịch sử (tối đa {days} ngày)...",
        "update_saved": "Đã lưu {n} kỳ vào {file}\n",
        "data_range": "Dữ liệu: {n_days} kỳ, {start} → {end}",
        "target": "Dự đoán cho: {target} ({weekday})\n",
        "ranking_header": "\nBảng xếp hạng top 10:",
        "final_pick": "\n>>> SỐ CHỐT: {pick} <<<",
        "table_headers": ("#", "Lô", "Điểm", "TS dài", "Nóng 30", "Gan/CK", "Theo thứ"),
        "backtest_header": "\nBacktest {days} kỳ gần nhất:",
        "backtest_row": "  top-{k}: model trúng {hit_rate:.1%} | đoán bừa {base_rate:.1%} | chênh {edge:+.1%} ({sigma:+.2f} sigma)",
        "backtest_verdict_insig": "chênh lệch không phân biệt được với nhiễu ngẫu nhiên (<2 sigma)",
        "backtest_verdict_sig": "chênh lệch có ý nghĩa thống kê (>=2 sigma), nhưng vẫn nên kiểm tra lại trước khi tin",
        "backtest_footer": "  => {verdict}. Các số về XSMB coi như độc lập — đừng đặt cược thật theo số chốt này.",
        "explain_header": "\n=== SỐ CHỐT HÔM NAY ({target}): {n} ===",
        "explain_why": "Vì sao {n} đứng đầu:",
        "explain_freq": "  - Tần suất dài hạn : {rate:.3f} lô/kỳ (top {rank}/100 toàn thời gian)",
        "explain_heat": "  - Đang nóng (30 kỳ): {rate:.3f} lô/kỳ ({side} trung bình {avg:.2f})",
        "heat_above": "trên",
        "heat_below": "dưới",
        "explain_gan": "  - Lô gan           : gan {streak} kỳ (về gần nhất {date}), chu kỳ TB {gap:.2f} kỳ -> gấp {ratio:.2f}x",
        "explain_gan_never": "  - Lô gan           : chưa từng về trong lịch sử đã crawl",
        "explain_weekday": "  - Theo {weekday}      : {rate:.3f} lô/kỳ",
        "explain_top5": "Top 5 kế tiếp nếu muốn dàn rộng: {list}",
        "waitk_header": "\n=== Gợi ý \"đợi thua liên tiếp rồi mới vào\" ===",
        "waitk_none": "Không có k nào đủ mẫu (>= {min_sample} lần vào) để gợi ý.",
        "waitk_best": "k={k}: {n} lần vào, thắng {rate:.1%} (baseline {baseline:.1%}, chênh {edge:+.1%}, {sigma:+.2f}σ) — |sigma| cao nhất trong các k có mẫu đủ lớn",
        "waitk_insig": "=> Chênh lệch này KHÔNG có ý nghĩa thống kê — đây không phải một mức k thật sự tốt hơn, chỉ là ít nhiễu nhất trong dữ liệu hiện có.",
        "waitk_no_streak": "Hiện tại: kỳ gần nhất đang THẮNG, chưa bắt đầu chuỗi thua nào.",
        "waitk_streak": "Hiện tại: đang thua liên tiếp {n} kỳ (từ {date}).",
        "waitk_ready": "=> Đã đủ {k} kỳ thua theo gợi ý trên (dù vậy vẫn không có edge thật, xem cảnh báo trên).",
        "waitk_wait_more": "=> Còn thiếu {n} kỳ thua nữa mới đủ điều kiện theo gợi ý trên.",
        "recent_header": "\nHiệu suất top-1 trong {n} kỳ gần nhất: thắng {wins}/{n}",
        "recent_row": "  {date}  lô {pick}  {result}",
        "win": "Thắng",
        "lose": "Thua",
        "weekdays": ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"],
    },
    "en": {
        "update_existing": "Existing data up to {last_date}, fetching {days} more day(s)...",
        "update_fresh": "No local data yet, crawling full history (up to {days} days)...",
        "update_saved": "Saved {n} draws to {file}\n",
        "data_range": "Data: {n_days} draws, {start} -> {end}",
        "target": "Prediction for: {target} ({weekday})\n",
        "ranking_header": "\nTop 10 ranking:",
        "final_pick": "\n>>> FINAL PICK: {pick} <<<",
        "table_headers": ("#", "No.", "Score", "LongFreq", "Hot30", "Overdue", "Weekday"),
        "backtest_header": "\nBacktest, last {days} draws:",
        "backtest_row": "  top-{k}: model hit {hit_rate:.1%} | random guess {base_rate:.1%} | edge {edge:+.1%} ({sigma:+.2f} sigma)",
        "backtest_verdict_insig": "the edge is indistinguishable from random noise (<2 sigma)",
        "backtest_verdict_sig": "the edge is statistically significant (>=2 sigma), but still double-check before trusting it",
        "backtest_footer": "  => {verdict}. XSMB draws are effectively independent — don't bet real money on this pick.",
        "explain_header": "\n=== TODAY'S PICK ({target}): {n} ===",
        "explain_why": "Why {n} ranks first:",
        "explain_freq": "  - Long-run frequency: {rate:.3f} hits/draw (top {rank}/100 all-time)",
        "explain_heat": "  - Hot (last 30 draws): {rate:.3f} hits/draw ({side} the average {avg:.2f})",
        "heat_above": "above",
        "heat_below": "below",
        "explain_gan": "  - Dry streak        : {streak} draws since last hit (last seen {date}), avg cycle {gap:.2f} draws -> {ratio:.2f}x",
        "explain_gan_never": "  - Dry streak        : never hit in the crawled history",
        "explain_weekday": "  - On {weekday}     : {rate:.3f} hits/draw",
        "explain_top5": "Next 5 for a wider spread: {list}",
        "waitk_header": '\n=== "Wait for a losing streak before betting" suggestion ===',
        "waitk_none": "No k has enough samples (>= {min_sample} entries) to suggest.",
        "waitk_best": "k={k}: {n} entries, won {rate:.1%} (baseline {baseline:.1%}, edge {edge:+.1%}, {sigma:+.2f}σ) — highest |sigma| among k values with a large enough sample",
        "waitk_insig": "=> This edge is NOT statistically significant — it isn't a genuinely better threshold, just the least noisy one in the current data.",
        "waitk_no_streak": "Right now: the last draw was a WIN, no losing streak in progress.",
        "waitk_streak": "Right now: {n} losses in a row (since {date}).",
        "waitk_ready": "=> Already at {k} losses per the suggestion above (still no real edge though, see the warning above).",
        "waitk_wait_more": "=> {n} more {loss_word} needed to meet the suggestion above.",
        "recent_header": "\nTop-1 performance over the last {n} draws: won {wins}/{n}",
        "recent_row": "  {date}  number {pick}  {result}",
        "win": "Win",
        "lose": "Loss",
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    },
}


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


def update_dataset(today, msg):
    existing = load_existing(DATA_FILE)
    if existing:
        last_date = max(existing, key=lambda d: datetime.strptime(d, crawler.DATE_FMT))
        last_dt = datetime.strptime(last_date, crawler.DATE_FMT)
        days_needed = (today - last_dt).days + 2  # small overlap, cheap & safe
        print(msg["update_existing"].format(last_date=last_date, days=days_needed))
    else:
        days_needed = FULL_HISTORY_DAYS
        print(msg["update_fresh"].format(days=days_needed))

    session = requests.Session()
    new_data = crawler.crawl(session, today, days_needed, delay=1.0, log=print)
    existing.update(new_data)
    save(DATA_FILE, existing)
    print(msg["update_saved"].format(n=len(existing), file=DATA_FILE))
    return existing


def print_ranking_table(ranked, msg):
    h = msg["table_headers"]
    print(f"{h[0]:<3}{h[1]:<5}{h[2]:>8}{h[3]:>9}{h[4]:>9}{h[5]:>8}{h[6]:>10}")
    print("-" * 52)
    for rank, (n, s) in enumerate(ranked[:10], 1):
        print(
            f"{rank:<3}{n:<5}{s['score']:>8.2f}{s['freq_long']:>9.3f}"
            f"{s['heat']:>9.3f}{s['overdue']:>8.2f}{s['weekday']:>10.3f}"
        )


def print_backtest(hist, msg):
    bt = predictor.backtest(hist, BACKTEST_DAYS)
    print(msg["backtest_header"].format(days=bt["days"]))
    max_sigma = 0.0
    for k, r in bt["results"].items():
        edge = r["hit_rate"] - r["base_rate"]
        se = math.sqrt(r["base_rate"] * (1 - r["base_rate"]) / bt["days"])
        sigma = edge / se if se else 0.0
        max_sigma = max(max_sigma, abs(sigma))
        print(msg["backtest_row"].format(k=k, hit_rate=r["hit_rate"], base_rate=r["base_rate"], edge=edge, sigma=sigma))
    verdict = msg["backtest_verdict_insig"] if max_sigma < 2 else msg["backtest_verdict_sig"]
    print(msg["backtest_footer"].format(verdict=verdict))


def explain_top_pick(hist, target, ranked, msg, weekdays):
    """Human-readable breakdown of why the #1 pick scored highest."""
    n, s = ranked[0]
    t = hist.cutoff(target)

    by_freq = sorted(ranked, key=lambda kv: -kv[1]["freq_long"])
    freq_rank = next(i for i, (num, _) in enumerate(by_freq, 1) if num == n)

    hits_idx = hist.hit_idx[n]
    j = bisect_left(hits_idx, t)

    lines = [
        msg["explain_header"].format(target=f"{target:%d/%m/%Y}", n=n),
        msg["explain_why"].format(n=n),
        msg["explain_freq"].format(rate=s["freq_long"], rank=freq_rank),
    ]
    side = msg["heat_above"] if s["heat"] > AVG_HITS_PER_NUMBER else msg["heat_below"]
    lines.append(msg["explain_heat"].format(rate=s["heat"], side=side, avg=AVG_HITS_PER_NUMBER))
    if j:
        last_hit_date = hist.dates[hits_idx[j - 1]]
        streak = t - 1 - hits_idx[j - 1]
        mean_gap = (hits_idx[j - 1] - hits_idx[0]) / (j - 1) if j > 1 else float(t)
        lines.append(
            msg["explain_gan"].format(
                streak=streak, date=f"{last_hit_date:%d/%m/%Y}", gap=mean_gap, ratio=s["overdue"]
            )
        )
    else:
        lines.append(msg["explain_gan_never"])
    lines.append(msg["explain_weekday"].format(weekday=weekdays[target.weekday()], rate=s["weekday"]))

    top5_next = ", ".join(num for num, _ in ranked[1:5])
    lines.append(msg["explain_top5"].format(list=top5_next))
    return "\n".join(lines)


def print_recent(results, msg):
    wins = sum(1 for _, _, hit in results if hit)
    print(msg["recent_header"].format(n=len(results), wins=wins))
    for date, pick, hit in results:
        result = msg["win"] if hit else msg["lose"]
        print(msg["recent_row"].format(date=f"{date:%d/%m/%Y}", pick=pick, result=result))


def suggest_wait_k(full_results, msg):
    """Best k among candidates with enough sample, ranked by |sigma| (not raw
    edge — sigma already accounts for each k's sample size, so it doesn't
    reward a big-looking edge that's really just a small-n fluke) — plus the
    current live losing streak, so the reader knows whether they'd be "in"
    right now under that suggestion. Purely descriptive: see the disclaimer
    this prints alongside the number.
    """
    baseline_rate = sum(1 for _, _, hit in full_results if hit) / len(full_results)

    candidates = []
    for k in WAIT_K_RANGE:
        bets = waitk.simulate_wait_k_losses(full_results, k)
        if len(bets) < WAIT_K_MIN_SAMPLE:
            continue
        s = waitk.summarize(bets, baseline_rate)
        candidates.append((k, s))

    cur_len, cur_start = streaks.current_streak(full_results)

    print(msg["waitk_header"])
    if not candidates:
        print(msg["waitk_none"].format(min_sample=WAIT_K_MIN_SAMPLE))
        return

    best_k, best_s = max(candidates, key=lambda kv: abs(kv[1]["sigma"]))
    print(
        msg["waitk_best"].format(
            k=best_k, n=best_s["n"], rate=best_s["rate"], baseline=baseline_rate,
            edge=best_s["edge"], sigma=best_s["sigma"],
        )
    )
    if abs(best_s["sigma"]) < 2:
        print(msg["waitk_insig"])

    if cur_len == 0:
        print(msg["waitk_no_streak"])
    else:
        print(msg["waitk_streak"].format(n=cur_len, date=f"{cur_start:%d/%m/%Y}"))
        if cur_len >= best_k:
            print(msg["waitk_ready"].format(k=best_k))
        else:
            remaining = best_k - cur_len
            loss_word = "loss" if remaining == 1 else "losses"
            print(msg["waitk_wait_more"].format(n=remaining, loss_word=loss_word))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["vi", "en"], default="vi", help="Ngôn ngữ output (default: vi)")
    args = parser.parse_args()
    msg = STRINGS[args.lang]
    weekdays = msg["weekdays"]

    today = datetime.now()
    by_date = update_dataset(today, msg)

    hist = predictor.History(
        [(datetime.strptime(d, predictor.DATE_FMT), c) for d, c in by_date.items()]
    )
    target = hist.dates[-1] + timedelta(days=1)

    print(msg["data_range"].format(n_days=hist.n_days, start=f"{hist.dates[0]:%d/%m/%Y}", end=f"{hist.dates[-1]:%d/%m/%Y}"))
    print(msg["target"].format(target=f"{target:%d/%m/%Y}", weekday=weekdays[target.weekday()]))

    ranked = sorted(predictor.score(hist, target).items(), key=lambda kv: -kv[1]["score"])
    print(explain_top_pick(hist, target, ranked, msg, weekdays))

    print(msg["ranking_header"])
    print_ranking_table(ranked, msg)
    print_backtest(hist, msg)

    full_results = streaks.run_strategy(hist, warmup=STREAK_WARMUP)
    print_recent(full_results[-RECENT_DAYS:], msg)
    suggest_wait_k(full_results, msg)

    print(msg["final_pick"].format(pick=ranked[0][0]))


if __name__ == "__main__":
    main()
