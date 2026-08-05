#!/usr/bin/env python3
"""Crawl loto (2-digit lottery number) frequency history for XSMB
(Miền Bắc) from xoso.com.vn.

Uses the same AJAX endpoint the site's own frontend calls
(/ThongKe/AjaxTanSuatLo) instead of scraping rendered HTML.

Examples:
    python3 crawl_loto.py --days 100 --output mb_100d
    python3 crawl_loto.py --days 2000 --end-date 29/07/2026 --output mb_2000d
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://xoso.com.vn"
AJAX_URL = f"{BASE_URL}/ThongKe/AjaxTanSuatLo"
DATE_FMT = "%d/%m/%Y"
MAX_ROLL = 1000
LOTTERY_ID = 0  # Miền Bắc

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": f"{BASE_URL}/thong-ke-tan-suat-loto.html",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_chunk(session, end_date, days):
    """Fetch one AJAX response: `days` draws ending the day before end_date."""
    params = {
        "number": "",
        "date": end_date.strftime(DATE_FMT),
        "numberRoll": days,
        "bangtkType": 0,  # 0 = horizontal table
        "chonType": 0,    # 0 = all 100 numbers
        "lotteryId": LOTTERY_ID,
    }
    resp = session.get(AJAX_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def parse_chunk(html):
    """Parse one AJAX response into {date_str: {number_str: count}}."""
    soup = BeautifulSoup(html, "lxml")
    dates = []
    for a in soup.select("thead th a[title]"):
        title = a["title"]  # e.g. "XSMB 28/07/2026"
        dates.append(title.split()[-1])

    by_date = {d: {} for d in dates}
    for tr in soup.select("tbody tr.tansuatrow"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        number = cells[0].get_text(strip=True).zfill(2)
        day_cells = cells[1 : 1 + len(dates)]
        for date_str, cell in zip(dates, day_cells):
            text = cell.get_text(strip=True)
            by_date[date_str][number] = int(text) if text else 0
    return by_date


def crawl(session, end_date, total_days, delay=1.0, log=lambda msg: None):
    """Fetch `total_days` of history ending the day before end_date.

    Chunks requests since the site caps a single call at MAX_ROLL days.
    Returns {date_str ("dd/mm/yyyy"): {number_str ("00".."99"): count}}.
    """
    merged = {}
    remaining = total_days
    cursor = end_date
    while remaining > 0:
        chunk_days = min(remaining, MAX_ROLL)
        log(f"fetching {chunk_days} day(s) ending before {cursor.strftime(DATE_FMT)}")
        html = fetch_chunk(session, cursor, chunk_days)
        merged.update(parse_chunk(html))
        remaining -= chunk_days
        cursor = cursor - timedelta(days=chunk_days)
        if remaining > 0:
            time.sleep(delay)
    return merged


def summarize(by_date):
    """Roll {date: {number: count}} up into {number: total_count}, sorted desc."""
    totals = {f"{i:02d}": 0 for i in range(100)}
    for counts in by_date.values():
        for number, count in counts.items():
            totals[number] = totals.get(number, 0) + count
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def write_long_csv(path, by_date):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "number", "count"])
        for date_str in sorted(by_date, key=lambda d: datetime.strptime(d, DATE_FMT)):
            for number, count in sorted(by_date[date_str].items()):
                writer.writerow([date_str, number, count])


def write_summary_csv(path, totals):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "total_count"])
        for number, total in totals.items():
            writer.writerow([number, total])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end-date", default=None, help="dd/mm/yyyy, exclusive upper bound (default: today)")
    parser.add_argument("--days", type=int, default=100, help="How many days of history to pull (default: 100)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between chunk requests (default: 1.0)")
    parser.add_argument("--output", default="mb_loto", help="Output file prefix (default: mb_loto)")
    args = parser.parse_args()

    if args.days < 1:
        sys.exit("--days must be >= 1")

    end_date = (
        datetime.strptime(args.end_date, DATE_FMT) if args.end_date else datetime.now()
    )

    session = requests.Session()
    by_date = crawl(session, end_date, args.days, delay=args.delay, log=print)
    totals = summarize(by_date)

    long_path = f"{args.output}_long.csv"
    summary_path = f"{args.output}_summary.csv"
    write_long_csv(long_path, by_date)
    write_summary_csv(summary_path, totals)

    print(f"\nWrote {len(by_date)} day(s) x 100 numbers -> {long_path}")
    print(f"Wrote frequency summary -> {summary_path}")
    print(f"Most frequent: {list(totals.items())[:5]}")
    print(f"Least frequent: {list(totals.items())[-5:]}")


if __name__ == "__main__":
    main()
