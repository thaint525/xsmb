#!/usr/bin/env python3
"""Crawl loto (2-digit lottery number) frequency history for XSMB
(Miền Bắc) from minhngoc.net.vn.

Scrapes the per-day results pages (/ket-qua-xo-so/mien-bac/DD-MM-YYYY.html)
and derives the 2-digit "lô" counts from the last two digits of all 27
prize numbers. Each page happens to render the requested date plus the 6
days before it, so one request covers a whole week of history.

Data goes back to 01/01/2005 — the earliest date the site serves.

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

BASE_URL = "https://www.minhngoc.net.vn"
DATE_FMT = "%d/%m/%Y"
URL_DATE_FMT = "%d-%m-%Y"
EARLIEST_DATE = datetime(2005, 1, 1)  # site has nothing before this

# One page renders the requested day plus the 6 before it.
DAYS_PER_PAGE = 7
CHECKPOINT_EVERY = 50  # pages between checkpoint writes during a long backfill
PRIZE_CLASSES = ("giaidb", "giai1", "giai2", "giai3", "giai4", "giai5", "giai6", "giai7")
EXPECTED_PRIZES = 27  # 1+1+2+6+4+6+3+4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}


def fetch_page(session, date, retries=4, log=lambda msg: None):
    """Fetch the results page anchored at `date` (covers that day + 6 prior).

    Retries transient network failures with exponential backoff — a single
    timeout partway through a multi-thousand-day backfill shouldn't throw
    away the whole run.
    """
    url = f"{BASE_URL}/ket-qua-xo-so/mien-bac/{date.strftime(URL_DATE_FMT)}.html"
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            backoff = 2**attempt
            log(f"  {type(exc).__name__} on {url}, retrying in {backoff}s")
            time.sleep(backoff)


def parse_page(html):
    """Parse one results page into {date_str: {number_str: count}}.

    Each day's card is a table.bkqtinhmienbac holding one td per prize tier;
    every individual prize number sits in its own <div>. The "lô" for a prize
    is its last two digits, so a day yields 27 counted numbers.
    """
    soup = BeautifulSoup(html, "lxml")
    by_date = {}
    for table in soup.select("table.bkqtinhmienbac"):
        ngay_cell = table.select_one("td.ngay")
        if not ngay_cell:
            continue
        date_link = ngay_cell.select_one("a")
        if not date_link:
            continue
        date_str = date_link.get_text(strip=True)

        prizes = []
        for cls in PRIZE_CLASSES:
            td = table.select_one(f"td.{cls}")
            if not td:
                continue
            prizes.extend(
                d.get_text(strip=True) for d in td.select("div") if d.get_text(strip=True).isdigit()
            )

        # Skip partially-rendered or live-updating cards rather than storing
        # a day with missing draws.
        if len(prizes) != EXPECTED_PRIZES:
            continue

        counts = {f"{i:02d}": 0 for i in range(100)}
        for prize in prizes:
            counts[prize[-2:].zfill(2)] += 1
        by_date[date_str] = counts
    return by_date


def crawl(session, end_date, total_days, delay=1.0, log=lambda msg: None, checkpoint=None):
    """Fetch `total_days` of history ending the day before end_date.

    Walks backwards a page (a week) at a time. Returns
    {date_str ("dd/mm/yyyy"): {number_str ("00".."99"): count}} — same shape
    as the previous xoso.com.vn crawler, so callers need no changes.

    `checkpoint`, if given, is called with the merged dict every
    CHECKPOINT_EVERY pages so a long backfill can be salvaged if it dies
    partway through.
    """
    merged = {}
    cursor = end_date - timedelta(days=1)  # end_date is an exclusive bound
    oldest_wanted = end_date - timedelta(days=total_days)
    pages = 0

    while cursor >= oldest_wanted and cursor >= EARLIEST_DATE:
        log(f"fetching week ending {cursor.strftime(DATE_FMT)}")
        page = parse_page(fetch_page(session, cursor, log=log))
        if not page:
            log(f"  no results found for {cursor.strftime(DATE_FMT)}, stopping")
            break
        merged.update(page)
        pages += 1
        if checkpoint and pages % CHECKPOINT_EVERY == 0:
            checkpoint(merged)
            log(f"  checkpoint: {len(merged)} day(s) saved")
        cursor -= timedelta(days=DAYS_PER_PAGE)
        if cursor >= oldest_wanted and cursor >= EARLIEST_DATE:
            time.sleep(delay)

    # A page may reach past the requested window; trim to what was asked for.
    return {
        d: counts
        for d, counts in merged.items()
        if oldest_wanted <= datetime.strptime(d, DATE_FMT) < end_date
    }


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
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between page requests (default: 1.0)")
    parser.add_argument("--output", default="mb_loto", help="Output file prefix (default: mb_loto)")
    args = parser.parse_args()

    if args.days < 1:
        sys.exit("--days must be >= 1")

    end_date = (
        datetime.strptime(args.end_date, DATE_FMT) if args.end_date else datetime.now()
    )

    long_path = f"{args.output}_long.csv"
    session = requests.Session()
    by_date = crawl(
        session, end_date, args.days, delay=args.delay, log=print,
        checkpoint=lambda partial: write_long_csv(long_path, partial),
    )
    totals = summarize(by_date)

    summary_path = f"{args.output}_summary.csv"
    write_long_csv(long_path, by_date)
    write_summary_csv(summary_path, totals)

    print(f"\nWrote {len(by_date)} day(s) x 100 numbers -> {long_path}")
    print(f"Wrote frequency summary -> {summary_path}")
    print(f"Most frequent: {list(totals.items())[:5]}")
    print(f"Least frequent: {list(totals.items())[-5:]}")


if __name__ == "__main__":
    main()
