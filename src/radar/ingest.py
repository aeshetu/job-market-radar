"""Download every monthly "Ask HN: Who is hiring?" thread from the HN Algolia API.

Since April 2011 the `whoishiring` account has posted one thread per month. Each
top-level comment is one job posting. Threads are cached locally, and the two most
recent months are always re-fetched because they are still collecting posts.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

from .config import INTERIM, RAW

log = logging.getLogger(__name__)

API = "https://hn.algolia.com/api/v1"
TITLE_RE = re.compile(
    r"^Ask HN: Who is hiring\?\s*\((?P<month>[A-Za-z]+)\s+(?P<year>\d{4})\)", re.IGNORECASE
)
MONTHS = {
    name: i
    for i, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}

_session = requests.Session()
_session.headers["User-Agent"] = "job-market-radar (student research project)"


def _get(url: str, params: dict | None = None, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2**attempt
            log.warning("GET %s failed (%s); retrying in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}")


def _thread_rows(hits: list[dict]) -> list[dict]:
    rows = []
    for hit in hits:
        m = TITLE_RE.match(hit.get("title") or "")
        if not m or m["month"].lower() not in MONTHS:
            continue
        rows.append({
            "thread_id": int(hit["objectID"]),
            "title": hit["title"],
            "author": hit.get("author"),
            "num_comments": hit.get("num_comments") or 0,
            "month": pd.Timestamp(int(m["year"]), MONTHS[m["month"].lower()], 1),
        })
    return rows


def list_threads(min_comments: int = 50) -> pd.DataFrame:
    """Return one row per monthly hiring thread: thread_id, title, author, month.

    Most threads come from `whoishiring`, but a few months were posted by the
    stand-in `_whoishiring` account or by ordinary users (e.g. July 2011), and in
    December 2011 the official thread was nearly empty. Any month that is missing
    or has fewer than `min_comments` comments is looked up by title instead, and
    the most-commented matching story wins.
    """
    rows = []
    for author in ("whoishiring", "_whoishiring"):
        data = _get(f"{API}/search_by_date",
                    {"tags": f"story,author_{author}", "hitsPerPage": 1000})
        rows += _thread_rows(data["hits"])
    df = pd.DataFrame(rows)

    best = df.sort_values("num_comments").groupby("month").tail(1)
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    weak = months.difference(best.loc[best["num_comments"] >= min_comments, "month"])
    for month in weak:
        data = _get(f"{API}/search", {"query": f"Who is hiring {month:%B %Y}",
                                      "tags": "story", "hitsPerPage": 20})
        found = [r for r in _thread_rows(data["hits"]) if r["month"] == month]
        if found:
            rows += found
            log.info("Recovered %s thread %s by %s", f"{month:%Y-%m}",
                     max(found, key=lambda r: r["num_comments"])["thread_id"],
                     max(found, key=lambda r: r["num_comments"])["author"])
        else:
            log.warning("No hiring thread found for %s", f"{month:%Y-%m}")

    df = pd.DataFrame(rows).sort_values(["month", "num_comments"])
    return df.drop_duplicates("month", keep="last").sort_values("month").reset_index(drop=True)


def fetch_thread(thread_id: int, force: bool = False) -> list[dict]:
    """Return the top-level comments (job posts) of one thread, using the local cache."""
    path = RAW / f"{thread_id}.json.gz"
    if path.exists() and not force:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    item = _get(f"{API}/items/{thread_id}")
    posts = [
        {
            "post_id": c["id"],
            "author": c.get("author"),
            "created_at": c.get("created_at"),
            "html": c.get("text"),
        }
        for c in item.get("children", [])
        if c.get("type") == "comment"
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(posts, f)
    return posts


def run(sleep: float = 0.5) -> pd.DataFrame:
    threads = list_threads()
    threads.to_parquet(INTERIM / "threads.parquet", index=False)
    refresh_from = threads["month"].max() - pd.DateOffset(months=1)

    rows = []
    for t in tqdm(threads.itertuples(), total=len(threads), desc="threads"):
        cached = (RAW / f"{t.thread_id}.json.gz").exists()
        posts = fetch_thread(t.thread_id, force=t.month >= refresh_from)
        if not cached or t.month >= refresh_from:
            time.sleep(sleep)  # be polite to the free API
        for p in posts:
            if p["html"] and p["author"]:  # skip deleted / dead comments
                rows.append({**p, "thread_id": t.thread_id, "month": t.month})

    posts_df = pd.DataFrame(rows)
    posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], utc=True)
    posts_df.to_parquet(INTERIM / "posts_raw.parquet", index=False)
    log.info("Saved %d posts from %d threads", len(posts_df), len(threads))
    return posts_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = run()
    print(f"{len(df):,} posts, {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
