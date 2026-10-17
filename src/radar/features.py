"""Raw posts -> one structured row per posting (data/interim/posts.parquet)."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import parse
from .config import INTERIM
from .salary import extract_salary
from .skills import extract_skills

log = logging.getLogger(__name__)


def featurize(html: str) -> dict:
    text = parse.html_to_text(html)
    header = parse.header_of(text)
    clean = parse.strip_urls(text)
    company = parse.company_of(header)
    return {
        "text": text,
        "header": header,
        "company": company,
        "company_key": parse.company_key(company),
        "n_chars": len(text),
        "work_mode": parse.work_mode(parse.strip_urls(header), clean),
        "visa": parse.offers_visa(clean),
        "metro": parse.metro_of(header),
        "skills": extract_skills(clean),
        **parse.seniority_flags(clean),
        **parse.role_flags(parse.role_text(header, clean)),
        **extract_salary(clean),
    }


def run() -> pd.DataFrame:
    raw = pd.read_parquet(INTERIM / "posts_raw.parquet")
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        feats = pd.DataFrame(list(pool.map(featurize, raw["html"], chunksize=500)))
    df = pd.concat([raw.drop(columns="html").reset_index(drop=True), feats], axis=1)
    # A handful of top-level comments are meta-discussion, not job posts.
    df = df[df["n_chars"] >= 80].reset_index(drop=True)
    df["year"] = df["month"].dt.year
    df.to_parquet(INTERIM / "posts.parquet", index=False)
    log.info("Featurized %d posts", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
