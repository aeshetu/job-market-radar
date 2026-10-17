"""Market-level monthly metrics, skill demand shares, statistically tested movers,
and "A overtook B" crossover detection."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from .config import INTERIM, PROCESSED
from .skills import CATEGORY, SKILLS

log = logging.getLogger(__name__)

CROSSOVER_PAIRS = [
    ("PyTorch", "TensorFlow"),
    ("TypeScript", "JavaScript"),
    ("React", "Angular"),
    ("Go", "Ruby"),
    ("Rust", "C++"),
    ("Kubernetes", "Heroku"),
    ("PostgreSQL", "MySQL"),
    ("Snowflake", "Redshift"),
    ("Kotlin", "Java"),
    ("Swift", "Objective-C"),
    ("LLMs / GenAI", "Machine learning"),
    ("Python", "JavaScript"),
]


def market_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")
    out = pd.DataFrame({
        "posts": g.size(),
        "companies": g["company_key"].nunique(),
        "remote_share": g["work_mode"].apply(lambda s: (s == "remote").mean()),
        "hybrid_share": g["work_mode"].apply(lambda s: (s == "hybrid").mean()),
        "onsite_share": g["work_mode"].apply(lambda s: (s == "onsite").mean()),
        "salary_share": g["salary_mid"].apply(lambda s: s.notna().mean()),
        "visa_share": g["visa"].mean(),
        # If posts list fewer technologies over time, every skill's share falls even
        # with flat demand. Tracked so that confound is visible.
        "skills_per_post": g["skills"].apply(lambda s: s.str.len().mean()),
        "median_chars": g["n_chars"].median(),
    })
    usd = df[df["salary_currency"] == "USD"]
    out["median_salary_usd"] = usd.groupby("month")["salary_mid"].median()
    out["salary_n_usd"] = usd.groupby("month").size()
    flag_cols = [c for c in df.columns if c.startswith(("role_", "lvl_"))]
    out = out.join(g[flag_cols].mean())
    return out.reset_index()


def skill_monthly(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("month").size().rename("posts")
    counts = (df[["month", "skills"]].explode("skills").dropna()
              .groupby(["month", "skills"]).size().rename("n"))
    grid = pd.MultiIndex.from_product([totals.index, SKILLS], names=["month", "skill"])
    out = counts.rename_axis(["month", "skill"]).reindex(grid, fill_value=0).reset_index()
    out = out.merge(totals.reset_index(), on="month")
    out["share"] = out["n"] / out["posts"]
    out["ci_lo"], out["ci_hi"] = proportion_confint(out["n"], out["posts"], alpha=0.05, method="wilson")

    # Smooth by pooling counts over a centred 3-month window (better than averaging
    # shares, because busy months get more weight).
    out = out.sort_values(["skill", "month"])
    roll = out.groupby("skill")[["n", "posts"]].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).sum())
    out["share_smooth"] = roll["n"] / roll["posts"]
    out["category"] = out["skill"].map(CATEGORY)
    return out.reset_index(drop=True)


def movers(df: pd.DataFrame, years_back: int, window_months: int = 12) -> pd.DataFrame:
    """Compare each skill's share in the latest window vs. the same window `years_back` ago.

    Two-proportion z-tests with Benjamini-Hochberg FDR control across all skills, so
    "significant" means q < 0.05 after testing ~90 skills at once.
    """
    end = df["month"].max()
    now_start = end - pd.DateOffset(months=window_months - 1)
    then_end = end - pd.DateOffset(years=years_back)
    then_start = then_end - pd.DateOffset(months=window_months - 1)
    now = df[df["month"].between(now_start, end)]
    then = df[df["month"].between(then_start, then_end)]

    rows = []
    for skill in SKILLS:
        a = int(now["skills"].apply(lambda s: skill in s).sum())
        b = int(then["skills"].apply(lambda s: skill in s).sum())
        if a + b < 20:
            continue
        _, p = proportions_ztest([a, b], [len(now), len(then)])
        rows.append({
            "skill": skill, "category": CATEGORY[skill],
            "share_now": a / len(now), "share_then": b / len(then),
            "n_now": a, "n_then": b, "p": p,
        })
    out = pd.DataFrame(rows)
    out["p"] = out["p"].fillna(1.0)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    out["delta_pp"] = 100 * (out["share_now"] - out["share_then"])
    out["ratio"] = (out["share_now"] + 1e-4) / (out["share_then"] + 1e-4)
    out["significant"] = out["q"] < 0.05
    out["window"] = f"{years_back}y"
    out["period_now"] = f"{now_start:%b %Y}–{end:%b %Y}"
    out["period_then"] = f"{then_start:%b %Y}–{then_end:%b %Y}"
    return out.sort_values("delta_pp", ascending=False).reset_index(drop=True)


def crossovers(sm: pd.DataFrame, min_hold: int = 6, window: int = 12) -> pd.DataFrame:
    """For each pair, the most recent month after which the leader has stayed ahead
    for at least `min_hold` months through the present.

    Uses a trailing `window`-month pooled share: monthly counts for niche skills are
    small, and a 3-month window lets noise flip the leader back and forth.
    """
    n = sm.pivot(index="month", columns="skill", values="n")
    posts = sm.pivot(index="month", columns="skill", values="posts")
    wide = n.rolling(window, min_periods=window).sum() / posts.rolling(window, min_periods=window).sum()
    wide = wide.dropna()
    rows = []
    for a, b in CROSSOVER_PAIRS:
        diff = (wide[a] - wide[b]).to_numpy()
        leader, other = (a, b) if diff[-1] > 0 else (b, a)
        sign = np.sign(diff[-1])
        flips = np.where(np.sign(diff) != sign)[0]
        since = 0 if len(flips) == 0 else flips[-1] + 1
        held = len(diff) - since
        rows.append({
            "pair": f"{a} vs {b}", "leader": leader, "trailer": other,
            "since": wide.index[since] if held >= min_hold else pd.NaT,
            "led_whole_period": len(flips) == 0,
            "leader_share_now": wide[leader].iloc[-1], "trailer_share_now": wide[other].iloc[-1],
        })
    return pd.DataFrame(rows)


def run() -> None:
    df = pd.read_parquet(INTERIM / "posts.parquet")
    # The current month's thread is still filling up; its *shares* are fine to use.
    market_monthly(df).to_parquet(PROCESSED / "market_monthly.parquet", index=False)
    sm = skill_monthly(df)
    sm.to_parquet(PROCESSED / "skill_monthly.parquet", index=False)
    pd.concat([movers(df, 1), movers(df, 5)]).to_parquet(PROCESSED / "skill_movers.parquet", index=False)
    crossovers(sm).to_parquet(PROCESSED / "crossovers.parquet", index=False)
    log.info("Wrote trend tables")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
