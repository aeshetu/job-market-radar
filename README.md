# 📡 Job Market Radar

**15 years of tech hiring, mined from 94,821 Hacker News "Who is hiring?" posts (Apr 2011 – Oct 2026).**

An end-to-end data science / ML project: API ingestion → tested NLP extraction → statistical trend
analysis → forecasting with calibrated uncertainty → sentence-embedding clustering → salary modeling →
an interactive Streamlit dashboard that refreshes itself every month via GitHub Actions.

> **Live dashboard: [job-market-radar.streamlit.app](https://job-market-radar.streamlit.app/)**

---

## Key findings

| | |
|---|---|
| **AI went from niche to default** | 6.9% of posts mentioned AI in Nov 2020 – Oct 2021; **51.7%** do in the last 12 months. "AI agents" went from 0% to 14.2%. |
| **ML engineer ↑, "data scientist" ↓** | ML/AI-engineer titles grew from 0.4% of posts (2011) to **~13%** (2025–26), while data-scientist/analyst titles fell from a 6.8% peak (2018) to **2.5%**. |
| **Remote peaked and is receding** | Posts allowing remote: ~20% before 2019 → **82%** in 2022 → 54% in 2026. |
| **Pay transparency arrived** | Posts listing a salary: 1% in 2014 → **27%** in 2026, tracking the CO/NYC/CA/WA disclosure laws of 2021–23. |
| **AI companies reshaped the board** | The "AI / LLM products" cluster (found unsupervised from post text) grew from **1.6%** of posts in 2017–19 to **22.4%** in the last 12 months. Mobile-app companies fell from 4.5% to 1.4% (peak: 2013). Robotics & hardware is at an all-time high. |
| **Crossovers** | TypeScript passed JavaScript (Oct 2020), PyTorch passed TensorFlow (Mar 2022), Snowflake passed Redshift (Aug 2023), Rust passed C++ (Jun 2024), LLMs passed general ML (Jun 2025). |
| **Location and level drive pay, not most skills** | vs. remote: SF **+22%**, NYC **+18%**; staff/principal **+13%**, senior **+7%**, junior **−12%**. Only 3 of 46 skills keep a significant premium after controls and FDR correction. Advertised pay grew **4.8%/yr**. |

---

## What makes it more than a dashboard

**1. Extraction that's tested, not assumed.** Naive keyword matching is badly wrong on this data: *Go*
("go to our site"), *C* ("Series C"), *R* ("R&D"), *React* ("react quickly"), *Spark* ("spark joy"),
salaries vs. funding rounds ("$10–20M") or MRR. Each skill is a regex with explicit negative test cases
(**73 unit tests**), and random samples of matches were read by hand to check precision. That
review caught, and the fixes removed, lowercase "series C" false positives and salary false
positives such as MRR figures and "Americans making $40,000 or less".

**2. Statistics that survive multiple testing.** ~90 skills are compared at once, so "rising" skills
are tested with two-proportion z-tests under **Benjamini–Hochberg FDR control**.

**3. A forecast evaluation that doesn't flatter itself.**
- Rolling-origin backtest of 5 models (naive, 6-month mean, damped ETS ×2, ensemble) across 54 skills.
- The model is **selected on 2019–22 origins and scored on untouched 2023+ origins**. The simple 6-month mean won selection and beats naive by 5% on held-out years.

**4. Conformal prediction under distribution shift.** 80% intervals calibrated on 2019–22 covered
only **59%** of 2023+ outcomes. The AI boom broke the exchangeability assumption behind conformal
guarantees. Fixes, evaluated on held-out origins:

| Interval method | Held-out coverage (target 80%) |
|---|---|
| Static split-conformal | 58.9% |
| Recalibrated as data arrives | 63.7% |
| **Sliding 36-month window + per-skill volatility normalization** | **79.1%** |

*Caveat:* the window length was picked from four options on the test period.

**5. Leakage-aware salary modeling.** Companies repost identical ranges month after month, so:
- OLS uses **company-clustered standard errors**.
- Predictive models use **GroupKFold by company** plus a **temporal holdout** (train ≤ 2024, test 2025+).
- Hand-built features reach R² ≈ 0.24 (log scale). Text embeddings help across companies but not forward in time; see [Salary prediction](#salary-prediction).

**6. Unsupervised industry map, debugged in three iterations.**
- MiniLM sentence embeddings on CPU, clustered with k-means and **named zero-shot** by matching each centroid to the nearest plain-English industry description. Each name sits beside company-deduplicated c-TF-IDF keywords, so you can check it.
- *v1* embedded the whole post, and clusters formed around the metadata header ("London | ONSITE | Full-time") instead of what companies do.
- *v2* embedded only the body, but every job ad shares a strong "we're hiring, great benefits" direction, which pulled generic clusters toward "HR & recruiting".
- *v3* **centers the embeddings** (subtracts the mean vector, as in all-but-the-top post-processing).
- Result: 13 clean industry clusters with name-match similarity ≥ 0.52, and 11 stack/format clusters at ≤ 0.41. These are labelled *General: \<keywords\>* instead of being forced into an industry, with the 0.5 threshold sitting in the empty gap.
- Silhouette is flat (0.022–0.025) for k = 12–32, as usual for text embeddings, so k = 24 was chosen for granularity.

**7. Resume matcher.** Paste a resume, and semantic search returns the closest *current* posts (last 1–3
months, since older HN roles are usually filled), one per company, plus the skills that are common in
those matches but missing from your text.

---

## Salary prediction

4,521 USD posts from 1,813 companies (2016–2026), target = log(salary midpoint).

| Model | Grouped CV by company, R² (log) | Train ≤2024 → test 2025+, R² (log) | MAE (CV) |
|---|---|---|---|
| Median by year (baseline) | 0.170 | −0.015 | $29.9k |
| Gradient boosting, hand-built features | 0.157 | 0.111 | $31.3k |
| Ridge, hand-built features | 0.240 | 0.110 | $29.7k |
| Ridge, text embedding only (32 PCs) | 0.250 | 0.055 | $29.4k |
| Ridge, features + raw 384-d embedding | 0.165 | −0.034 | $32.2k |
| **Ridge, features + 32-PC text embedding** | **0.273** | 0.107 | **$29.2k** |

Takeaways:
- **Advertised pay is hard to predict from a post.** Even the best model leaves most of the variance unexplained, and an honest write-up says so.
- **Raw embeddings hurt.** 384 dense dimensions swamp a single ridge penalty. Compressing them to 32 principal components, fit *inside each training fold*, fixes that.
- **Text helps across companies but not forward in time.** It lifts grouped-CV R² from 0.240 to 0.273, but on the 2025+ holdout it's flat (0.107 vs 0.110). Whatever pay signal the text carries doesn't transfer to the newest postings.
- **Gradient boosting loses to ridge** on these sparse binary features.
- PCA size was picked from {16, 32, 64} on the CV folds, so the temporal holdout is the cleaner check.

---

## Architecture

```
HN Algolia API ──► ingest.py ──► data/raw/*.json.gz             (cached; 2 newest months re-fetched)
                                    │
                     features.py ◄──┘  parse.py · skills.py · salary.py   (multiprocess, ~60s)
                          │
        ┌─────────────────┼──────────────────┬─────────────────────┐
   trends.py         forecast.py         embed.py (MiniLM)     salary_model.py
   movers, FDR,      backtest +          incremental,          OLS (clustered SE),
   crossovers        conformal PIs       checkpointed          ridge / GBM / text
        │                 │                  │                     │
        │                 │             clusters.py                │
        │                 │   centering · k-means · zero-shot names│
        │                 │        c-TF-IDF · UMAP · match index   │
        └─────────────────┴──────────┬───────┴─────────────────────┘
                                     ▼
                     data/processed/*.parquet|json  (small, committed)
                                     ▼
                       app/streamlit_app.py (dashboard)

GitHub Actions: CI on every push · monthly refresh on the 5th, commits new tables
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[embed,dev]"

pytest -q                                   # 73 tests
python -m radar.pipeline                    # full pipeline, ~1 hour on a laptop CPU (embeddings dominate)
python -m radar.pipeline --skip-embed       # ~2 min, everything except clusters/matcher
streamlit run app/streamlit_app.py
```

**Deploy free:** push to GitHub → [share.streamlit.io](https://share.streamlit.io) → *New app* → pick
`app/streamlit_app.py`. `requirements.txt` is already set up. The monthly workflow keeps the data current.

## Repo layout

```
src/radar/      pipeline package (ingest, parse, skills, salary, features, trends,
                forecast, embed, clusters, salary_model, pipeline)
app/            Streamlit dashboard
tests/          extraction edge cases + dashboard smoke test
data/processed/ committed outputs the dashboard reads
.github/        CI + scheduled monthly refresh
```

## Limitations

- HN skews toward startups, SF/NYC and engineering, so this is not the whole labor market.
- Shares measure what companies *write*, not headcount. One post can cover many openings, and companies repost monthly.
- Advertised ranges are not offers, and *which* companies disclose pay changed when the laws changed.
- Rule-based extraction favors precision over recall.

## Data

Public posts via the [HN Algolia API](https://hn.algolia.com/api). Full post text isn't committed to
this repo; the pipeline downloads it. The dashboard tables keep only aggregates, each post's first
(header) line, and links back to the original post on Hacker News.
