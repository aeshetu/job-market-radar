"""Job Market Radar: Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
Reads only the small precomputed tables in data/processed/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="Job Market Radar", page_icon="📡", layout="wide")

ACCENT = "#2563eb"
PALETTE = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#ef4444", "#0ea5e9", "#eab308", "#64748b"]
EVENTS = [
    ("2020-03-01", "COVID"),
    ("2022-03-01", "Rate hikes"),
    ("2022-11-01", "ChatGPT"),
]
HN_ITEM = "https://news.ycombinator.com/item?id={}"


# ----------------------------------------------------------------------------- data
@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@st.cache_data
def load_json(name: str) -> dict:
    p = PROCESSED / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def has(name: str) -> bool:
    return (PROCESSED / f"{name}.parquet").exists()


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def add_events(fig: go.Figure, start: pd.Timestamp | None = None) -> go.Figure:
    for date, label in EVENTS:
        d = pd.Timestamp(date)
        if start is not None and d < start:
            continue
        fig.add_vline(x=d, line_width=1, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=d, y=1.02, yref="paper", text=label, showarrow=False,
                           font=dict(size=11, color="#64748b"))
    return fig


def style(fig: go.Figure, height: int = 380, pct_axis: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.15, title=None),
                      hovermode="x unified", colorway=PALETTE)
    if pct_axis:
        fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title=None)
    return fig


market = load("market_monthly")
skills_m = load("skill_monthly")
latest = market["month"].max()

# --------------------------------------------------------------------------- header
st.title("📡 Job Market Radar")
st.markdown(
    f"**{int(market['posts'].sum()):,} job posts** from every Hacker News *“Who is hiring?”* thread, "
    f"{market['month'].min():%B %Y} – {latest:%B %Y}, mined for skills, salaries, remote policy and "
    "industry. Every number below is computed from the raw posts by an open, tested pipeline."
)

tab_market, tab_skills, tab_ind, tab_sal, tab_match, tab_methods = st.tabs(
    ["Market", "Skills", "Industries", "Salaries", "Resume match", "Methods"])

# =========================================================================== MARKET
with tab_market:
    yr = market.set_index("month")
    last12 = yr[yr.index > latest - pd.DateOffset(months=12)]
    peak = yr["posts"].rolling(12).mean().max()
    ai_now = skills_m[(skills_m["skill"] == "AI (any mention)") & (skills_m["month"].isin(last12.index))]
    ai_then = skills_m[(skills_m["skill"] == "AI (any mention)")
                       & skills_m["month"].between(latest - pd.DateOffset(months=71), latest - pd.DateOffset(months=60))]

    c = st.columns(5)
    c[0].metric("Posts / month (last 12m)", f"{last12['posts'].mean():.0f}",
                f"{last12['posts'].mean() / peak - 1:+.0%} vs. peak", delta_color="off")
    c[1].metric("Mention AI", pct(ai_now["n"].sum() / ai_now["posts"].sum()),
                f"from {pct(ai_then['n'].sum() / ai_then['posts'].sum())} 5y ago", delta_color="off")
    c[2].metric("Allow remote", pct(last12["remote_share"].mean()),
                f"peak {pct(yr['remote_share'].rolling(6).mean().max())}", delta_color="off")
    c[3].metric("List a salary", pct(last12["salary_share"].mean()),
                f"{pct(yr.loc[yr.index.year == 2019, 'salary_share'].mean())} in 2019", delta_color="off")
    med = (last12["median_salary_usd"] * last12["salary_n_usd"]).sum() / last12["salary_n_usd"].sum()
    c[4].metric("Median advertised (USD)", f"${med / 1000:.0f}k", "midpoint of range", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Hiring volume")
        m = market.copy()
        m["12-month average"] = m["posts"].rolling(12, min_periods=6).mean()
        fig = go.Figure()
        fig.add_bar(x=m["month"], y=m["posts"], name="Posts per month", marker_color="#cbd5e1")
        fig.add_scatter(x=m["month"], y=m["12-month average"], name="12-month average",
                        line=dict(color=ACCENT, width=3))
        st.plotly_chart(add_events(style(fig)), width="stretch")
        st.caption("A rough barometer of startup hiring. The 2021 boom and the 2022–23 "
                   "downturn are both visible. The current month is still collecting posts.")
    with right:
        st.subheader("Remote, hybrid, onsite")
        wm = market.set_index("month")[["remote_share", "hybrid_share", "onsite_share"]].rolling(3, min_periods=1).mean()
        wm.columns = ["Remote allowed", "Hybrid", "Onsite only"]
        fig = px.area(wm.reset_index().melt("month", var_name="Mode", value_name="Share"),
                      x="month", y="Share", color="Mode",
                      color_discrete_sequence=[ACCENT, "#a855f7", "#f97316"])
        st.plotly_chart(add_events(style(fig, pct_axis=True)), width="stretch")
        st.caption("Share of all posts (3-month smoothing); the remainder don't say. "
                   "“Remote allowed” includes “remote or onsite”.")

    left, right = st.columns(2)
    with left:
        st.subheader("Which roles are being hired")
        roles = {"role_ml_ai": "ML / AI engineer", "role_data_science": "Data scientist / analyst",
                 "role_data_engineering": "Data engineer", "role_devops_sre": "DevOps / SRE",
                 "role_security": "Security", "role_mobile": "Mobile"}
        pick = st.multiselect("Roles", list(roles.values()),
                              default=["ML / AI engineer", "Data scientist / analyst", "Data engineer"])
        r = market.set_index("month")[list(roles)].rename(columns=roles).rolling(6, min_periods=3).mean()
        fig = px.line(r[pick].reset_index().melt("month", var_name="Role", value_name="Share"),
                      x="month", y="Share", color="Role")
        fig.update_traces(line_width=3)
        st.plotly_chart(add_events(style(fig, pct_axis=True)), width="stretch")
        st.caption("Share of posts whose job titles include the role (6-month smoothing). "
                   "A post can list several roles.")
    with right:
        st.subheader("Pay transparency")
        s = market.set_index("month")["salary_share"].rolling(3, min_periods=1).mean()
        fig = px.line(s.reset_index(), x="month", y="salary_share")
        fig.update_traces(line=dict(color="#10b981", width=3), name="Lists a salary")
        for date, label in [("2021-01-01", "Colorado law"), ("2022-11-01", "NYC law"),
                            ("2023-01-01", "CA & WA laws")]:
            fig.add_vline(x=pd.Timestamp(date), line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=pd.Timestamp("2022-06-01"), y=1.02, yref="paper", showarrow=False,
                           text="Pay-transparency laws 2021–23", font=dict(size=11, color="#64748b"))
        st.plotly_chart(style(fig, pct_axis=True), width="stretch")
        st.caption("Share of posts with a parseable annual salary (any currency).")

# =========================================================================== SKILLS
with tab_skills:
    forecast = load("skill_forecast") if has("skill_forecast") else None
    fm = load_json("forecast_metrics")
    all_skills = sorted(skills_m["skill"].unique(), key=str.lower)

    st.subheader("Skill demand over time")
    c1, c2 = st.columns([4, 1])
    chosen = c1.multiselect("Compare skills", all_skills,
                            default=["Python", "TypeScript", "LLMs / GenAI", "Rust", "Ruby"])
    show_fc = c2.toggle("12-month forecast", value=True, disabled=forecast is None)
    since = c2.slider("From", 2011, int(latest.year) - 1, 2015)

    fig = go.Figure()
    for i, sk in enumerate(chosen):
        color = PALETTE[i % len(PALETTE)]
        d = skills_m[(skills_m["skill"] == sk) & (skills_m["month"].dt.year >= since)]
        fig.add_scatter(x=d["month"], y=d["share_smooth"], name=sk, line=dict(color=color, width=3))
        if show_fc and forecast is not None and sk in set(forecast["skill"]):
            f = forecast[forecast["skill"] == sk]
            fig.add_scatter(x=pd.concat([f["month"], f["month"][::-1]]),
                            y=pd.concat([f["hi"], f["lo"][::-1]]), fill="toself",
                            fillcolor=color, opacity=0.15, line=dict(width=0),
                            hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=f["month"], y=f["mean"], name=f"{sk} (forecast)",
                            line=dict(color=color, width=2, dash="dash"), showlegend=False)
    st.plotly_chart(add_events(style(fig, height=460, pct_axis=True), pd.Timestamp(f"{since}-01-01")),
                    width="stretch")
    if fm:
        cov = fm["interval_test_coverage"]["normalized_sliding (shipped)"]
        st.caption(
            "Share of posts mentioning each skill (3-month pooled). Dashed: 12-month forecast with an "
            f"80% conformal interval. In a held-out backtest (2023+) the selected model "
            f"(`{fm['selected_model']}`) beat a naive forecast by {fm['test_improvement_vs_naive']:.0%} "
            f"and the intervals covered {cov:.0%} of outcomes. See *Methods*.")

    st.subheader("Biggest movers")
    movers = load("skill_movers")
    c1, c2 = st.columns([1, 3])
    window = c1.radio("Compare last 12 months with", ["1y", "5y"],
                      format_func=lambda w: "a year earlier" if w == "1y" else "five years earlier")
    only_sig = c1.checkbox("Only statistically significant (FDR q < 0.05)", value=True)
    mv = movers[movers["window"] == window]
    if only_sig:
        mv = mv[mv["significant"]]
    top = pd.concat([mv.nlargest(10, "delta_pp"), mv.nsmallest(10, "delta_pp")]).drop_duplicates("skill")
    top = top.sort_values("delta_pp")
    fig = px.bar(top, x="delta_pp", y="skill", orientation="h",
                 color=np.where(top["delta_pp"] > 0, "Rising", "Falling"),
                 color_discrete_map={"Rising": "#10b981", "Falling": "#ef4444"},
                 custom_data=["share_then", "share_now", "q"])
    fig.update_traces(hovertemplate="%{y}: %{customdata[0]:.1%} → %{customdata[1]:.1%}"
                                    "<br>q = %{customdata[2]:.3g}<extra></extra>")
    fig.update_layout(xaxis_title="Change in share of posts (percentage points)", yaxis_title=None,
                      legend_title=None)
    with c2:
        st.plotly_chart(style(fig, height=520), width="stretch")
        if len(mv):
            st.caption(f"{mv['period_then'].iloc[0]} vs. {mv['period_now'].iloc[0]}. Two-proportion z-tests "
                       "with Benjamini–Hochberg correction across all skills.")

    st.subheader("Who overtook whom")
    xo = load("crossovers")
    xo = xo.assign(**{
        "Now leading": xo["leader"],
        "Ahead since": xo["since"].dt.strftime("%b %Y").where(~xo["led_whole_period"], "always"),
        "Share now (leader vs. other)": [f"{a:.1%} vs {b:.1%}" for a, b in
                                         zip(xo["leader_share_now"], xo["trailer_share_now"])],
    })
    st.dataframe(xo[["pair", "Now leading", "Ahead since", "Share now (leader vs. other)"]]
                 .rename(columns={"pair": "Matchup"}), hide_index=True, width="stretch")
    st.caption("Leader on a trailing 12-month share, ahead continuously through today.")

# ======================================================================= INDUSTRIES
with tab_ind:
    if not has("clusters"):
        st.info("Run the embedding and clustering steps to populate this tab.")
    else:
        info = load("clusters")
        cm = load("cluster_monthly")
        n_ind = int((~info["name"].str.startswith("General")).sum())
        st.markdown(
            f"Each post body was embedded with a sentence-transformer and grouped into **{len(info)} clusters**. "
            f"**{n_ind}** were named *zero-shot* by matching the centroid to the nearest plain-English industry "
            "description. The rest weren't close to any industry, so they're labelled *General* with their "
            "keywords instead of being forced into a category.")
        show_general = st.toggle("Include *General* (non-industry) clusters", value=False)
        if not show_general:
            cm = cm[~cm["name"].str.startswith("General")]

        wide = cm.pivot_table(index="month", columns="name", values="share", fill_value=0)
        wide = wide.rolling(6, min_periods=3).mean()
        recent = wide.iloc[-12:].mean()
        before = wide[(wide.index.year >= 2017) & (wide.index.year <= 2019)].mean()
        change = (recent - before).sort_values()

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Industry share over time")
            default = list(change.index[-3:][::-1]) + list(change.index[:2])
            pick = st.multiselect("Industries", sorted(wide.columns), default=default)
            fig = px.line(wide[pick].reset_index().melt("month", var_name="Industry", value_name="Share"),
                          x="month", y="Share", color="Industry")
            fig.update_traces(line_width=3)
            st.plotly_chart(add_events(style(fig, height=430, pct_axis=True)), width="stretch")
        with right:
            st.subheader("Change vs. 2017–19")
            ch = change.rename("pp").mul(100).reset_index().rename(columns={"name": "Industry"})
            ch = pd.concat([ch.head(6), ch.tail(6)])
            fig = px.bar(ch, x="pp", y="Industry", orientation="h",
                         color=np.where(ch["pp"] > 0, "Up", "Down"),
                         color_discrete_map={"Up": "#10b981", "Down": "#ef4444"})
            fig.update_layout(xaxis_title="Percentage points of all posts", yaxis_title=None, showlegend=False)
            st.plotly_chart(style(fig, height=430), width="stretch")

        st.subheader("Map of postings")
        mp = load("cluster_map")
        years = st.slider("Years", int(mp["year"].min()), int(mp["year"].max()),
                          (int(mp["year"].min()), int(mp["year"].max())))
        sub = mp[mp["year"].between(*years)]
        fig = px.scatter(sub, x="x", y="y", color="name", hover_name="header",
                         hover_data={"x": False, "y": False, "year": True, "name": True},
                         opacity=0.55, color_discrete_sequence=px.colors.qualitative.Alphabet)
        fig.update_traces(marker_size=4)
        fig.update_layout(height=620, xaxis_visible=False, yaxis_visible=False,
                          legend=dict(title=None, font=dict(size=11)), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("UMAP projection of a year-stratified sample of embeddings. Nearby points are "
                   "semantically similar posts. Hover to read them.")

        with st.expander("Cluster details: keywords and example posts"):
            st.dataframe(info[["name", "size", "keywords", "examples"]].sort_values("size", ascending=False),
                         hide_index=True, width="stretch")

# ========================================================================= SALARIES
with tab_sal:
    prem = load("salary_premiums")
    sm_ = load_json("salary_metrics")
    calc = load_json("salary_calculator")

    st.markdown(
        f"Fit on **{sm_['n_posts']:,} US-dollar posts** from {sm_['n_companies']:,} companies "
        f"({sm_['years'][0]}–{sm_['years'][1]}) in US metros or remote. The model is log(salary midpoint) "
        "regressed on location, seniority, role, skills and year, with **standard errors clustered by "
        "company** because companies repost the same range month after month.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What moves advertised pay")
        groups = {"loc": "Location (vs. remote)", "level": "Seniority", "role": "Role", "skill": "Skill"}
        g = st.radio("Factor", list(groups), format_func=groups.get, horizontal=True)
        p = prem[prem["group"] == g].copy()
        if g == "skill":
            sig_only = st.checkbox("Only skills significant after FDR correction", value=False)
            if sig_only:
                p = p[p["q"] < 0.05]
        p = p.sort_values("coef_pct")
        p["sig"] = np.where(p["q"] < 0.05, "Significant (q < 0.05)", "Not significant")
        fig = px.scatter(p, x="coef_pct", y="label", color="sig",
                         error_x=p["ci_hi_pct"] - p["coef_pct"], error_x_minus=p["coef_pct"] - p["ci_lo_pct"],
                         color_discrete_map={"Significant (q < 0.05)": ACCENT, "Not significant": "#94a3b8"},
                         hover_data={"n": True, "q": ":.3f", "sig": False, "label": False})
        fig.add_vline(x=0, line_color="#64748b")
        fig.update_layout(xaxis_title="Premium on advertised salary (%), 95% CI", yaxis_title=None,
                          hovermode="closest", height=max(320, 26 * len(p)))
        st.plotly_chart(style(fig, height=max(320, 26 * len(p))), width="stretch")
        yr_row = prem[prem["group"] == "year"].iloc[0]
        st.caption(f"Holding everything else fixed, advertised salaries grew **{yr_row['coef_pct']:.1f}% "
                   f"per year** (95% CI {yr_row['ci_lo_pct']:.1f}–{yr_row['ci_hi_pct']:.1f}%). "
                   "These are associations in posted pay, not causal effects of learning a skill.")

    with right:
        st.subheader("Salary estimator")
        loc = st.selectbox("Location", calc["locations"], index=calc["locations"].index("SF Bay Area")
                           if "SF Bay Area" in calc["locations"] else 0)
        lv = st.multiselect("Level", calc["levels"], default=[])
        rl = st.multiselect("Role", calc["roles"], default=["ml_ai"] if "ml_ai" in calc["roles"] else [])
        sk = st.multiselect("Skills", calc["skills"], default=[s for s in ["Python", "PyTorch"] if s in calc["skills"]])
        z = calc["intercept"]
        z += calc["coef"].get(f"loc: {loc}", 0)
        z += sum(calc["coef"].get(f"level: {x}", 0) for x in lv)
        z += sum(calc["coef"].get(f"role: {x}", 0) for x in rl)
        z += sum(calc["coef"].get(f"skill: {x}", 0) for x in sk)
        sd = calc["resid_sd"]
        st.metric(f"Typical advertised midpoint ({calc['base_year']})", f"${np.exp(z) / 1000:,.0f}k")
        st.write(f"80% of similar posts: **\\${np.exp(z - 1.2816 * sd) / 1000:,.0f}k – "
                 f"\\${np.exp(z + 1.2816 * sd) / 1000:,.0f}k**")
        st.caption("From the interpretable OLS model above. The wide range is honest: posted ranges vary "
                   "a lot even for similar-looking roles.")

    st.subheader("How predictable is pay?")
    rows = []
    for split, key in [("Grouped CV (by company)", "group_kfold_by_company"),
                       ("Train ≤2024, test 2025+", "temporal_train_le_2024_test_2025plus")]:
        for model, label in [("baseline", "Median by year"), ("ridge", "Ridge, hand-built features"),
                             ("gbm", "Gradient boosting, hand-built features")]:
            rows.append({"Split": split, "Model": label, **sm_[key][model]})
        for name, label in [("text_only", "Ridge, text embedding only (PCA)"),
                            ("features_plus_raw_text", "Ridge, features + raw 384-d embedding"),
                            ("features_plus_text", "Ridge, features + text embedding (PCA)")]:
            if "ridge_with_text_embeddings" in sm_:
                rows.append({"Split": split, "Model": label, **sm_["ridge_with_text_embeddings"][name][key]})
    perf = pd.DataFrame(rows).rename(columns={"mae_usd": "MAE ($)", "median_ape": "Median % error",
                                              "r2_log": "R² (log)"})
    st.dataframe(perf.style.format({"MAE ($)": "${:,.0f}", "Median % error": "{:.1%}", "R² (log)": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("No company appears in both train and test folds. Otherwise the same company's repeated "
               "posts leak across folds and inflate scores.")

# ===================================================================== RESUME MATCH
with tab_match:
    st.subheader("Which recent posts fit you best?")
    if not has("match_posts"):
        st.info("Run the embedding and clustering steps to enable the matcher.")
    else:
        mp = load("match_posts")

        @st.cache_resource(show_spinner="Loading the embedding model…")
        def model():
            from radar.embed import load_model
            return load_model()

        @st.cache_data
        def match_matrix() -> np.ndarray:
            return np.load(PROCESSED / "match_embeddings.npy").astype(np.float32)

        st.markdown(f"Paste your resume or a short description of what you want. It's compared "
                    f"semantically against the **{len(mp):,} posts from the last 12 months**. "
                    "Nothing is stored.")
        text = st.text_area("Resume / profile", height=180, placeholder=(
            "CS + Data Science senior. Python, PyTorch, scikit-learn, SQL. Built an NLP pipeline over "
            "95k job posts, time-series forecasting with conformal intervals. Looking for ML engineer "
            "or data scientist roles, remote or NYC."))
        c1, c2, c3 = st.columns(3)
        remote_only = c1.checkbox("Remote-friendly only")
        salary_only = c2.checkbox("Lists a salary only")
        k = c3.slider("Results", 5, 30, 12)

        if text.strip():
            try:
                m = model()
            except Exception as exc:  # sentence-transformers not installed
                st.error(f"The embedding model isn't available here ({exc}). "
                         "Install the `embed` extra: `pip install -e .[embed]`.")
                st.stop()
            from radar.embed import encode
            from radar.parse import strip_urls
            from radar.skills import extract_skills

            words = strip_urls(text).split()
            chunks = [" ".join(words[i:i + 180]) for i in range(0, max(len(words), 1), 150)] or [text]
            q = encode(m, chunks).mean(axis=0)
            q /= np.linalg.norm(q)
            sims = match_matrix() @ q
            res = mp.assign(score=sims)
            if remote_only:
                res = res[res["work_mode"] == "remote"]
            if salary_only:
                res = res[res["salary_low"].notna()]
            res = res.nlargest(k, "score")

            mine = set(extract_skills(strip_urls(text)))
            top50 = mp.assign(score=sims).nlargest(50, "score")
            demand = top50["skills"].explode().value_counts()
            gaps = [s for s in demand.index if s not in mine and demand[s] >= 5][:8]
            if mine:
                st.write("**Skills detected in your text:** " + ", ".join(sorted(mine)))
            if gaps:
                st.write("**Common in your 50 closest matches but missing from your text:** "
                         + ", ".join(f"{s} ({demand[s]})" for s in gaps))

            for r in res.itertuples():
                sal = ""
                if pd.notna(r.salary_low):
                    sal = f" · {r.salary_currency} {r.salary_low / 1000:.0f}k–{r.salary_high / 1000:.0f}k"
                overlap = [s for s in r.skills if s in mine]
                st.markdown(
                    f"**[{r.header[:140]}]({HN_ITEM.format(r.post_id)})**  \n"
                    f"<span style='color:#64748b'>{r.month:%b %Y} · {r.work_mode} · match {r.score:.2f}{sal}"
                    f"{' · shared skills: ' + ', '.join(overlap) if overlap else ''}</span>",
                    unsafe_allow_html=True)

# ========================================================================== METHODS
with tab_methods:
    fm = load_json("forecast_metrics")
    sm_ = load_json("salary_metrics")
    cmx = load_json("cluster_metrics")
    st.markdown(f"""
### Pipeline
1. **Ingest:** every monthly *Ask HN: Who is hiring?* thread from the HN Algolia API. Missing or
   near-empty months (posted by stand-in accounts) are recovered by title search. Each top-level
   comment is one post.
2. **Extract:** tested, rule-based parsers for ~90 skills, salary ranges (currency, k-suffixes, EU
   number formats; rejects hourly rates, funding rounds, MRR), remote/hybrid/onsite, visa sponsorship,
   seniority, role titles and metro. Ambiguous terms (*Go*, *C*, *R*, *React*, *Spark*) have explicit
   negative test cases, and samples of matches were read by hand to check precision.
3. **Trends:** monthly shares with Wilson intervals; movers via two-proportion z-tests with
   Benjamini–Hochberg FDR control; crossovers on trailing 12-month shares.
4. **Forecasts:** rolling-origin backtest of 5 models across {fm.get('n_series', '–')} skills. The model is
   selected on pre-2023 origins and scored on untouched 2023+ origins.
5. **Embeddings & clusters:** `all-MiniLM-L6-v2` on CPU over the post body (the metadata header is
   dropped; with it, clusters formed around location and formatting). Embeddings are **centered** to remove
   the shared "this is a job ad" direction, then clustered with k-means (k = {cmx.get('k', '–')}). Names come
   zero-shot from the nearest industry description above a similarity threshold, next to company-deduplicated
   c-TF-IDF keywords and a UMAP map.
6. **Salaries:** OLS with company-clustered SEs for interpretation; ridge, gradient boosting and
   text-embedding ridge for prediction, under company-grouped CV and a temporal holdout.

### Forecast intervals: a regime-shift story
Conformal intervals calibrated on 2019–22 covered only
**{fm.get('interval_test_coverage', {}).get('static_pooled', 0):.0%}** of 2023+ outcomes (target 80%).
The AI boom broke the exchangeability that conformal guarantees rely on. Recalibrating on a sliding
window and normalizing each skill's errors by its own volatility restored coverage to
**{fm.get('interval_test_coverage', {}).get('normalized_sliding (shipped)', 0):.0%}**. Caveat: the
window length was chosen from four candidates on the test period.

### Limitations
* HN skews toward startups, SF/NYC and engineering roles, so this is not the whole labor market.
* Shares measure what companies *write*, not headcount, and one post can cover many openings.
* Advertised salary ranges are not offers, and disclosure itself changed over time with pay-transparency laws.
* Rule-based extraction trades recall for precision; unusual phrasings are missed.
""")
    with st.expander("Raw metrics"):
        st.json({"forecast": fm, "salary": sm_, "clusters": cmx})
