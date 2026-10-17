"""What predicts an advertised salary?

Two complementary models on log(salary midpoint), US-dollar posts in US metros or remote:

  * OLS with company-clustered standard errors -> interpretable "premiums"
    (e.g. "+8% for Rust, holding location, seniority, role and year fixed").
    Clustering matters: one company reposts the same range for months, so
    i.i.d. standard errors would be far too small.
  * Ridge and gradient boosting -> predictive accuracy, evaluated with
    GroupKFold by company (no company in both train and test) and with a
    temporal holdout (train <= 2024, test 2025+).

Premiums are associations in *advertised* pay, not causal effects of learning a skill.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

from .config import INTERIM, PROCESSED, RANDOM_STATE
from .parse import US_METROS

log = logging.getLogger(__name__)

MIN_YEAR = 2016
MIN_SKILL_COUNT = 75
SALARY_RANGE = (30_000, 600_000)
BASE_YEAR = 2026
REF_LOCATION = "Remote"


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    d = df[(df["salary_currency"] == "USD") & (df["year"] >= MIN_YEAR)
           & df["salary_mid"].between(*SALARY_RANGE)].copy()
    d["location"] = np.where(d["metro"].isin(US_METROS), d["metro"],
                             np.where(d["work_mode"] == "remote", "Remote", None))
    d = d[d["location"].notna()].reset_index(drop=True)

    skill_counts = d["skills"].explode().value_counts()
    skills = sorted(skill_counts[skill_counts >= MIN_SKILL_COUNT].index)

    X = pd.DataFrame({"year": d["year"] - BASE_YEAR})
    for loc in sorted(set(d["location"]) - {REF_LOCATION}):
        X[f"loc: {loc}"] = (d["location"] == loc).astype(float)
    for c in sorted(c for c in d.columns if c.startswith("lvl_")):
        X[f"level: {c[4:]}"] = d[c].astype(float)
    for c in sorted(c for c in d.columns if c.startswith("role_")):
        X[f"role: {c[5:]}"] = d[c].astype(float)
    skill_sets = d["skills"].apply(set)
    for s in skills:
        X[f"skill: {s}"] = skill_sets.apply(lambda ss: s in ss).astype(float)
    return d, X, skills


def fit_ols(d: pd.DataFrame, X: pd.DataFrame):
    y = np.log(d["salary_mid"])
    groups = pd.factorize(d["company_key"])[0]
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": groups})


def premiums(res, X: pd.DataFrame) -> pd.DataFrame:
    ci = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index, "coef": res.params.values,
        "ci_lo": ci[0].values, "ci_hi": ci[1].values, "p": res.pvalues.values,
    })
    out = out[out["term"] != "const"].copy()
    out["group"] = out["term"].str.split(":").str[0].replace({"year": "year"})
    out["label"] = out["term"].str.split(": ", n=1).str[-1]
    out["n"] = [int(X[t].sum()) if t != "year" else len(X) for t in out["term"]]
    for col in ("coef", "ci_lo", "ci_hi"):
        out[f"{col}_pct"] = 100 * np.expm1(out[col])
    out["q"] = np.nan
    for g, idx in out.groupby("group").groups.items():
        out.loc[idx, "q"] = multipletests(out.loc[idx, "p"], method="fdr_bh")[1]
    return out.sort_values(["group", "coef"], ascending=[True, False]).reset_index(drop=True)


def _models():
    return {
        "ridge": RidgeCV(alphas=np.logspace(-2, 3, 20)),
        "gbm": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
                                             min_samples_leaf=30, l2_regularization=1.0,
                                             random_state=RANDOM_STATE),
    }


def _score(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y, p = np.exp(y_log), np.exp(pred_log)
    return {"mae_usd": round(float(mean_absolute_error(y, p))),
            "median_ape": round(float(np.median(np.abs(p - y) / y)), 4),
            "r2_log": round(float(r2_score(y_log, pred_log)), 4)}


def evaluate(d: pd.DataFrame, X: pd.DataFrame) -> dict:
    y = np.log(d["salary_mid"]).to_numpy()
    Xn = X.to_numpy()
    groups = d["company_key"].to_numpy()
    year = d["year"].to_numpy()

    oof = {name: np.zeros(len(y)) for name in ("baseline", *_models())}
    for tr, te in GroupKFold(n_splits=5).split(Xn, y, groups):
        med = pd.Series(y[tr]).groupby(year[tr]).median()
        oof["baseline"][te] = pd.Series(year[te]).map(med).fillna(np.median(y[tr])).to_numpy()
        for name, model in _models().items():
            oof[name][te] = model.fit(Xn[tr], y[tr]).predict(Xn[te])

    tr, te = year <= 2024, year >= 2025
    temporal = {"baseline": _score(y[te], np.repeat(np.median(y[tr & (year == 2024)]), te.sum()))}
    for name, model in _models().items():
        temporal[name] = _score(y[te], model.fit(Xn[tr], y[tr]).predict(Xn[te]))

    return {
        "n_posts": int(len(d)), "n_companies": int(d["company_key"].nunique()),
        "n_features": int(X.shape[1]), "years": [int(year.min()), int(year.max())],
        "group_kfold_by_company": {k: _score(y, v) for k, v in oof.items()},
        "temporal_train_le_2024_test_2025plus": {"n_test": int(te.sum()), **temporal},
    }


TEXT_PCA = 32


def _text_pipeline(n_hand: int, n_text: int, pca: int | None):
    """Scale hand features; optionally compress the embedding with PCA, which is
    fit inside each training fold. Raw 384-d embeddings swamp a single ridge
    penalty and hurt; ~32 principal components help."""
    text = make_pipeline(PCA(pca, random_state=RANDOM_STATE), StandardScaler()) if pca else StandardScaler()
    ct = ColumnTransformer([("hand", StandardScaler(), list(range(n_hand))),
                            ("text", text, list(range(n_hand, n_hand + n_text)))])
    return make_pipeline(ct, RidgeCV(alphas=np.logspace(-1, 4, 30)))


def evaluate_text(d: pd.DataFrame, X: pd.DataFrame) -> dict | None:
    """Does the posting text (MiniLM embedding) add signal beyond hand-built features?

    Same splits as `evaluate`. PCA dimension was chosen from {16, 32, 64} on the
    grouped CV folds, a mild selection effect; the temporal holdout is the cleaner check.
    """
    from . import embed
    if not embed.CACHE.exists():
        return None
    ids, emb = embed.load()
    pos = pd.Series(np.arange(len(ids)), index=ids).reindex(d["post_id"])
    if pos.isna().any():
        return None
    E = emb[pos.to_numpy().astype(int)]
    y = np.log(d["salary_mid"]).to_numpy()
    groups, year = d["company_key"].to_numpy(), d["year"].to_numpy()
    yr = X[["year"]].to_numpy()
    configs = {
        "text_only": (yr, TEXT_PCA),
        "features_plus_raw_text": (X.to_numpy(), None),
        "features_plus_text": (X.to_numpy(), TEXT_PCA),
    }

    out = {}
    tr_t, te_t = year <= 2024, year >= 2025
    for name, (H, pca) in configs.items():
        F = np.hstack([H, E])
        make = lambda: _text_pipeline(H.shape[1], E.shape[1], pca)  # noqa: E731
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(F, y, groups):
            oof[te] = make().fit(F[tr], y[tr]).predict(F[te])
        temporal = make().fit(F[tr_t], y[tr_t]).predict(F[te_t])
        out[name] = {"group_kfold_by_company": _score(y, oof),
                     "temporal_train_le_2024_test_2025plus": _score(y[te_t], temporal)}
    out["text_pca_components"] = TEXT_PCA
    return out


def run() -> None:
    df = pd.read_parquet(INTERIM / "posts.parquet")
    d, X, skills = build_dataset(df)
    res = fit_ols(d, X)
    prem = premiums(res, X)
    prem.to_parquet(PROCESSED / "salary_premiums.parquet", index=False)

    metrics = evaluate(d, X)
    metrics["ols_r2_in_sample"] = round(float(res.rsquared), 4)
    text = evaluate_text(d, X)
    if text:
        metrics["ridge_with_text_embeddings"] = text
    (PROCESSED / "salary_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Everything the dashboard's salary calculator needs, with no model pickles.
    calc = {
        "intercept": float(res.params["const"]), "base_year": BASE_YEAR,
        "coef": {k: float(v) for k, v in res.params.items() if k != "const"},
        "resid_sd": float(np.sqrt(res.scale)),
        "locations": [REF_LOCATION] + sorted(c[5:] for c in X.columns if c.startswith("loc: ")),
        "levels": [c[7:] for c in X.columns if c.startswith("level: ")],
        "roles": [c[6:] for c in X.columns if c.startswith("role: ")],
        "skills": skills,
    }
    (PROCESSED / "salary_calculator.json").write_text(json.dumps(calc, indent=2))
    d[["post_id", "month", "company", "location", "salary_low", "salary_high", "salary_mid"]].to_parquet(
        PROCESSED / "salary_posts.parquet", index=False)
    log.info("Salary model: %s", metrics["group_kfold_by_company"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
