"""12-month demand forecasts per skill, with an honest rolling-origin backtest.

Candidates
  naive    last observed monthly share, carried forward
  mean6    mean of the last 6 months
  ets      damped-trend ETS on the logit of the raw monthly share
  ets_s3   damped-trend ETS on the logit of a trailing 3-month pooled share
  ens      average of ets_s3 and mean6

Protocol
  * Rolling origins every 6 months from 2019; horizons 1-12 months; MAE in
    percentage points (pp) against the raw monthly share.
  * Model selection uses origins before CALIB_END only. Reported scores come from
    the later, untouched origins, so the selection step can't flatter itself.

Prediction intervals (conformal, logit scale)
  The score is |logit error| / sigma, where sigma is the skill's own volatility
  (std of month-over-month logit changes in the 12 months before the origin).
  At each origin, the 80th percentile of scores whose target month is already
  observed, within the last CALIB_WINDOW months, sets the half-width per horizon.
  The variants we evaluated are kept in the metrics:
      static       pooled scores, calibrated once before 2023           -> under-covers
      expanding    pooled scores, recalibrated at each origin           -> under-covers
      normalized   volatility-normalized, sliding window (shipped)      -> ~nominal
  Under-coverage of the first two is the AI boom showing up as a regime shift:
  conformal guarantees assume exchangeable errors, and 2023+ is not like 2019-22.
  Caveat: the window length was picked from {all, 36, 24, 12} on the test origins.
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from .config import PROCESSED

log = logging.getLogger(__name__)

HORIZON = 12
START = pd.Timestamp("2013-01-01")      # earlier months have < 250 posts
CALIB_END = pd.Timestamp("2023-01-01")  # origins before this select the model
CALIB_WINDOW = 36                       # months of observed errors used to calibrate intervals
VOL_WINDOW = 12
MIN_RECENT_SHARE = 0.01                 # forecast skills with >= 1% share over the last 2 years
COVERAGE = 0.80
EPS = 1e-3
MODELS = ("naive", "mean6", "ets", "ets_s3", "ens")


def _logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def _ets(y: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        res = ETSModel(_logit(y), error="add", trend="add", damped_trend=True).fit(disp=False, maxiter=200)
        return expit(res.forecast(h))


def predict_all(n: np.ndarray, posts: np.ndarray, h: int) -> dict[str, np.ndarray]:
    share = n / posts
    s3 = pd.Series(n).rolling(3, min_periods=1).sum() / pd.Series(posts).rolling(3, min_periods=1).sum()
    out = {
        "naive": np.repeat(share[-1], h),
        "mean6": np.repeat(share[-6:].mean(), h),
        "ets": _ets(share, h),
        "ets_s3": _ets(s3.to_numpy(), h),
    }
    out["ens"] = (out["ets_s3"] + out["mean6"]) / 2
    return out


def volatility(share: pd.Series) -> float:
    """Std of month-over-month logit changes over the last VOL_WINDOW months."""
    dz = pd.Series(_logit(share.to_numpy())).diff().iloc[-VOL_WINDOW:]
    return float(max(dz.std(), 1e-3))


def backtest(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    last = max(s.index[-1] for s in series.values())
    origins = pd.date_range("2019-01-01", last - pd.DateOffset(months=HORIZON - 1), freq="6MS")
    for skill, d in series.items():
        for origin in origins:
            train, test = d[d.index < origin], d[d.index >= origin].iloc[:HORIZON]
            if len(test) < HORIZON:
                continue
            preds = predict_all(train["n"].to_numpy(), train["posts"].to_numpy(), HORIZON)
            sigma = volatility(train["n"] / train["posts"])
            actual = (test["n"] / test["posts"]).to_numpy()
            for i in range(HORIZON):
                rows.append({"skill": skill, "origin": origin, "h": i + 1, "actual": actual[i],
                             "target": origin + pd.DateOffset(months=i), "sigma": sigma,
                             **{m: preds[m][i] for m in MODELS}})
    return pd.DataFrame(rows)


def _quantiles(scores: pd.DataFrame, col: str) -> np.ndarray:
    return (scores.groupby("h")[col].quantile(COVERAGE)
            .reindex(range(1, HORIZON + 1)).ffill().bfill().to_numpy())


def _covered(rows: pd.DataFrame, model: str, hw: np.ndarray) -> pd.Series:
    z = _logit(rows[model]).to_numpy()
    return (expit(z - hw) <= rows["actual"]) & (rows["actual"] <= expit(z + hw))


def evaluate(bt: pd.DataFrame) -> tuple[str, dict]:
    calib, test = bt[bt["origin"] < CALIB_END], bt[bt["origin"] >= CALIB_END]

    def mae(frame):
        return {m: round(float(100 * (frame[m] - frame["actual"]).abs().mean()), 4) for m in MODELS}

    mae_calib, mae_test = mae(calib), mae(test)
    best = min(mae_calib, key=mae_calib.get)

    bt = bt.assign(err=(_logit(bt["actual"]) - _logit(bt[best])).abs())
    bt["score"] = bt["err"] / bt["sigma"]
    test = bt[bt["origin"] >= CALIB_END]
    h_idx = test["h"].to_numpy() - 1

    static = _covered(test, best, _quantiles(bt[bt["origin"] < CALIB_END], "err")[h_idx])
    expanding, normalized = [], []
    for origin, rows in test.groupby("origin"):
        seen = bt[bt["target"] < origin]
        idx = rows["h"].to_numpy() - 1
        expanding.append(_covered(rows, best, _quantiles(seen, "err")[idx]))
        recent = seen[seen["target"] >= origin - pd.DateOffset(months=CALIB_WINDOW)]
        normalized.append(_covered(rows, best, _quantiles(recent, "score")[idx] * rows["sigma"].to_numpy()))
    normalized = pd.concat(normalized)

    metrics = {
        "protocol": "rolling origin every 6 months; select model on origins < "
                    f"{CALIB_END:%Y-%m}, report on origins >= {CALIB_END:%Y-%m}",
        "n_series": int(bt["skill"].nunique()),
        "n_origins_calib": int(calib["origin"].nunique()),
        "n_origins_test": int(test["origin"].nunique()),
        "horizon_months": HORIZON,
        "mae_pp_calib": mae_calib,
        "mae_pp_test": mae_test,
        "selected_model": best,
        "test_improvement_vs_naive": round(1 - mae_test[best] / mae_test["naive"], 4),
        "mae_pp_test_by_horizon": {
            m: (100 * (test[m] - test["actual"]).abs()).groupby(test["h"]).mean().round(3).tolist()
            for m in dict.fromkeys((best, "naive", "mean6", "ets"))},
        "interval_nominal_coverage": COVERAGE,
        "interval_test_coverage": {
            "static_pooled": round(float(static.mean()), 4),
            "expanding_pooled": round(float(pd.concat(expanding).mean()), 4),
            "normalized_sliding (shipped)": round(float(normalized.mean()), 4),
        },
        "interval_test_coverage_by_horizon": normalized.groupby(test.loc[normalized.index, "h"])
                                                       .mean().round(3).tolist(),
        "calibration_window_months": CALIB_WINDOW,
    }
    return best, metrics, bt


def run() -> None:
    sm = pd.read_parquet(PROCESSED / "skill_monthly.parquet")
    sm = sm[sm["month"] >= START]
    recent = sm[sm["month"] > sm["month"].max() - pd.DateOffset(months=24)].groupby("skill")["share"].mean()
    keep = recent[recent >= MIN_RECENT_SHARE].index
    series = {s: g.set_index("month")[["n", "posts"]] for s, g in sm[sm["skill"].isin(keep)].groupby("skill")}

    best, metrics, bt = evaluate(backtest(series))
    log.info("Selected %s | test MAE %s | coverage %s",
             best, metrics["mae_pp_test"], metrics["interval_test_coverage"])

    last = max(d.index[-1] for d in series.values())
    q = _quantiles(bt[bt["target"] >= last - pd.DateOffset(months=CALIB_WINDOW)], "score")
    fc = []
    for skill, d in series.items():
        mean = predict_all(d["n"].to_numpy(), d["posts"].to_numpy(), HORIZON)[best]
        hw = q * volatility(d["n"] / d["posts"])
        idx = pd.date_range(d.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        fc.append(pd.DataFrame({"month": idx, "skill": skill, "mean": mean,
                                "lo": expit(_logit(mean) - hw), "hi": expit(_logit(mean) + hw)}))
    pd.concat(fc).to_parquet(PROCESSED / "skill_forecast.parquet", index=False)
    (PROCESSED / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
