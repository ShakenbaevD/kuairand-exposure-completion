"""Personalized vs random exposure and video completion (KuaiRand-1K)."""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import norm

log = logging.getLogger("pipeline")


def setup_logging(log_file):
    """Write log messages both to the console and to a file."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(log_file, mode="w", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


def load_data(cfg):
    """Read the three logs and attach user activity (08-21.04)."""
    c, d = cfg["columns"], cfg["data"]
    use = [c["user_id"], c["play_time_ms"], c["duration_ms"], c["tab"]]
    rename = {c["user_id"]: "user_id", c["play_time_ms"]: "play_time_ms",
              c["duration_ms"]: "duration_ms", c["tab"]: "tab"}

    # User activity = number of interactions in the earlier period.
    act = pd.read_csv(d["activity_log"], usecols=[c["user_id"]])
    act = act.rename(columns=rename)
    activity = act.groupby("user_id").size().rename("activity").reset_index()

    # Group is set by the source file: personalized -> is_rand = 0, random -> is_rand = 1
    pers = pd.read_csv(d["personalized_log"], usecols=use).rename(columns=rename)
    pers["is_rand"] = 0
    rnd = pd.read_csv(d["random_log"], usecols=use).rename(columns=rename)
    rnd["is_rand"] = 1

    df = pd.concat([pers, rnd], ignore_index=True)
    df = df.merge(activity, on="user_id", how="left")
    df["activity"] = df["activity"].fillna(0)  # no records before = activity 0
    return df


def clean_and_prepare(df):
    """Count missing values, apply filters, create model variables."""
    model_vars = ["play_time_ms", "duration_ms", "tab"]
    missing = {v: float(df[v].isna().mean()) for v in model_vars}
    n_raw = len(df)

    df = df.dropna(subset=model_vars)
    df = df[(df["duration_ms"] > 0) & (df["play_time_ms"] >= 0)].copy()
    log.info("Rows: %d raw -> %d after filters and complete-case removal",
             n_raw, len(df))

    # IMPORTANT: in the data is_rand = 1 means RANDOM exposure.
    # In the model exposure = 1 means PERSONALIZED (random is the reference).
    df["exposure"] = 1 - df["is_rand"]
    df["duration_s"] = df["duration_ms"] / 1000
    df["watch_time_s"] = df["play_time_ms"] / 1000
    df["log_duration"] = np.log(df["duration_s"])
    df["log_activity"] = np.log1p(df["activity"])
    return df.reset_index(drop=True), missing


def describe(df, thr):
    """Completion rate and watch time by group (descriptive results)."""
    d = df.copy()
    d["completed"] = (d["play_time_ms"] >= thr * d["duration_ms"]).astype(int)
    g = d.groupby("exposure").agg(n=("completed", "size"),
                                  completion_rate=("completed", "mean"),
                                  mean_watch_time_s=("watch_time_s", "mean"))
    log.info("Descriptive results (exposure 1 = personalized, 0 = random):\n%s",
             g.round(4).to_string())
    out = {"group_n": {int(k): int(v) for k, v in g["n"].items()}}
    if {0, 1} <= set(g.index):
        out["delta_cr_pp"] = float((g.loc[1, "completion_rate"]
                                    - g.loc[0, "completion_rate"]) * 100)
        out["delta_wt_s"] = float(g.loc[1, "mean_watch_time_s"]
                                  - g.loc[0, "mean_watch_time_s"])
        log.info("Delta completion rate: %.2f pp | Delta watch time: %.2f s",
                 out["delta_cr_pp"], out["delta_wt_s"])
    return out


def balance_report(df):
    """Show how comparable the two groups are."""
    a = df.loc[df["exposure"] == 1, "log_duration"]
    b = df.loc[df["exposure"] == 0, "log_duration"]
    pooled = np.sqrt((a.var() + b.var()) / 2)
    smd = float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")
    log.info("Standardized difference of mean log(duration_s): %.3f", smd)
    tabs = pd.crosstab(df["tab"], df["exposure"], normalize="columns").round(3)
    log.info("Share of each tab within group:\n%s", tabs.to_string())
    return smd


def fit_model(df, thr):
    """Logistic regression with standard errors clustered by user_id."""
    d = df.copy()
    d["completed"] = (d["play_time_ms"] >= thr * d["duration_ms"]).astype(int)
    formula = "completed ~ exposure + log_duration + log_activity + C(tab)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.logit(formula, data=d).fit(
            disp=0, maxiter=100, cov_type="cluster",
            cov_kwds={"groups": d["user_id"].values})
    text = " ".join(type(w.message).__name__ + str(w.message) for w in caught)
    separation = "separation" in text.lower()
    converged = bool(res.mle_retvals.get("converged", True))

    beta, se = float(res.params["exposure"]), float(res.bse["exposure"])
    lo, hi = res.conf_int(alpha=0.05).loc["exposure"]
    return {
        "threshold": thr,
        "beta": beta,
        "se": se,
        "odds_ratio": float(np.exp(beta)),
        "or_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "p_one_sided": float(norm.sf(beta / se)),  # test of beta > 0
        "converged": converged,
        "separation": separation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    start = time.time()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    setup_logging(cfg["output"]["log_file"])
    log.info("Config: %s | mode: %s", args.config, cfg["project"]["mode"])

    a, g = cfg["analysis"], cfg["guardrails"]
    df, missing = clean_and_prepare(load_data(cfg))

    result = {"mode": cfg["project"]["mode"], "missing_share": missing}
    result["descriptive"] = describe(df, a["completion_threshold"])
    result["smd_log_duration"] = balance_report(df)

    # Main model and robustness check
    main_fit, robust_fit = None, None
    try:
        main_fit = fit_model(df, a["completion_threshold"])
        robust_fit = fit_model(df, a["robustness_threshold"])
    except Exception as e:  # tiny samples can break the model
        log.info("MODEL FAILED: %s", e)

    # Guardrails
    sizes = result["descriptive"]["group_n"]
    guard = {
        "min_obs_per_group": min(sizes.get(0, 0), sizes.get(1, 0)) >= g["min_obs_per_group"],
        "missing_share": all(v <= g["max_missing_share"] for v in missing.values()),
        "model_ok": bool(main_fit and main_fit["converged"] and not main_fit["separation"]),
        "robustness_sign": bool(main_fit and robust_fit
                                and np.sign(main_fit["beta"]) == np.sign(robust_fit["beta"])),
    }
    result["guardrails"] = guard
    log.info("Guardrails: %s", guard)

    if main_fit:
        result["main_model"], result["robustness_model"] = main_fit, robust_fit
        log.info("MAIN MODEL: OR = %.3f, 95%% CI [%.3f, %.3f], one-sided p = %.4f",
                 main_fit["odds_ratio"], *main_fit["or_ci95"], main_fit["p_one_sided"])
        reject = main_fit["p_one_sided"] < a["alpha"] and all(guard.values())
        result["reject_h0"] = bool(reject)
        log.info("DECISION: %s", "reject H0" if reject else "do not reject H0")
    if cfg["project"]["mode"] == "sample":
        log.info("Sample mode: this run only checks that the pipeline works.")

    result["runtime_s"] = round(time.time() - start, 3)
    log.info("Runtime: %.2f s", result["runtime_s"])
    out = Path(cfg["output"]["dir"]) / f"results_{cfg['project']['mode']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()