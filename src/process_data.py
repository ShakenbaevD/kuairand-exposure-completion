"""Data processing and statistical aggregation (Task 3).

Reads the same three KuaiRand-1K logs as ``main.py`` and produces small,
git-friendly artifacts in ``data/processed/``:

* ``metrics_summary.csv``   - cleaned and aggregated metrics (per group, per tab,
                              and the Personalized - Random difference)
* ``data_quality.csv``      - share of invalid / missing rows (guardrail metric)
* ``watch_time_sample.csv`` - seeded sample of views used for the boxplot
* ``model_summary.json``    - adjusted OR copied from ``outputs/results_full.json``
                              (produced by ``src/main.py``), if that file exists

Mapping of the course template to this study (no latency benchmark here):
* "warm-up filtering"  -> removal of invalid views (missing values, duration <= 0,
                          negative play time); every removed row is counted
* "error rate"         -> ``invalid_share_pct``, reported as a guardrail metric
* "latency"            -> watch time in seconds; "load" -> recommendation scenario (tab)

Usage:
    python src/process_data.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GROUPS = {0: "Personalized", 1: "Random (baseline)"}
DIFF_LABEL = "Personalized - Random"
README_START, README_END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"


# --------------------------------------------------------------------------- #
# Step 1. Ingestion and filtering
# --------------------------------------------------------------------------- #
def load_views(cfg):
    """Load personalized and random logs, attach nothing else, count invalid rows.

    Returns (clean_df, quality_df). ``clean_df`` has columns
    user_id, tab, is_rand, group, watch_time_s, duration_s, completed.
    """
    c, d = cfg["columns"], cfg["data"]
    use = [c["user_id"], c["play_time_ms"], c["duration_ms"], c["tab"]]
    rename = {c["user_id"]: "user_id", c["play_time_ms"]: "play_time_ms",
              c["duration_ms"]: "duration_ms", c["tab"]: "tab"}
    thr = cfg["analysis"]["completion_threshold"]

    parts, quality = [], []
    for is_rand, key in ((0, "personalized_log"), (1, "random_log")):
        raw = pd.read_csv(d[key], usecols=use).rename(columns=rename)
        n_raw = len(raw)
        missing = raw[["play_time_ms", "duration_ms", "tab"]].isna().any(axis=1)
        ok = ~missing
        invalid = ok & ~((raw["duration_ms"] > 0) & (raw["play_time_ms"] >= 0))
        keep = raw[ok & ~invalid].copy()
        keep["is_rand"] = is_rand
        parts.append(keep)
        quality.append({
            "group": GROUPS[is_rand], "n_raw": n_raw,
            "n_missing": int(missing.sum()), "n_invalid": int(invalid.sum()),
            "n_clean": len(keep),
            "invalid_share_pct": 100 * (int(missing.sum()) + int(invalid.sum())) / max(n_raw, 1),
        })

    df = pd.concat(parts, ignore_index=True)
    df["group"] = df["is_rand"].map(GROUPS)
    df["tab"] = df["tab"].astype(int)
    df["watch_time_s"] = df["play_time_ms"] / 1000
    df["duration_s"] = df["duration_ms"] / 1000
    df["completed"] = (df["play_time_ms"] >= thr * df["duration_ms"]).astype(int)
    df = df[["user_id", "tab", "is_rand", "group", "watch_time_s", "duration_s", "completed"]]
    return df, pd.DataFrame(quality)


def iqr_outlier_flag(x):
    """Non-destructive outlier flag (1.5 * IQR rule). Nothing is deleted."""
    q1, q3 = np.percentile(x, [25, 75])
    return (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))


# --------------------------------------------------------------------------- #
# Step 2. Statistical aggregation
# --------------------------------------------------------------------------- #
def _pct_ci(values, alpha=0.05):
    return np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def cluster_boot_rates(df, rng, n_boot):
    """Bootstrap distribution of the completion rate, resampling USERS.

    Views of the same user are not independent, so whole users are resampled.
    Returns an array of n_boot rates (in %); empty array if no users.
    """
    g = df.groupby("user_id")["completed"].agg(["sum", "size"])
    if len(g) == 0:
        return np.array([])
    s, n = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    return 100 * s[idx].sum(axis=1) / n[idx].sum(axis=1)


def boot_stat_ci(x, stat, rng, n_boot, alpha=0.05):
    """Percentile bootstrap CI of a statistic of a 1-D sample (rows resampled)."""
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return tuple(_pct_ci(stat(x[idx]), alpha))


def _median(a): return np.median(a, axis=1)
def _p95(a): return np.percentile(a, 95, axis=1)
def _p99(a): return np.percentile(a, 99, axis=1)


def summarize(df, rng, n_boot, quantile_sample, with_quantile_ci=True):
    """One row of metrics for a block of views."""
    w = df["watch_time_s"].to_numpy()
    row = {
        "n_views": len(df), "n_users": int(df["user_id"].nunique()),
        "completion_rate_pct": 100 * df["completed"].mean(),
        "watch_mean_s": w.mean(), "watch_sd_s": w.std(ddof=1) if len(w) > 1 else np.nan,
        "watch_median_s": np.median(w),
        "watch_p25_s": np.percentile(w, 25), "watch_p75_s": np.percentile(w, 75),
        "watch_p95_s": np.percentile(w, 95), "watch_p99_s": np.percentile(w, 99),
        "duration_median_s": df["duration_s"].median(),
        "outlier_share_iqr_pct": 100 * iqr_outlier_flag(w).mean(),
    }
    boots = cluster_boot_rates(df, rng, n_boot)
    row["cr_ci_low_pct"], row["cr_ci_high_pct"] = (_pct_ci(boots) if len(boots) else (np.nan, np.nan))
    if with_quantile_ci and len(w) > 1:
        sample = w if len(w) <= quantile_sample else rng.choice(w, quantile_sample, replace=False)
        for name, fn in (("median", _median), ("p95", _p95), ("p99", _p99)):
            lo, hi = boot_stat_ci(sample, fn, rng, n_boot)
            row[f"watch_{name}_ci_low_s"], row[f"watch_{name}_ci_high_s"] = lo, hi
    return row, boots


def build_summary(df, quality, cfg):
    """Aggregate per group, per (group, tab) and the group difference."""
    p = cfg["processing"]
    rng = np.random.default_rng(p["seed"])
    rows, group_boots = [], {}
    for is_rand in (0, 1):  # fixed order -> reproducible random stream
        sub = df[df["is_rand"] == is_rand]
        row, boots = summarize(sub, rng, p["n_boot"], p["quantile_sample"])
        group_boots[is_rand] = boots
        row.update(scope="group", group=GROUPS[is_rand], tab=np.nan)
        row["invalid_share_pct"] = float(
            quality.loc[quality["group"] == GROUPS[is_rand], "invalid_share_pct"].iloc[0])
        rows.append(row)

    if len(group_boots[0]) and len(group_boots[1]):
        diff = group_boots[0] - group_boots[1]  # independent groups -> pairwise difference
        d0, d1 = rows[0], rows[1]
        lo, hi = _pct_ci(diff)
        rows.append({"scope": "difference", "group": DIFF_LABEL, "tab": np.nan,
                     "n_views": d0["n_views"] + d1["n_views"],
                     "completion_rate_pct": d0["completion_rate_pct"] - d1["completion_rate_pct"],
                     "cr_ci_low_pct": lo, "cr_ci_high_pct": hi})

    for is_rand in (0, 1):
        for tab, sub in df[df["is_rand"] == is_rand].groupby("tab"):
            row, _ = summarize(sub, rng, p["n_boot"], p["quantile_sample"], with_quantile_ci=False)
            row.update(scope="tab", group=GROUPS[is_rand], tab=int(tab))
            rows.append(row)

    cols = ["scope", "group", "tab", "n_views", "n_users", "completion_rate_pct",
            "cr_ci_low_pct", "cr_ci_high_pct", "watch_mean_s", "watch_sd_s",
            "watch_median_s", "watch_median_ci_low_s", "watch_median_ci_high_s",
            "watch_p25_s", "watch_p75_s", "watch_p95_s", "watch_p95_ci_low_s",
            "watch_p95_ci_high_s", "watch_p99_s", "watch_p99_ci_low_s",
            "watch_p99_ci_high_s", "duration_median_s", "outlier_share_iqr_pct",
            "invalid_share_pct"]
    return pd.DataFrame(rows).reindex(columns=cols)


def make_plot_sample(df, cfg):
    """Seeded sample of views for the distribution plot (small, tracked in git)."""
    p = cfg["processing"]
    parts = []
    for is_rand in (0, 1):
        sub = df[df["is_rand"] == is_rand]
        n = min(len(sub), p["plot_sample"])
        parts.append(sub.sample(n=n, random_state=p["seed"]))
    return pd.concat(parts, ignore_index=True)[["group", "tab", "watch_time_s", "duration_s"]]


# --------------------------------------------------------------------------- #
# Model results (computed by main.py) and README table
# --------------------------------------------------------------------------- #
def load_model_summary(results_path="outputs/results_full.json"):
    path = Path(results_path)
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    m = r.get("main_model")
    if not m:
        return None
    return {"odds_ratio": m["odds_ratio"], "or_ci95_low": m["or_ci95"][0],
            "or_ci95_high": m["or_ci95"][1], "p_one_sided": m["p_one_sided"],
            "guardrails": r.get("guardrails", {}), "reject_h0": r.get("reject_h0"),
            "source": str(results_path)}


def _fmt(x, nd=2):
    return "n/a" if pd.isna(x) else f"{x:,.{nd}f}"


def results_table_md(summary, quality, model):
    """Baseline vs Proposed table: Mean +- SD, Median (p25-p75), 95% CI."""
    g = summary[summary["scope"] == "group"].set_index("group")
    base, prop = g.loc["Random (baseline)"], g.loc["Personalized"]
    diff = summary[summary["scope"] == "difference"]
    q = quality.set_index("group")

    def cr(r): return f"{_fmt(r.completion_rate_pct)} [{_fmt(r.cr_ci_low_pct)}, {_fmt(r.cr_ci_high_pct)}]"
    def ms(r): return f"{_fmt(r.watch_mean_s)} ± {_fmt(r.watch_sd_s)}"
    def med(r): return f"{_fmt(r.watch_median_s)} ({_fmt(r.watch_p25_s)}–{_fmt(r.watch_p75_s)})"
    def pc(r, k): return f"{_fmt(r[f'watch_{k}_s'])} [{_fmt(r[f'watch_{k}_ci_low_s'])}, {_fmt(r[f'watch_{k}_ci_high_s'])}]"

    lines = [
        "| Metric | Baseline: Random exposure | Proposed: Personalized exposure |",
        "|---|---|---|",
        f"| Views after cleaning (n) | {int(base.n_views):,} | {int(prop.n_views):,} |",
        f"| Users (n) | {int(base.n_users):,} | {int(prop.n_users):,} |",
        f"| Completion rate, % [95% bootstrap CI] | {cr(base)} | {cr(prop)} |",
        f"| Watch time, s: Mean ± SD | {ms(base)} | {ms(prop)} |",
        f"| Watch time, s: Median (p25–p75) | {med(base)} | {med(prop)} |",
        f"| Watch time p95, s [95% CI] | {pc(base, 'p95')} | {pc(prop, 'p95')} |",
        f"| Watch time p99, s [95% CI] | {pc(base, 'p99')} | {pc(prop, 'p99')} |",
        f"| Guardrail: invalid/missing rows, % | {_fmt(q.loc['Random (baseline)', 'invalid_share_pct'])} | {_fmt(q.loc['Personalized', 'invalid_share_pct'])} |",
    ]
    text = "\n".join(lines)
    if len(diff):
        d = diff.iloc[0]
        text += (f"\n\nDifference in completion rate (Personalized − Random): "
                 f"**{_fmt(d.completion_rate_pct)} p.p.** "
                 f"(95% bootstrap CI [{_fmt(d.cr_ci_low_pct)}, {_fmt(d.cr_ci_high_pct)}]).")
    if model:
        text += (f"\n\nAdjusted odds ratio of completion (logistic regression, "
                 f"cluster-robust SE): **OR = {_fmt(model['odds_ratio'], 3)}**, "
                 f"95% CI [{_fmt(model['or_ci95_low'], 3)}, {_fmt(model['or_ci95_high'], 3)}], "
                 f"one-sided p = {model['p_one_sided']:.4g}.")
    return text


def update_readme(table_md, readme_path="README.md"):
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"Markers {README_START} / {README_END} not found in {readme_path}")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{table_md}\n{README_END}{tail}", encoding="utf-8")


# --------------------------------------------------------------------------- #
def run(cfg, update_readme_file=None):
    out = Path(cfg["processing"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df, quality = load_views(cfg)
    summary = build_summary(df, quality, cfg)
    sample = make_plot_sample(df, cfg)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    quality.to_csv(out / "data_quality.csv", index=False)
    sample.to_csv(out / "watch_time_sample.csv", index=False)
    model = load_model_summary(Path(cfg["output"]["dir"]) / f"results_{cfg['project']['mode']}.json")
    if model:
        (out / "model_summary.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    table = results_table_md(summary, quality, model)
    print(table)
    if update_readme_file:
        update_readme(table, update_readme_file)
        print(f"\nREADME table updated: {update_readme_file}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--update-readme", action="store_true",
                    help="insert the results table between the README markers")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run(cfg, "README.md" if args.update_readme else None)


if __name__ == "__main__":
    main()
