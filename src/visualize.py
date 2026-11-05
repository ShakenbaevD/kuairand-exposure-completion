"""Publication-ready figures (Task 3). Reads only the small files in data/processed/.

The three rules of scientific graphics are applied:
1. explicit axis titles with units; log scale where values span orders of magnitude;
2. uncertainty always shown (95% bootstrap CI error bars / boxplot notches);
3. colorblind-safe palette + different markers and line styles (readable in grayscale).

File names follow the course template. Here "load" on the x-axis of Fig. 1 is the
recommendation scenario (tab), because this study has no request-rate benchmark.

Usage:
    python src/visualize.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import cbook

from process_data import GROUPS, boot_stat_ci, _median

ORDER = [GROUPS[1], GROUPS[0]]  # baseline first
PALETTE = dict(zip(ORDER, sns.color_palette("colorblind")[:2]))
MARKERS = {GROUPS[1]: "s", GROUPS[0]: "o"}
LINESTYLES = {GROUPS[1]: "--", GROUPS[0]: "-"}


def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # embed fonts
                         "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name, figures_dir):
    """Vector PDF for papers and 300 DPI PNG for the README."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(figures_dir / f"{name}.png", dpi=300)


MIN_VIEWS = 50  # scenario points with fewer views in a group are not drawn (CI would be meaningless)


def fig1_completion_by_tab(summary, min_views=MIN_VIEWS):
    """Completion rate (%) per recommendation scenario with 95% bootstrap CI.

    Scenarios with fewer than ``min_views`` views in a group are left out, and the
    line is broken there instead of interpolating over missing data.
    """
    set_style()
    tabs = summary[summary["scope"] == "tab"]
    all_tabs = list(range(15))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    offsets = {GROUPS[1]: -0.12, GROUPS[0]: 0.12}  # avoid overlapping error bars
    for g in ORDER:
        d = tabs[tabs["group"] == g].set_index("tab").reindex(all_tabs)
        d.loc[d["n_views"].fillna(0) < min_views,
              ["completion_rate_pct", "cr_ci_low_pct", "cr_ci_high_pct"]] = np.nan
        if d["completion_rate_pct"].isna().all():
            continue
        yerr = np.vstack([d["completion_rate_pct"] - d["cr_ci_low_pct"],
                          d["cr_ci_high_pct"] - d["completion_rate_pct"]])
        ax.errorbar(np.array(all_tabs) + offsets[g], d["completion_rate_pct"], yerr=yerr,
                    color=PALETTE[g], marker=MARKERS[g], linestyle=LINESTYLES[g],
                    capsize=3, linewidth=1.6, markersize=6, label=g)
    ax.set_xlabel(f"Recommendation scenario (tab ID, unitless)\n"
                  f"Points with fewer than {min_views} views in a group are omitted")
    ax.set_ylabel("Completion rate (% of views)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(all_tabs)
    ax.legend(title="Exposure", frameon=True)
    fig.tight_layout()
    return fig


def _bxp_stats(values, rng, n_boot):
    """Box statistics with a seeded bootstrap 95% CI of the median (notches)."""
    s = cbook.boxplot_stats(values, whis=1.5)[0]
    s["cilo"], s["cihi"] = boot_stat_ci(values, _median, rng, n_boot)
    return s


def fig2_distribution_boxplot(sample, seed=42, n_boot=1000):
    """Watch time and video duration (log scale) by exposure group."""
    set_style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    panels = [("watch_time_s", "Watch time (s, log scale)", "(a)"),
              ("duration_s", "Video duration (s, log scale)", "(b)")]
    for ax, (col, ylabel, tag) in zip(axes, panels):
        stats = []
        for g in ORDER:
            vals = sample.loc[sample["group"] == g, col].to_numpy(float)
            st = _bxp_stats(vals, rng, n_boot)
            st["label"] = g
            stats.append(st)
        bp = ax.bxp(stats, showfliers=False, shownotches=True, patch_artist=True,
                    widths=0.5)
        for patch, g, hatch in zip(bp["boxes"], ORDER, ["//", ""]):
            patch.set_facecolor(PALETTE[g]); patch.set_alpha(0.75)
            patch.set_hatch(hatch)  # hatch keeps boxes distinguishable in grayscale
        for g, m, med in zip(ORDER, ["s", "o"], bp["medians"]):
            med.set_color("black"); med.set_linewidth(1.8)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
        ax.set_xlabel("Exposure type")
        ax.set_ylabel(ylabel)
        ax.set_title(tag, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def run(cfg):
    matplotlib.use("Agg")  # command-line run: no window; the notebook keeps its inline backend
    p = cfg["processing"]
    proc = Path(p["out_dir"])
    summary = pd.read_csv(proc / "metrics_summary.csv")
    sample = pd.read_csv(proc / "watch_time_sample.csv")
    f1 = fig1_completion_by_tab(summary)
    save(f1, "fig1_latency_vs_load", p["figures_dir"])
    f2 = fig2_distribution_boxplot(sample, p["seed"], p["n_boot"])
    save(f2, "fig2_distribution_boxplot", p["figures_dir"])
    print(f"Figures saved to {p['figures_dir']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config, encoding="utf-8")))


if __name__ == "__main__":
    main()
