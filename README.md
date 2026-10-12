# Personalized vs. Random Exposure and Video Completion Rate (KuaiRand-1K)

**Authors:** Azamat Zhumanov, Diyar Shakenbayev
**Repository:** https://github.com/ShakenbaevD/kuairand-exposure-completion

## Research Question

Does personalized (algorithmic) video exposure increase the probability that a user watches a video to completion, compared with randomly exposed videos, after adjusting for video duration, prior user activity, and platform tab?

## Overview

This is a quantitative, quasi-experimental study on the public KuaiRand-1K dataset. Two exposure groups are compared over 2022-04-22 to 2022-05-08:

- **Personalized** (`is_rand = 0`): `log_standard_4_22_to_5_08_1k.csv`
- **Random** (`is_rand = 1`): `log_random_4_22_to_5_08_1k.csv`

A view is *completed* if `play_time_ms >= 0.9 * duration_ms` (robustness check: threshold 1.0).
Prior user activity is the number of interactions in `log_standard_4_08_to_4_21_1k.csv` (no records = 0).

Model (logistic regression, standard errors clustered by `user_id`):

```
logit(completed) ~ exposure + log(duration_s) + log(1 + activity) + C(tab)
```

where `exposure = 1 - is_rand` (1 = personalized, baseline = random).

- **H0:** OR <= 1 (personalized exposure does not increase completion odds)
- **H1:** OR > 1
- **Primary metric:** adjusted odds ratio of `exposure` with 95% CI. Decision rule: one-sided p < 0.05 and all guardrails pass.
- **Secondary metrics:** difference in completion rate (percentage points), difference in mean watch time (seconds).

## Planned Benchmarks

| ID | Metric | Type | Groups compared | Pass condition |
|----|--------|------|-----------------|----------------|
| B1 | Adjusted OR of exposure (95% CI) | Primary | Personalized vs. random | One-sided p < 0.05 and OR > 1 |
| B2 | Completion rate difference (p.p.) | Secondary | Personalized vs. random | Reported only |
| B3 | Mean watch time difference (s) | Secondary | Personalized vs. random | Reported only |
| G1 | Observations per group | Guardrail | Each group | >= 1000 |
| G2 | Share of missing values | Guardrail | Whole dataset | <= 1% |
| G3 | Model convergence | Guardrail | Logit model | Converges, no perfect separation |
| G4 | Sign robustness | Guardrail | Threshold 0.9 vs. 1.0 | Sign of beta_1 unchanged |

## Repository Structure

```
.
├── configs/
│   ├── config.yaml          # full experiment (KuaiRand-1K in data/raw/)
│   └── config_sample.yaml   # smoke test on data/sample/
├── data/
│   ├── raw/                 # KuaiRand-1K files (NOT tracked by git)
│   └── sample/              # tiny synthetic sample (tracked)
├── docs/                    # methodology card
├── notebooks/               # exploratory notebooks
├── outputs/                 # results and logs (NOT tracked by git)
├── src/
│   └── main.py              # benchmark entry point
├── Dockerfile
├── requirements.txt         # exact version pins
├── LICENSE
└── README.md
```

## System Requirements

- Python 3.13.9 (or Docker)
- Packages from `requirements.txt` (all versions pinned exactly)
- For the full run: KuaiRand-1K files in `data/raw/` (several hundred MB, free disk space recommended)

## Quickstart (sample data, ~1 minute)

```bash
git clone https://github.com/ShakenbaevD/kuairand-exposure-completion.git
cd kuairand-exposure-completion
pip install -r requirements.txt && python src/main.py --config configs/config_sample.yaml
```

Results are written to `outputs/` (`results_*.json` and a log file).

> Note: the sample is tiny and synthetic. The model may show perfect separation and the `model_ok` guardrail will be `False`. This is expected and only verifies that the pipeline runs end to end.

### Docker alternative

```bash
docker build -t kuairand-exposure .
docker run --rm -v "$(pwd)/outputs:/app/outputs" kuairand-exposure
```

## Running on the Full Dataset

1. Download KuaiRand-1K from the official dataset site (https://kuairand.com) and unpack it.
2. Copy the three log files (`log_standard_4_08_to_4_21_1k.csv`, `log_standard_4_22_to_5_08_1k.csv`, `log_random_4_22_to_5_08_1k.csv`) into `data/raw/`.
3. Run:

```bash
python src/main.py --config configs/config.yaml
```

Raw data is excluded from git by `.gitignore`.

## Dataset Citation

Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., He, X. (2022). *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos.* CIKM 2022.

## License

Released under the MIT License (see `LICENSE`).