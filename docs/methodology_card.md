# Methodology Passport Card

## 1. Project Information

- **Title:** Comparative Evaluation of the Impact of Personalized Recommendations and Random Video Exposure on User Attention Retention in Video Services
- **Authors:** Zhumanov Azamat, Shakenbayev Diyar
- **GitHub repository:** https://github.com/ShakenbaevD/kuairand-exposure-completion

## 2. Type of Research

The study is **quantitative**. We analyze interaction logs from the open KuaiRand-1K dataset and compare two groups of video views: views shown by the standard (personalized) recommender and views that the platform replaced with randomly chosen videos. All conclusions are based on numerical metrics (odds ratio, completion rate, watch time) and on a statistical hypothesis test.

We chose a quantitative design because our research question asks about the size and direction of a difference between two exposure conditions, and the dataset already contains the necessary behavioral signals. A qualitative or mixed-methods design would require interviews or user feedback, which we do not collect and which would not help answer this specific question. Because the exposure type was assigned by the platform and not by us, the study is **quasi-experimental and observational**, so we describe results as associations, not as proven causal effects.

## 3. Hypotheses

**Research question.** Is personalized video exposure associated with higher odds of video completion than random exposure, after controlling for video duration, user activity and recommendation scenario?

This question comes from RQ2 of Task 1. We made it the single primary question because earlier work in our related-work matrix (Zhao et al., 2023; Quan et al., 2023; Zhao et al., 2024) shows that raw watch time is strongly affected by video length, so a comparison without a duration control would be unreliable. Watch time (RQ1 in Task 1) is kept as a secondary, descriptive result.

**Operational definitions.**

- **Completed view:** `play_time_ms >= 0.9 * duration_ms`. We use 90% instead of 100% so that the result is not sensitive to a user leaving in the last second or to rounding. The 100% threshold is checked later as a robustness test.
- **Completion rate (CR) of a group:** the share of completed views in that group.
- **Groups (same period, 22.04-08.05.2022):**
  - personalized exposure: `log_standard_4_22_to_5_08_1k.csv`, `is_rand = 0`;
  - random exposure: `log_random_4_22_to_5_08_1k.csv`, `is_rand = 1`.

  Using the same period for both groups prevents time effects from being mixed with the exposure effect.
- **User activity:** number of interactions of the user in `log_standard_4_08_to_4_21_1k.csv` (08-21.04.2022); users with no records get 0. It is computed before the comparison period, so it does not depend on the views being compared.

**Model (logistic regression):**

```
logit P(completed) = β0 + β1*exposure + β2*log(duration_s) + β3*log(1 + activity) + sum_t γ_t*tab_t
```

- `exposure = 1 - is_rand`: **1 = personalized, 0 = random (reference level)**. Note that this is the inverse of the raw `is_rand` column in the data.
- Standard errors are clustered by `user_id`, because views of the same user are not independent.
- `OR = exp(β1)` is the adjusted odds ratio of completing a video under personalized exposure compared with random exposure.

**Hypotheses.**

- **H0:** OR ≤ 1. After controlling for video duration, user activity and recommendation scenario, personalized exposure is not associated with higher odds of completion than random exposure.
- **H1:** OR > 1. After the same controls, personalized exposure is associated with higher odds of completion than random exposure.

H0 and H1 together cover all possible outcomes, and both are tested with the same regression coefficient, so the hypotheses are empirically testable.

**Decision rule (fixed before running the experiment).** We reject H0 only if both conditions hold:

1. the one-sided p-value for β1 (test of β1 > 0) is below 0.05;
2. all guardrail conditions in Section 5 are satisfied.

The decision is based on the one-sided p-value only; we do not require the lower bound of a two-sided confidence interval to be above 1, because that would be stricter than the one-sided test. In the report we always give the OR with its 95% two-sided confidence interval and the difference in completion rate in percentage points, because with a large sample even a very small effect can be statistically significant. A significant result shows an association, not that personalization causes longer attention.

## 4. Variable Matrix

| Type | Variable | Values / units |
|------|----------|----------------|
| **Independent** | Exposure type | binary: personalized (1) / random (0, reference) |
| **Dependent (primary)** | Video completion | 0/1 (rule from Section 3) |
| **Dependent (secondary)** | Watch time | seconds (`play_time_ms / 1000`) |
| **Covariates** (controlled statistically in the model) | Video duration | `log(duration_s)` |
| | User activity | `log(1 + number of interactions, 08-21.04.2022)` |
| | Recommendation scenario | `tab`, categorical (0-14) |
| **Controlled conditions** (fixed) | Data | KuaiRand-1K, same period for both groups |
| | Filters | `duration_ms > 0`, `play_time_ms >= 0` |
| | Completion threshold | 0.9 (robustness: 1.0) |
| | Model | same logistic regression specification for all runs |
| | Standard errors | clustered by `user_id` |
| | Missing data | complete-case analysis over the model variables (`play_time_ms`, `duration_ms`, `is_rand`, `tab`, activity); number of removed rows is reported |
| | Environment | Python 3.13.9; Docker base image `python:3.13.9-slim`; all libraries pinned to exact versions in `requirements.txt` (e.g. pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, PyYAML 6.0.3) |
| | Randomness | no random seed is needed: the main analysis uses no random sampling |

**Why these covariates.** Duration is the main known confounder of watch time and completion; activity captures differences between users; `tab` is added because KuaiRand contains several recommendation scenarios and their share may differ between the two groups. `tab` was not in our Task 1 list of controls, so it is a refinement made at the design stage.

## 5. Metrics and Baseline

**Primary metric:** the adjusted odds ratio `OR = exp(β1)` with a 95% confidence interval. It is the only metric that takes all three controls into account and it answers the hypothesis directly: OR > 1 means that, for videos of the same duration, users of the same activity and the same scenario, personalized exposure is associated with higher odds of completion.

**Secondary metrics:**

- Difference in completion rate: `ΔCR = CR_personalized - CR_random`, in percentage points (shows the practical size of the effect; unadjusted).
- Difference in mean watch time: `ΔWT`, in seconds (descriptive, not used for the decision).

**Informational metric:** pipeline run time in seconds, written to the log.

**Guardrail metrics** (conditions under which the result is considered reliable):

1. at least 1000 observations in each group after all filters and removal of rows with missing model variables (checked in code, not assumed);
2. no more than 1% missing values in each model variable, counted separately for every variable before complete-case removal;
3. the model converged and there is no complete separation of classes;
4. robustness: the sign of β1 does not change when the completion threshold is 100% instead of 90% (100% corresponds to the platform's own definition of a complete play);
5. the report includes the standardized difference of mean `log(duration_s)` between the groups and the distribution of `tab` in each group, so the reader can see how comparable the groups are.

**Baseline (control group):** views under random exposure in the same period. In the model it is the reference level, so the odds ratio always shows how much personalized exposure differs from this baseline.

## 6. Threats to Validity

**Internal validity**

- *Different video pools.* Randomly exposed videos come from a small candidate pool (7,388 videos in KuaiRand-1K), while standard recommendations are drawn from millions of videos, so the groups may differ in duration and popularity. We reduce this with the duration control; the remaining difference is a limitation.
- *Different scenarios.* The share of recommendation scenarios may differ between groups. We control for `tab`.
- *Dependent observations.* Views of the same user are correlated. We use standard errors clustered by `user_id`.
- *Threshold choice.* The result could depend on the 90% threshold. We check the 100% threshold.
- *Unmeasured differences.* Even after controlling for duration, activity and scenario, the groups may still differ in video popularity, user preferences and other unobserved factors. The estimate is therefore an adjusted statistical association.
- *Unequal group sizes.* Random interactions are only about 0.37% of all interactions, so the groups are very unbalanced. We report group sizes and confidence intervals, not only p-values.

**External validity**

- The sample is 1,000 users of one app (Kuaishou), so results may not transfer to other video services.
- The data are offline logs, so behavior under a newly deployed recommendation policy may differ.
- The period covers about two and a half weeks and does not show seasonal effects.

**Interpretation.** A higher completion rate shows engagement, not necessarily user benefit or satisfaction. Because the study is observational, we interpret results as an adjusted association between exposure type and the odds of completion, not as proof that personalization causes it. The secondary difference in completion rate (ΔCR) is an unadjusted difference in proportions and is reported only to show the practical size of the effect.
