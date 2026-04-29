# Statistical Analysis

## Paired Significance Tests (Wilcoxon Signed-Rank)

| comparison                  |   w_statistic |     p_value | significant_005   |   wins |   ties |   losses |   n_pairs |
|:----------------------------|--------------:|------------:|:------------------|-------:|-------:|---------:|----------:|
| Multi-Agent+IDF > TF-IDF    |       16119   | 1.34959e-08 | True              |    142 |     39 |       69 |       250 |
| Multi-Agent+IDF > Skill-IDF |       16596   | 1.92892e-10 | True              |    145 |     40 |       65 |       250 |
| Skill-IDF > TF-IDF          |       13806.5 | 0.432161    | False             |    114 |     17 |      119 |       250 |

## Bootstrap 95% Confidence Intervals (NDCG@5)

| method          |   mean_ndcg |   ci_lower |   ci_upper |   bootstrap_std |
|:----------------|------------:|-----------:|-----------:|----------------:|
| TF-IDF          |      0.7197 |     0.6919 |     0.7483 |          0.0141 |
| Skill-IDF       |      0.7134 |     0.6846 |     0.741  |          0.0147 |
| Multi-Agent+IDF |      0.8145 |     0.7896 |     0.8388 |          0.0127 |

## Improvement Over Baselines

| baseline   | metric   |   baseline_value |   multi_agent_value |   absolute_gain |   relative_gain_pct |
|:-----------|:---------|-----------------:|--------------------:|----------------:|--------------------:|
| TF-IDF     | ndcg@5   |           0.7197 |              0.8145 |          0.0948 |                13.2 |
| TF-IDF     | prec@5   |           0.7136 |              0.8032 |          0.0896 |                12.6 |
| TF-IDF     | rec@5    |           0.3568 |              0.4016 |          0.0448 |                12.6 |
| TF-IDF     | map      |           0.3066 |              0.3676 |          0.061  |                19.9 |
| Skill-IDF  | ndcg@5   |           0.7134 |              0.8145 |          0.1011 |                14.2 |
| Skill-IDF  | prec@5   |           0.656  |              0.8032 |          0.1472 |                22.4 |
| Skill-IDF  | rec@5    |           0.328  |              0.4016 |          0.0736 |                22.4 |
| Skill-IDF  | map      |           0.2972 |              0.3676 |          0.0704 |                23.7 |