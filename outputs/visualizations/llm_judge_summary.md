# LLM-as-Judge Evaluation (AWS Bedrock Nova Lite)

**Pairs evaluated:** 200
**Model:** amazon.nova-lite-v1:0
**LLM score distribution:** {0: 35, 1: 126, 2: 39}

## Spearman Correlation with LLM Judgments

Higher correlation = method rankings align better with independent LLM assessment.

| method          |   global_spearman |   global_p_value | significant   |   mean_per_jd_spearman |   n_jds_with_variance |
|:----------------|------------------:|-----------------:|:--------------|-----------------------:|----------------------:|
| TF-IDF          |            0.3686 |            0     | True          |                 0.5906 |                     9 |
| Skill-IDF       |            0.311  |            7e-06 | True          |                 0.5754 |                     5 |
| Multi-Agent+IDF |            0.5767 |            0     | True          |                 0.7474 |                     9 |

## LLM Agreement with Proxy Labels

- Relevant pairs: LLM mean = 1.39
- Irrelevant pairs: LLM mean = 0.72
- Mann-Whitney p-value = 0.000000
