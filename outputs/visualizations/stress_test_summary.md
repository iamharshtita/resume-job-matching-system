# Stress Test - Domain Discrimination

## Domain Separation

| method          |   relevant_mean |   irrelevant_mean |    gap |   cohens_d |    auc | significant   |
|:----------------|----------------:|------------------:|-------:|-----------:|-------:|:--------------|
| TF-IDF          |          0.048  |            0.0366 | 0.0114 |      0.63  | 0.6723 | True          |
| Skill-IDF       |          0.0611 |            0.019  | 0.0421 |      0.562 | 0.6313 | True          |
| Multi-Agent+IDF |          0.4451 |            0.4258 | 0.0193 |      0.816 | 0.7302 | True          |

## Easy vs Hard Irrelevant

| method          |   relevant_mean |   hard_irr_mean |   easy_irr_mean |   rel_hard_gap |   rel_easy_gap |
|:----------------|----------------:|----------------:|----------------:|---------------:|---------------:|
| TF-IDF          |          0.048  |          0.0374 |          0.0358 |         0.0106 |         0.0122 |
| Skill-IDF       |          0.0611 |          0.0278 |          0.0102 |         0.0333 |         0.0509 |
| Multi-Agent+IDF |          0.4451 |          0.4319 |          0.4196 |         0.0132 |         0.0254 |