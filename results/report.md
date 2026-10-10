# Ibà-ATLAS benchmark: fine-tuned vs base N-ATLAS

## all (n=710)

| Metric | base N-ATLaS | Ibà-ATLAS v1 | Ibà-ATLAS v2 |
|---|---|---|---|
| valid_json_pct | 99.9 | 100.0 | 100.0 |
| triage_exact_pct | 50.6 | 86.1 | 93.5 |
| under_triage_pct | 23.1 | 5.9 | 3.5 |
| refer_now_recall_pct | 77.6 | 92.2 | 97.4 |
| danger_sign_recall_pct | 17.3 | 93.0 | 100.0 |
| danger_sign_precision_pct | 19.3 | 99.1 | 98.8 |
| malaria_risk_acc_pct | 26.0 | 80.6 | 89.7 |

## hospital (n=649)

| Metric | base N-ATLaS | Ibà-ATLAS v1 | Ibà-ATLAS v2 |
|---|---|---|---|
| valid_json_pct | 99.8 | 100.0 | 100.0 |
| triage_exact_pct | 48.8 | 90.6 | 94.8 |
| under_triage_pct | 22.3 | 2.3 | 2.2 |
| refer_now_recall_pct | 79.8 | 100.0 | 100.0 |
| danger_sign_recall_pct | 14.1 | 100.0 | 100.0 |
| danger_sign_precision_pct | 15.6 | 100.0 | 100.0 |
| malaria_risk_acc_pct | 26.0 | 80.6 | 89.7 |

## lang_en (n=499)

| Metric | base N-ATLaS | Ibà-ATLAS v1 | Ibà-ATLAS v2 |
|---|---|---|---|
| valid_json_pct | 100.0 | 100.0 | 100.0 |
| triage_exact_pct | 53.7 | 85.4 | 92.8 |
| under_triage_pct | 19.6 | 6.6 | 4.2 |
| refer_now_recall_pct | 83.2 | 90.2 | 96.7 |
| danger_sign_recall_pct | 22.3 | 91.2 | 100.0 |
| danger_sign_precision_pct | 24.5 | 98.6 | 98.3 |
| malaria_risk_acc_pct | 25.3 | 81.0 | 90.0 |

## lang_pcm (n=211)

| Metric | base N-ATLaS | Ibà-ATLAS v1 | Ibà-ATLAS v2 |
|---|---|---|---|
| valid_json_pct | 99.5 | 100.0 | 100.0 |
| triage_exact_pct | 43.1 | 87.7 | 95.3 |
| under_triage_pct | 31.3 | 4.3 | 1.9 |
| refer_now_recall_pct | 64.4 | 97.1 | 99.0 |
| danger_sign_recall_pct | 5.8 | 97.1 | 100.0 |
| danger_sign_precision_pct | 6.7 | 100.0 | 100.0 |
| malaria_risk_acc_pct | 27.5 | 79.7 | 88.9 |

## iba_vignette (n=61)

| Metric | base N-ATLaS | Ibà-ATLAS v1 | Ibà-ATLAS v2 |
|---|---|---|---|
| valid_json_pct | 100.0 | 100.0 | 100.0 |
| triage_exact_pct | 68.9 | 37.7 | 80.3 |
| under_triage_pct | 31.1 | 44.3 | 18.0 |
| refer_now_recall_pct | 58.3 | 25.0 | 75.0 |
| danger_sign_recall_pct | 50.0 | 20.0 | 100.0 |
| danger_sign_precision_pct | 62.5 | 66.7 | 88.2 |
| malaria_risk_acc_pct | – | – | – |

Under-triage (predicted less urgent than gold, invalid output counted as missed) is the headline safety metric; lower is better.