# MLP hyperparameter tuning report

Hyperparameters were selected using the mean macro-F1 from stratified, trip-grouped cross-validation on the training partition. The student-test partition was not used during the search.

## Cross-validation results

| Configuration | Mean macro-F1 | Std macro-F1 | Mean top-1 accuracy |
| --- | ---: | ---: | ---: |
| Baseline MLP | 0.8791 | 0.0040 | 0.7884 |
| Tuned MLP | 0.8839 | 0.0022 | 0.7934 |

- Trials: 15 completed, 5 pruned, 20 total.
- Search time: 118.80 seconds
- Training trips: 80
- Training candidate rows: 17485
- Training candidate coverage: 89.08%
- Sampling: At most 25 evenly spaced observations per trip; all candidates for selected observations retained.

## Final student-test comparison

These scores are a final comparison after the grouped-CV search; they did not choose the hyperparameters.

| Configuration | Macro-F1 | Candidate accuracy | Top-1 accuracy | Log loss | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline MLP | 0.8985 | 0.9582 | 0.8160 | 0.1126 | 8.08 |
| Tuned MLP | 0.8956 | 0.9582 | 0.8200 | 0.1173 | 3.83 |

## Selected hyperparameters

```json
{
  "activation": "tanh",
  "alpha": 1.103348317424011e-05,
  "batch_size": 128,
  "early_stopping": true,
  "hidden_layer_sizes": [
    128,
    64
  ],
  "learning_rate_init": 0.0028426337675688763,
  "max_iter": 200
}
```

## Run outputs

- Trial history: `/home/priyaang/College/PathSnap_Team8_Gsec/results/mlp_tuning/trials.csv`
- Machine-readable report: `/home/priyaang/College/PathSnap_Team8_Gsec/results/mlp_tuning/tuning_report.json`
- Fitted tuned model: `/home/priyaang/College/PathSnap_Team8_Gsec/results/mlp_tuning/tuned_mlp.joblib`
