# Best model report: mlp

Selected by validation macro-F1. Use the student-test partition for model comparison; do not use evaluator labels for model selection.

## Validation metrics

| Metric | Value |
| --- | ---: |
| validation_accuracy | 0.958217 |
| validation_balanced_accuracy | 0.902731 |
| validation_macro_precision | 0.894459 |
| validation_macro_recall | 0.902731 |
| validation_macro_f1 | 0.898535 |
| validation_weighted_f1 | 0.958434 |
| validation_log_loss | 0.112621 |
| validation_roc_auc | 0.978239 |
| validation_top1_accuracy | 0.816000 |
| training_seconds | 8.494527 |
| inference_seconds | 0.011011 |

## Dataset summary

- Training candidate rows: 17485
- Validation candidate rows: 2154
- Numeric features: 14
- Training candidate coverage: 89.08%
- Validation candidate coverage: 99.79%
- Sampling: At most 25 evenly spaced observations per trip; all candidates for selected observations retained.
- Training label counts: `{'0': 15542, '1': 1943}`
- Validation label counts: `{'0': 1906, '1': 248}`
- Random state: 42

## Cross-validation

| Metric | Mean | Std |
| --- | ---: | ---: |
| macro_f1 | 0.879088 | 0.003971 |
| candidate_accuracy | 0.953168 | 0.001864 |
| log_loss | 0.126194 | 0.006450 |
| fit_seconds | 4.798394 | 1.104097 |
| inference_seconds | 0.011888 | 0.003610 |

### Per-fold scores

| Fold | Macro-F1 | Accuracy | Log loss | Fit seconds | Inference seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.877104 | 0.954060 | 0.121565 | 5.552462 | 0.015306 |
| 2 | 0.875530 | 0.950573 | 0.135315 | 5.605455 | 0.013462 |
| 3 | 0.884629 | 0.954870 | 0.121702 | 3.237266 | 0.006895 |

## Confusion matrix

Class order: `[0, 1]`

```json
[
  [
    1858,
    48
  ],
  [
    42,
    206
  ]
]
```

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.977895 | 0.974816 | 0.976353 | 1906.0 |
| 1 | 0.811024 | 0.830645 | 0.820717 | 248.0 |
| macro avg | 0.894459 | 0.902731 | 0.898535 | 2154.0 |
| weighted avg | 0.958682 | 0.958217 | 0.958434 | 2154.0 |

## Model parameters

```json
{
  "classifier": "MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)",
  "classifier__activation": "relu",
  "classifier__alpha": 0.0001,
  "classifier__batch_size": "auto",
  "classifier__beta_1": 0.9,
  "classifier__beta_2": 0.999,
  "classifier__early_stopping": false,
  "classifier__epsilon": 1e-08,
  "classifier__hidden_layer_sizes": [
    64,
    32
  ],
  "classifier__learning_rate": "constant",
  "classifier__learning_rate_init": 0.001,
  "classifier__max_fun": 15000,
  "classifier__max_iter": 100,
  "classifier__momentum": 0.9,
  "classifier__n_iter_no_change": 10,
  "classifier__nesterovs_momentum": true,
  "classifier__power_t": 0.5,
  "classifier__random_state": 42,
  "classifier__shuffle": true,
  "classifier__solver": "adam",
  "classifier__tol": 0.0001,
  "classifier__validation_fraction": 0.1,
  "classifier__verbose": false,
  "classifier__warm_start": false,
  "memory": null,
  "scaler": "StandardScaler()",
  "scaler__copy": true,
  "scaler__with_mean": true,
  "scaler__with_std": true,
  "steps": [
    [
      "scaler",
      "StandardScaler()"
    ],
    [
      "classifier",
      "MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)"
    ]
  ],
  "transform_input": null,
  "verbose": false
}
```

## Feature importance

This estimator does not expose a supported importance measure.

