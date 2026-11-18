# PathSnap: Map-Matching & Route Reconstruction

**Team 8** · UE24CS352A Machine Learning · PES University, EC Campus

| Team member | SRN |
|-------------|-----|
|Prithvi S P  | PES2UG24CS375 |
| Rohan Chirbi | PES2UG24CS408 |
| Priyaan G | PES2UG24CS376 |
| Sahana M S | PES2UG24CS426 |

## Problem

GPS readings from a moving vehicle are noisy: points drift onto parallel streets, jump across buildings or drop out entirely. PathSnap matches each GPS observation to the road segment the vehicle was actually on, and reconstructs a connected route through the road network.

Planned pipeline:

1. Generate candidate road segments near each GPS point.
2. Score the candidates with a trained classifier.
3. Decode the most likely connected route with an HMM and the Viterbi algorithm.
4. Serve predictions through a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

## Repository layout

```
├── configs/            # paths, seeds, parameters
├── data/
│   └── raw/            # dataset as provided: train/ and student_test/
├── notebooks/          # EDA and experiments
├── src/
│   ├── data/           # loading, cleaning, per-point label alignment
│   ├── features/       # candidate generation, feature engineering
│   ├── models/         # training and evaluation of the models
│   ├── tuning/         # Bayesian hyperparameter optimisation
│   ├── hmm/            # HMM and Viterbi route reconstruction
│   └── viz/            # Folium maps and plots
├── api/                # FastAPI backend (Laptop 1)
├── ui/                 # evaluator interface (Laptop 2)
├── artifacts/          # final model, preprocessing objects, HMM config
└── results/            # CSVs, experiment logs, optimisation trials, figures, maps
```

## Data

The dataset is included as provided, under `data/raw/`:

```
data/raw/
├── train/                     # 80 trips
│   └── 00000001/
│       ├── 00000001.track     # longitude, latitude, time (s)
│       ├── 00000001.nodes     # longitude, latitude (row index = node id)
│       ├── 00000001.arcs      # from_node, to_node (row index = arc id)
│       └── 00000001.route     # ground-truth arc ids, in travel order
├── student_test/              # 10 trips, same format
└── eval/                      # 10 evaluator trips; keep separate
```

Paths are set in [`configs/config.yaml`](configs/config.yaml).

## Loading and inspecting trips

Each numbered trip folder is one trip/case; the partition folders already define the training, student-testing, and evaluator-testing sets. The 80/10/10 values are trip counts, not a split ratio to apply again. `load_split` reads all trip folders directly inside one existing partition:

```python
from src.data.loaders import load_split

train_trips = load_split("data/raw/train")
student_test_trips = load_split("data/raw/student_test")
```

Keep the supplied partitions intact: never move observations or trips between them. Use training trips for fitting and cross-validation, student-testing trips for development-stage comparison, and evaluator-testing data only for final evaluation. Do not use evaluator labels for training, feature selection, or model selection.

`validate_trip` returns a list of issues for one trip, and `validate_trips` returns issues keyed by trip ID. Empty files load as empty DataFrames so validation can report them. `clean_trips` drops and warns about a trip if any required table is empty; missing files or required columns raise an error instead. For retained trips, cleaning removes invalid or duplicate GPS observations and sorts tracks by time; it leaves the nodes, arcs, and route tables unchanged. Validation can be run again afterward.

Run the data, feature, and model tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Models

The model modules provide Logistic Regression, Random Forest, KNN, MLP, and
LightGBM classifiers for candidate-arc features. They accept a numeric feature
matrix `X` and candidate labels `y`; `groups` contains the trip ID for each
candidate row. Labels should use `1` for the ground-truth candidate and `0` for
other candidates when computing observation-level top-1 accuracy.

```python
from src.models.registry import build_model_registry
from src.models.report import generate_model_reports
from src.models.train import cross_validate_models, train_model

models = build_model_registry(
    model_names=("logistic_regression", "random_forest")
)
fitted_model = train_model(models["logistic_regression"], X_train, y_train)

results = cross_validate_models(
    X_train,
    y_train,
    groups=trip_ids,
    model_names=("logistic_regression", "random_forest"),
)

report = generate_model_reports(
    X_train,
    y_train,
    X_student_test,
    y_student_test,
    trip_groups=train_trip_ids,
    feature_names=feature_names,
    output_dir="results/model_reports",
)
print(report["csv_path"], report["markdown_path"])
```

Cross-validation keeps all candidate rows from a trip together and reports
fold-level and mean/std macro-F1, candidate accuracy, log loss, and timing.
`evaluate_classifier` can additionally calculate per-observation top-1 accuracy
when passed an `observation_ids` value for every candidate row. Feature
generation and `.route` alignment helpers are in `src/features`; combine their
outputs into numeric feature matrices and labels before calling the model APIs.
The report generator writes a CSV row per model and a detailed Markdown report
for the model with the highest student-test macro-F1. Keep evaluator data out of
this comparison. LightGBM is imported only when its model is requested.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Running

Generate model reports from the supplied training and student-test partitions
from the repository root:

```bash
python3 report.py
```

The script builds candidate features and route-aligned labels, runs trip-grouped
cross-validation, writes `results/model_reports/model_comparison.csv`, and
selects the best model by student-test macro-F1 for
`results/model_reports/best_model_report.md`. To keep the default run practical,
it uses at most 25 evenly spaced observations per trip and 3 cross-validation
folds, retaining all candidates for each sampled observation. The report records
this sampling. Use `--max-observations-per-trip 0` and `--folds 5` to use all
observations for a longer, full-data run. Use `--models` to choose models. If
LightGBM is not installed in the active Python environment, the script prints a
notice and reports the other installed models.
