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

`validate_trip` returns a list of issues for one trip, and `validate_trips` returns issues keyed by trip ID. `clean_trip` and `clean_trips` remove invalid or duplicate GPS observations and sort tracks by time; they leave the nodes, arcs, and route tables unchanged. Cleaning emits a warning with the number of removed observations, and validation can be run again afterward.

Run the data pipeline tests from the repository root to see a sample raw and cleaned track:

```bash
python -m unittest discover -s tests -v
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Running

_To be added: EDA, training, tuning, API and UI commands._
