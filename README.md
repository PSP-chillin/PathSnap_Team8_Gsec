# PathSnap: Map-Matching & Route Reconstruction

**Team 8** · UE24CS352A Machine Learning · PES University, RR Campus

| Team member | SRN |
|---|---|
| _TBD_ | _TBD_ |

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
│   └── raw/            # dataset goes here (not committed)
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

The raw dataset is **not committed** to this repository. Place it locally as:

```
data/raw/
├── train/                     # 80 trips
│   └── 00000001/
│       ├── 00000001.track     # longitude, latitude, time (s)
│       ├── 00000001.nodes     # longitude, latitude (row index = node id)
│       ├── 00000001.arcs      # from_node, to_node (row index = arc id)
│       └── 00000001.route     # ground-truth arc ids, in travel order
└── student_test/              # 10 trips, same format
```

Paths are set in [`configs/config.yaml`](configs/config.yaml).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Running

_To be added: EDA, training, tuning, API and UI commands._
