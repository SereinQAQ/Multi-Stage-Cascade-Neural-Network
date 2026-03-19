# Multi-Stage Cascade Neural Network for Lithium-Ion Cathode Property Prediction

A multi-task deep learning framework that predicts physical and electrochemical properties of lithium-ion cathode materials across a 4-stage manufacturing process.


## Project Structure

```
wcf/
├── dataload.py            # Data loading, preprocessing, PyTorch Dataset
├── model.py               # Neural network architectures (Cascaded, PureMLPModel, OriginalCascaded)
├── util.py                # R2 metric and MetricTracker
├── plot.py                # Training history visualization
├── main.py                # Basic training entry point
├── bo_main.py             # Bayesian / NSGA-II hyperparameter search (Optuna)
├── w.py                   # Unified search or train-best script
├── c.py                   # Physics-causal curriculum learning training
├── train_optimal_model.py # Augmented Lagrangian Method (ALM) adaptive training
└── requirements.txt
```

**Required data file:** `data_fine0.xlsx` with sheets `data` and `scope` in the working directory.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Basic training:**
```bash
python main.py
```

**Bayesian hyperparameter search:**
```bash
python bo_main.py
```

**Curriculum learning training:**
```bash
python c.py
```

**ALM adaptive constraint training:**
```bash
python train_optimal_model.py
```

**Unified search or train-best:**
```bash
python w.py
```

## Models

- `CascadedModel`: 4-stage cascade with configurable fusion (`concat`, `glu`, `attention`, `cross_attention`)
- `PureMLPModel`: Flat MLP baseline concatenating all stage features
- `OriginalCascadedModel`: Simple Markov-chain cascade with ReLU

