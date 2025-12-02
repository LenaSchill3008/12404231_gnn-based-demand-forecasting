# Project Overview

This project implements a hybrid Graph Neural Network (GNN) and Long Short-Term Memory (LSTM) architecture to predict aggregate market demand for individual products across multiple future time slots. The task is framed as deterministic time series forecasting on graphs. The GNN captures static spatial dependencies (co-purchase relationships), while the LSTM models temporal demand dynamics across the product catalog.

### Key Components
- **Model Architecture:** Hybrid GNN–LSTM with GNN-derived features injected into both the LSTM initial state and the final prediction layer.
- **Data Processing:** Aggregation of order data into a multivariate time series representing demand volume per product per time slot.
- **Graph Construction:** Static co-purchase graph used for neighborhood aggregation via the GNN.

---

# Setup and Execution

## Prerequisites
- Python 3.6+
- Required libraries: `torch`, `pandas`, `numpy`, `pyyaml`, `torch-geometric`, `plotly`, `tqdm`

```bash
pip install -r requirements.txt
```

## Data Requirements
Place the following Instacart dataset files in the directories specified in `config.yaml`:
- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

## Configuration (`config.yaml`)
- All parameters are controlled through `config.yaml`.
- Set `RUN_GRID_SEARCH: True` to enable hyperparameter tuning.
- The model expects parameters under the `MODEL_DEFAULTS` section.

## Execution Workflow

### A. Training / Grid Search
Run the following to train the model or perform hyperparameter search:
```bash
python main.py
```

### B. Inference
Run the following to generate multi-step forecasts for a specific product:
```bash
python inferencing.py
```

The script prompts for a product name and generates a Plotly visualization showing historical demand and the deterministic forecast.

# Work Breakdown Structure and Time Log

| Task                                   | Description                                                                                     | Time (Hours) |
|----------------------------------------|-------------------------------------------------------------------------------------------------|--------------|
| 1. Data Engineering & Aggregation      | Data loading, time-based sampling, OHE features, multivariate demand time series creation       | 10           |
| 2. Graph Construction                  | Co-purchase self-join logic and static PyTorch Geometric graph construction                     | 15           |
| 3. Model Implementation (GNN–LSTM)     | `GNNForecastingModel` definition and coupled forward pass                                       | 30           |
| 4. Training Utilities                  | `ModelTrainer` implementation, evaluation, checkpointing, grid search                           | 5            |
| 5. Inference Module                    | `InferenceCoordinator`, recursive forecasting, Plotly visualization                             | 3            |
| 6. Configuration & Documentation       | Creating `config.yaml` and writing documentation                                                | 2            |
| **TOTAL ESTIMATED TIME**               | Sum of all tasks                                                                                | 65           |
