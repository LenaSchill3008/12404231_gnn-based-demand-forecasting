# Graph-Based Demand Forecasting with GNN–LSTM

## 1. What This Project Is About and Why It Is Necessary

Accurate demand forecasting at the product level is a critical problem in large-scale retail and e-commerce systems. Demand patterns are influenced not only by a product’s own historical sales but also by relationships between products, such as co-purchase behavior. Traditional time series models struggle to incorporate these relational dependencies in a principled way.

This project addresses that gap by framing **aggregate product demand forecasting as a deterministic time series prediction problem on graphs**. A hybrid **Graph Neural Network (GNN) + Long Short-Term Memory (LSTM)** architecture is used to jointly model:

- **Spatial structure**: Static co-purchase relationships between products.
- **Temporal dynamics**: Evolving demand patterns across time.

By combining graph-based representation learning with sequence modeling, the approach enables demand forecasts that are informed by both historical sales trends and the broader product ecosystem.

---

## 2. How the System Is Constructed

### 2.1 Model Architecture

The core model is a hybrid **GNN–LSTM** architecture:

- The **GNN** operates on a static product graph to learn embeddings that encode co-purchase structure and product metadata.
- The **LSTM** models multivariate demand time series across products.
- GNN-derived embeddings are injected into:
  - The **initial hidden state** of the LSTM.
  - The **final prediction layer**, ensuring graph information influences both temporal evolution and output forecasts.

This design allows the model to propagate relational information throughout the forecasting process.

---

### 2.2 Data Preprocessing and Feature Engineering

The preprocessing pipeline converts raw Instacart data into two model inputs: a static graph and a dynamic multivariate time series.

#### Static Graph Features (GNN Input)

- **Nodes**: Each product is represented as a node.
- **Node Features ($\mathbf{X}$)**:
  - One-hot encoded aisle and department metadata.
  - Global product purchase frequency, normalized to $[0, 1]$.
- **Edges**:
  - Undirected edges connect products frequently bought together.
- **Edge Weights ($\mathbf{E}$)**:
  - Normalized co-purchase frequency.
  - Thresholded to remove infrequent co-occurrences.

#### Dynamic Time Series (LSTM Input)

- Orders are aggregated into discrete time slots (e.g., weeks) using `order_id` as a proxy for time.
- A multivariate time series matrix is constructed where:
  - Rows correspond to time slots.
  - Columns correspond to product nodes.
  - Values are raw purchase counts (demand).
- Sliding windows are used to generate supervised input–target pairs.
- No scaling is applied to preserve count data and enable Poisson-based optimization.

---

## 3. How to Use the Project

### 3.1 Setup and Installation

## Prerequisites
- Python 3.6+
- Required libraries: `torch`, `pandas`, `numpy`, `pyyaml`, `torch-geometric`, `plotly`, `tqdm`

```bash
pip install uv
```

```bash
uv pip install -r requirements.txt
```

## Data Requirements
You have to download the csv files from kaggle, create a data folder and paste in the data from https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset in the directories specified in `config.yaml`:
- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

## Configuration (`config.yaml`)
- All parameters are controlled through `config.yaml`.
- Set `RUN_GRID_SEARCH: True` to enable hyperparameter tuning.
- The model expects parameters under the `MODEL_DEFAULTS` section.

### 3.2 Execution Workflow

### A. Training / Grid Search
Run the following to train the model or perform hyperparameter search:
```bash
python main.py
```

### B. Inference
The system includes a FastAPI backend to serve model predictions and a Streamlit frontend for interactive visualization.

Starting the Services
To run the full stack, open two terminal windows:

#### 1. Start the Backend (FastAPI): The API handles the recursive inference logic and serves the GNN-LSTM model.

```bash
uvicorn api_inference:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the Frontend (Streamlit): The dashboard allows you to select products and visualize historical vs. forecasted demand.

```bash
streamlit run ./streamlit_frontend.py
```

## API Endpoints

- GET /products: Returns a mapping of all product IDs and names available in the graph.
- POST /predict/{product_id}: Generates a recursive forecast (1, 3, or 6 months) for the specified product.
- GET /health: Checks if the model, graph, and product metadata are correctly loaded.

## 4.Work Breakdown Structure and Time Log

| Task                                   | Description                                                                                     | Time (Hours) |
|----------------------------------------|-------------------------------------------------------------------------------------------------|--------------|
| 1. Data Engineering & Aggregation      | Data loading, time-based sampling, OHE features, multivariate demand time series creation       | 20           |
| 2. Graph Construction                  | Co-purchase self-join logic and static PyTorch Geometric graph construction                     | 20           |
| 3. Model Implementation (GNN–LSTM)     | `GNNForecastingModel` definition and coupled forward pass                                       | 40           |
| 4. Training Utilities                  | `ModelTrainer` implementation, evaluation, checkpointing, grid search                           | 8            |
| 5. Configuration & Documentation       | Creating `config.yaml` and writing documentation                                                | 2            |
| 6. Configuration & Documentation       | Creating Demo Application and Inference                                                         | 30           |
| **TOTAL ESTIMATED TIME**               | Sum of all tasks                                                                                | 110          |

## 5. Error Metrics and Performance

The primary goal of this project is to predict future demand counts. The training process uses a specialized loss function for optimization, but the model's final performance is measured using an intuitive, unscaled error metric relevant to real-world demand forecasting (RMSE).

---

## Data Constraint

Due to performance and resource constraints, the model was trained and evaluated on only **10% of the total Instacart data**, specifically using **time-based sampling** to ensure sequence integrity. This limits the model's exposure to the complete product catalog and seasonal patterns, contributing to the achieved error metrics. For the final model for assignment 3 I will train the full model on the full dataset on 50 epochs to get more accurate inference results. Since this model is very small I uploaded it to GitHub. For the larger model I will upload to Github LFS.

## A. Training Loss (Optimization Metric)

| Metric | Target | Achieved Value |
|------|-------|----------------|
| Poisson Negative Log-Likelihood (NLL) Loss | Minimize loss value (target is task-dependent and converges to the minimum possible NLL) | **0.011159** |

**Description**

This is the internal loss function used during training. It is optimized because the target variable (demand count) is sparse, non-negative integer data (counts), which is better modeled by a Poisson distribution than a standard Gausian distribution assumed by MSE. The Poisson NLL loss naturally encourages the model to output non-negative predictions (counts). 

## B. Evaluation Metric (Final Performance)

| Metric | Target | Achieved Value |
|------|-------|----------------|
| Root Mean Squared Error (RMSE) on Unscaled Demand | Achieve an RMSE value below a threshold relevant to business constraints (Target RMSE ≤ 2.5 units) | **2.913205** |

**Description**

RMSE is the definitive measure of forecast accuracy, returning the average magnitude of the error in the original units of the demand count (e.g., number of items sold). The squaring of the errors within RMSE ensures that larger forecast errors are penalized more heavily than smaller errors, making it a conservative and robust metric for measuring the consistency of the forecast. The RMSE is calculated directly on the unscaled, true demand values.


