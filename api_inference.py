import os
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Union, Tuple
from torch_geometric.data import Data

from model import GNNForecastingModel, ModelTrainer 
from data_loader_preprocessor import DataLoader, DataPreprocessor, InstacartData
from main import load_config 

app = FastAPI(title="GNN Product Demand Forecasting API")
config: Dict[str, Any] = {}
trainer: ModelTrainer | None = None
preprocessor: DataPreprocessor | None = None
product_graph: Data | None = None
test_ts: pd.DataFrame | None = None
input_len: int = 0
output_len: int = 0
temporal_agg_factor: int = 1 
product_name_map: Dict[int, str] = {} 

class PredictionResponse(BaseModel):
    product_id: int
    historical_demand: List[float]
    prediction: List[float]
    input_sequence_length: int
    output_slots_predicted: int 

def recursive_predict(
    trainer: ModelTrainer, 
    graph_data: Data, 
    initial_input_ts: torch.Tensor, 
    steps_to_forecast: int
) -> torch.Tensor:
    """
    Performs recursive multi-step forecasting for an H=1 model. 
    
    """
    trainer.model.eval()
    device = trainer.device
    
    # Ensure static graph features are on the correct device
    x_static, edge_index, edge_attr = trainer._prepare_graph_data(graph_data)
    
    # Initialize the sequence that will be updated: [B=1, L, N]
    current_input = initial_input_ts.clone().to(device)
    
    # Initialize tensor for results: [Steps, N]
    all_predictions = [] 
    
    with torch.no_grad():
        for step in range(steps_to_forecast):
            # Predict the next step (H=1) with output shape: [B=1, N, H=1]
            next_prediction_tensor = trainer.model.forward(x_static, edge_index, edge_attr, current_input)
            
            # Remove all dimensions of size 1. Shape transition: [1, N, 1] -> [N]
            next_prediction_flat = next_prediction_tensor.squeeze().cpu() 
            
            all_predictions.append(next_prediction_flat)
            
            # The prediction must be permuted from [B, N, H] to [B, H, N] for concatenation. Since H=1: Shape [1, 1, N]
            next_input_slot = next_prediction_tensor.permute(0, 2, 1).cpu() 
            
            # Remove the oldest slot and append the newest predicted slot
            current_input = torch.cat(
                (current_input[:, 1:, :].cpu(), next_input_slot), 
                dim=1
            ).to(device)
            
    # Return [Steps, N]
    return torch.stack(all_predictions, dim=0) 


# API Utility Functions
def _load_static_data_and_model():
    """Loads configuration, data, graph, and the trained model."""
    global config, trainer, preprocessor, product_graph, test_ts, input_len, output_len, temporal_agg_factor, product_name_map
    
    # Load Configuration
    config = load_config('config.yaml')
    train_config = config['TRAINING']
    model_defaults = config['MODEL_DEFAULTS']
    data_config = config['DATA']
    
    # Set Global Variables from Configuration
    input_len = train_config['INPUT_SEQUENCE_LENGTH']
    output_len = train_config['OUTPUT_SEQUENCE_LENGTH']
    temporal_agg_factor = data_config['TEMPORAL_AGG_FACTOR']
    cache_dir = config.get('CACHE_DIR', 'cache')

    # Load and Prepare Data
    loader = DataLoader(data_config['FILE_PATHS'], sampling_rate=data_config['SAMPLING_FRACTION'])
    data_instance: Union[InstacartData, None] = loader.load_data()
    if data_instance is None:
        raise RuntimeError("Data loading failed.")

    preprocessor = DataPreprocessor(data_instance) 
    
    # Time series are created to get the test set
    _, test_ts = preprocessor.create_demand_time_series(
        num_weeks=data_config['TIME_SERIES_SLOTS'], 
        split_ratio=data_config['SPLIT_RATIO'],
        temporal_agg_factor=temporal_agg_factor
    )
    
    graph_cache_path = os.path.join(cache_dir, 'product_graph.pt')
    
    product_graph = preprocessor.build_graph_data(
        threshold=data_config['EDGE_THRESHOLD'],
        cache_path=graph_cache_path
    )
    
    # Extract and Store Product Names
    products_df = data_instance.get_df('products')
    if products_df is not None:
         # Filter names to only include products present in the graph nodes
         products_in_graph_df = products_df[
             products_df['product_id'].isin(preprocessor.product_id_to_index.keys())
         ]
         product_name_map = products_in_graph_df.set_index('product_id')['product_name'].to_dict()
         print(f"API: {len(product_name_map)} product names loaded for the graph.")
    
    if product_graph.x is None or product_graph.x.shape[0] == 0:
        raise RuntimeError("Graph initialization failed after loading/creation.")
        
    # 3Initialize Model and Load Checkpoint
    model = GNNForecastingModel( 
        num_nodes=product_graph.num_nodes, 
        node_features=product_graph.x.shape[1], 
        input_len=input_len, 
        output_len=output_len, 
        hidden_dim=model_defaults['HIDDEN_DIM'],
        gnn_layers=model_defaults['GNN_LAYERS'],
        lr=model_defaults['LR'],
        dropout_rate=model_defaults.get('DROPOUT_RATE', 0.3)
    )
    
    trainer = ModelTrainer(model)
    # Load Checkpoint (
    checkpoint_path = os.path.join(train_config['CHECKPOINT_DIR'], 'model.pt')    
    trainer.load_checkpoint(checkpoint_path) 
    trainer.model.eval() 
    
    print("API: Static data and model successfully loaded and evaluated.")

@app.on_event("startup")
async def startup_event():
    try:
        _load_static_data_and_model()
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        raise RuntimeError(f"Failed to load model or data for API: {e}")

# API Endpoints
@app.get("/products")
async def get_product_list() -> Dict[int, str]:
    """Returns a list of all available product IDs and their names."""
    if not product_name_map:
         raise HTTPException(status_code=503, detail="API not initialized or product names not loaded.")
    return product_name_map


@app.post("/predict/{product_id}", response_model=PredictionResponse)
async def predict_demand(product_id: int, forecast_months: int = 1):
    """
    Performs a recursive forecast for a given product (1, 3, or 6 months).
    """
    if trainer is None or product_graph is None or test_ts is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="API not ready (Model/Data not loaded).")
        
    if product_id not in preprocessor.product_id_to_index:
        raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found in graph nodes.")

    if forecast_months not in [1, 3, 6]:
        raise HTTPException(status_code=400, detail="forecast_months must be 1, 3, or 6.")

    # Define recursive steps
    target_output_slots = forecast_months * temporal_agg_factor
    
    # Create input tensor
    product_index = preprocessor.product_id_to_index[product_id]
    
    input_data_df = test_ts.tail(input_len)
    if input_data_df.shape[0] < input_len:
        raise HTTPException(status_code=500, detail="Not enough historical data to form a full input sequence.")
        
    # [L, N] -> [1, L, N] on CPU
    initial_input_tensor = torch.tensor(input_data_df.values, dtype=torch.float).unsqueeze(0).cpu() 

    # Perform recursive inference
    try:
        predictions_all_nodes_stacked = recursive_predict(
            trainer=trainer, 
            graph_data=product_graph, 
            initial_input_ts=initial_input_tensor, 
            steps_to_forecast=target_output_slots
        )
    except Exception as e:
        print(f"ERROR during recursive prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")
    
    # Extract prediction for the specific product shape [target_output_slots]
    prediction_ts = predictions_all_nodes_stacked[:, product_index].numpy()
    
    # Extract historical data
    historical_ts = test_ts.iloc[:, product_index].tolist()

    return PredictionResponse(
        product_id=product_id,
        historical_demand=historical_ts,
        prediction=prediction_ts.tolist(),
        input_sequence_length=input_len,
        output_slots_predicted=target_output_slots
    )

@app.get("/health")
def health_check():
    """Checks if the server and model are loaded."""
    if trainer is None or product_graph is None or not product_name_map:
        return {"status": "error", "message": "Model, Graph or Product Names not loaded."}
    return {"status": "ok", "message": "GNN API is ready."}