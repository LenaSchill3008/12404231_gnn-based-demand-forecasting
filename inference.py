import torch
import pandas as pd
import numpy as np 
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import datetime
import yaml
from typing import Dict, Any, Tuple, List, Union
from data_loader_preprocessor import DataLoader, DataPreprocessor, InstacartData
from model import GNNForecastingModel as ForecastingModel 


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads configuration parameters from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_demand_forecast(historic_df: pd.DataFrame, forecast_results: List[float], product_id: int, product_name: str, forecast_steps: int, temporal_agg_factor: int):
    """Generates the Plotly chart of historical demand and the predicted demand line."""
    
    demand_history = historic_df[product_id] 
    START_DATE = pd.to_datetime('2014-01-01')
    slot_duration_days = temporal_agg_factor * 7
    
    time_offset_days = (demand_history.index.to_series() - 1) * slot_duration_days
    dates = START_DATE + pd.to_timedelta(time_offset_days, unit='D')
    
    plot_df = pd.DataFrame({
        'Demand': demand_history.values,
        'Date': dates,
    })
    
    last_historical_date = plot_df['Date'].iloc[-1]
    
    forecast_dates = [
        last_historical_date + pd.Timedelta(days=(step + 1) * slot_duration_days)
        for step in range(forecast_steps)
    ]
    
    # Create DataFrame for prediction line
    forecast_df = pd.DataFrame({
        'Demand': forecast_results,
        'Date': forecast_dates,
    })
    
    fig = go.Figure()

    # Add Historical Demand 
    fig.add_trace(go.Scatter(
        x=plot_df['Date'], y=plot_df['Demand'],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=5, symbol='circle'),
        name='Historical Demand'
    ))

    # Add Predicted Demand 
    mean_line_dates = plot_df['Date'].tolist()[-1:] + forecast_df['Date'].tolist()
    mean_line_demand = plot_df['Demand'].tolist()[-1:] + forecast_df['Demand'].tolist()
    
    fig.add_trace(go.Scatter(
        x=mean_line_dates, y=mean_line_demand,
        mode='lines+markers',
        line=dict(color='darkblue', width=2, dash='dash'),
        marker=dict(size=8, symbol='star', color='darkblue'),
        name='Predicted Demand'
    ))
    
    # 4. Final Layout Configuration
    fig.update_layout(
        title=f'Deterministic Demand Forecast ({forecast_steps} Slots): {product_name}',
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
        hovermode="x unified"
    )
    
    fig.show()
    print(f"Plot for '{product_name}' displayed in browser.")


class InferenceCoordinator:
    """
    Coordinates the loading of the trained model and data, and executes 
    the iterative, multi-step demand forecasting process using the deterministic model.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
    
        self.training_config = config['TRAINING']
        self.data_config = config['DATA']
        self.model_defaults = config['MODEL_DEFAULTS']
        self.input_len = self.training_config['INPUT_SEQUENCE_LENGTH']
        self.output_len = self.training_config['OUTPUT_SEQUENCE_LENGTH']
        self.hidden_dim = self.model_defaults['HIDDEN_DIM']
        self.gnn_layers = self.model_defaults['GNN_LAYERS']
        self.checkpoint_path = config.get('CHECKPOINT_PATH', 'model_checkpoints/model_checkpoint_e50.pt')
        self.forecast_steps = config.get('FORECAST_STEPS', 3)
        self.temporal_agg_factor = self.data_config['TEMPORAL_AGG_FACTOR']

        self.model: Union[ForecastingModel, None] = None
        self.product_graph: Union[Data, None] = None
        self.full_ts: Union[pd.DataFrame, None] = None
        self.products_df: Union[pd.DataFrame, None] = None
        self.product_id_to_index: Union[Dict[int, int], None] = None

    def _load_data_and_model(self):
        """Loads data, initializes the model, and restores weights."""
        print("1. LOADING DATA AND PREPROCESSOR")
        
        loader = DataLoader(self.data_config['FILE_PATHS'], sampling_rate=self.data_config['SAMPLING_FRACTION'])
        data_instance = loader.load_data() 
        if data_instance is None:
            sys.exit(1)

        preprocessor = DataPreprocessor(data_instance)
        node_features_X = preprocessor.create_node_features()
        num_nodes = len(preprocessor.product_nodes)
        node_features_F = node_features_X.shape[1]
        
        # Load the complete time series matrix (train + test)
        train_ts, test_ts = preprocessor.create_demand_time_series(
            num_weeks=self.data_config['TIME_SERIES_SLOTS'], 
            split_ratio=self.data_config['SPLIT_RATIO'],
            temporal_agg_factor=self.temporal_agg_factor
        )
        self.full_ts = pd.concat([train_ts, test_ts])
        self.products_df = preprocessor._data.get_df('products')
        self.product_id_to_index = preprocessor.product_id_to_index
        
        self.product_graph = preprocessor.build_graph_data(threshold=self.data_config['EDGE_THRESHOLD'])
    
        print("\n2. LOADING MODEL CHECKPOINT")
        self.model = ForecastingModel( 
            num_nodes=num_nodes, 
            node_features=node_features_F, 
            input_len=self.input_len, 
            output_len=self.output_len,
            hidden_dim=self.hidden_dim,
            gnn_layers=self.gnn_layers
        ).to(self.device)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state from {self.checkpoint_path}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Checkpoint file not found at {self.checkpoint_path}. Exiting.")
            sys.exit(1)

        self.model.eval()
        
        # Move static graph tensors to the device
        self.product_graph.x = self.product_graph.x.to(self.device)
        self.product_graph.edge_index = self.product_graph.edge_index.to(self.device)
        self.product_graph.edge_attr = self.product_graph.edge_attr.to(self.device)


    def _get_product_input_data(self, product_name: str) -> Union[Tuple[int, str, int, torch.Tensor, pd.DataFrame], Tuple[None, ...]]:
        """Searches for product ID and extracts the last historical sequence."""
        
        # Search product ID
        product_search = self.products_df[
            self.products_df['product_name'].str.contains(product_name, case=False, na=False)
        ]

        if product_search.empty:
            print(f"ERROR: Product '{product_name}' not found.")
            return None, None, None, None, None

        pid = product_search.iloc[0]['product_id']
        name_found = product_search.iloc[0]['product_name']
        
        # Find product index in the time series matrix 
        if pid not in self.full_ts.columns:
            print(f"ERROR: Product ID {pid} ('{name_found}') is not in the time series matrix.")
            return None, None, None, None, None

        product_index = self.full_ts.columns.get_loc(pid)

        # Extract last input sequence
        last_idx = self.full_ts.shape[0] 
        
        if last_idx < self.input_len:
            print(f"ERROR: Not enough time slots ({last_idx}) for input length {self.input_len}.")
            return None, None, None, None, None

        # Sequence: [Input_Len, Num_Nodes]
        historic_sequence_df = self.full_ts.iloc[last_idx - self.input_len : last_idx, :] 

        # Convert DataFrame to initial tensor [B=1, L, N]
        sequence_tensor = torch.tensor(
            historic_sequence_df.values, 
            dtype=torch.float
        ).unsqueeze(0) 

        return pid, name_found, product_index, sequence_tensor, historic_sequence_df

    def run_inference(self, product_name: str):
        """Executes the multi-step (recursive) forecasting for a specified product."""

        pid, name_found, p_index, input_tensor, historic_sequence_df = self._get_product_input_data(product_name)
        
        if input_tensor is None:
            return

        print(f"\nINFERENCE START (Iterative Forecast)")
        print(f"Product: {name_found} (ID: {pid})")
        print(f"Using {self.input_len} slots history to predict the next {self.forecast_steps} slots.")

        current_input_tensor = input_tensor.to(self.device)
        forecast_results: List[float] = []
        
        x_static = self.product_graph.x
        edge_index = self.product_graph.edge_index
        edge_attr = self.product_graph.edge_attr

        with torch.no_grad():
            for step in range(self.forecast_steps):
                
                # Forward pass returns the single prediction tensor
                predictions = self.model(x_static, edge_index, edge_attr, current_input_tensor)
                
                # Extract the prediction
                predicted_demand_value = predictions[0, p_index, 0].item()
                predicted_demand_rounded = round(max(0, predicted_demand_value))
                
                forecast_results.append(predicted_demand_rounded)

                # Use the prediction as the input for the next time step
                new_time_step_data = predictions.permute(0, 2, 1).cpu() 
                
                # Concatenate the new step and drop the oldest step (sliding window)
                current_input_tensor = torch.cat(
                    (current_input_tensor[:, 1:, :].cpu(), new_time_step_data), 
                    dim=1
                ).to(self.device)

        # Output and Plotting
        print(f"\n[RESULT] Predicted demand for the next {self.forecast_steps} slots:")
        print(f" -> {name_found}: {forecast_results} units")
        print("----------------------")
        
        plot_demand_forecast(
            historic_sequence_df, 
            forecast_results,
            pid, 
            name_found,
            self.forecast_steps,
            self.temporal_agg_factor
        )
        
    def start(self):
        """Initializes components and enters the interactive inference loop."""
        print(f"Initializing on device: {self.device}")

        self._load_data_and_model()

        # Start inference loop
        while True:
            user_input = input(f"\nEnter product name (or 'exit') to forecast the next {self.forecast_steps} slots: ")
            if user_input.lower() == 'exit':
                break
            
            self.run_inference(user_input)


def main():
    """Loads configuration and runs the Inference Coordinator."""

    config_full = load_config()

    inference_config = {
        'CHECKPOINT_PATH': 'model_checkpoints/model_checkpoint_e50.pt', 
        'INPUT_SEQUENCE_LENGTH': config_full['TRAINING']['INPUT_SEQUENCE_LENGTH'],
        'OUTPUT_SEQUENCE_LENGTH': config_full['TRAINING']['OUTPUT_SEQUENCE_LENGTH'],
        'GNN_HIDDEN_DIM': config_full['MODEL_DEFAULTS']['HIDDEN_DIM'],
        'GNN_LAYERS': config_full['MODEL_DEFAULTS']['GNN_LAYERS'],
        'FORECAST_STEPS': config_full.get('FORECAST_STEPS', 3), 
        'DATA': config_full['DATA'],
        'MODEL_DEFAULTS': config_full['MODEL_DEFAULTS'],
        'TRAINING': config_full['TRAINING']
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate and start the engine
    coordinator = InferenceCoordinator(inference_config, DEVICE)
    coordinator.start()


if __name__ == "__main__":
    main()