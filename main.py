import os
import torch
import yaml
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, Any, List, Tuple
from data_loader_preprocessor import DataLoader, DataPreprocessor
from model import GNNForecastingModel, ModelTrainer

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads configuration parameters from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Orchestrator:
    """
    Orchestrates the entire GNN forecasting workflow.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.product_graph: Data | None = None
        self.train_ts: pd.DataFrame | None = None
        self.test_ts: pd.DataFrame | None = None
        self.num_nodes: int = 0
        self.node_features_F: int = 0

    def _setup_data_pipeline(self):
        """Executes data loading, preprocessing, and feature/graph creation."""
        print("\n--- 1. DATA LOADING & PREPROCESSING ---")
        data_config = self.config['DATA']
        
        # Data Loading
        loader = DataLoader(data_config['FILE_PATHS'], sampling_rate=data_config['SAMPLING_FRACTION'])
        data_instance = loader.load_data()
        if data_instance is None:
            raise RuntimeError("Data loading failed.")

        # Preprocessing, Features, and Graph Construction
        preprocessor = DataPreprocessor(data_instance)
        
        node_features_X = preprocessor.create_node_features()
        self.num_nodes = len(preprocessor.product_nodes)
        self.node_features_F = node_features_X.shape[1]
        
        self.train_ts, self.test_ts = preprocessor.create_demand_time_series(
            num_weeks=data_config['TIME_SERIES_SLOTS'], 
            split_ratio=data_config['SPLIT_RATIO'],
            temporal_agg_factor=data_config['TEMPORAL_AGG_FACTOR']
        )
        
        self.product_graph = preprocessor.build_graph_data(threshold=data_config['EDGE_THRESHOLD'])
        
        # Print intermediate results
        training_config = self.config['TRAINING']
        input_len = training_config['INPUT_SEQUENCE_LENGTH']
        output_len = training_config['OUTPUT_SEQUENCE_LENGTH']
    

    def _initialize_trainer(self) -> ModelTrainer:
        """Initializes the Model and the Trainer."""
        training_config = self.config['TRAINING']
        model_defaults = self.config['MODEL_DEFAULTS']

        # Initialize model with default hyperparameters
        base_model = GNNForecastingModel( 
            num_nodes=self.num_nodes, 
            node_features=self.node_features_F, 
            input_len=training_config['INPUT_SEQUENCE_LENGTH'], 
            output_len=training_config['OUTPUT_SEQUENCE_LENGTH'],
            hidden_dim=model_defaults['HIDDEN_DIM'],
            gnn_layers=model_defaults['GNN_LAYERS'],
            lr=model_defaults['LR']
        )
        return ModelTrainer(base_model)

    def _execute_single_run(self, trainer: ModelTrainer):
        """Executes a single training and evaluation cycle."""
        training_config = self.config['TRAINING']
        
        print("--- 2. SINGLE MODEL TRAINING & EVALUATION ---")
        
        # Training
        trainer.train_model(
            graph_data=self.product_graph, 
            train_ts=self.train_ts, 
            epochs=training_config['EPOCHS'], 
            checkpoint_dir=training_config['CHECKPOINT_DIR'], 
            batch_size=training_config['BATCH_SIZE']
        )

        # Evaluation
        trainer.evaluate_model(self.product_graph, self.test_ts, batch_size=training_config['BATCH_SIZE'])

    def _execute_grid_search(self, trainer: ModelTrainer):
        """Executes the hyperparameter grid search."""
        training_config = self.config['TRAINING']
        grid_params = self.config.get('GRID_SEARCH', {}).get('PARAMETERS', {})
        
        if not grid_params:
            self._execute_single_run(trainer)
            return

        print("--- 2. STARTING HYPERPARAMETER GRID SEARCH ---")
        
        best_params, best_loss = trainer.grid_search(
            param_grid=grid_params,
            graph_data=self.product_graph,
            train_ts=self.train_ts,
            test_ts=self.test_ts,
            epochs=training_config['EPOCHS'],
            batch_size=training_config['BATCH_SIZE']
        )
        print(f"\nOptimal Configuration found: {best_params} with Test MSE: {best_loss:.6f}")

    def run(self):
        """Main execution method to run the workflow based on the config flag."""
        try:
            self._setup_data_pipeline()
            trainer = self._initialize_trainer()

            if self.config.get('RUN_GRID_SEARCH', False):
                self._execute_grid_search(trainer)
            else:
                self._execute_single_run(trainer)

        except RuntimeError as e:
            print(f"FATAL ERROR during workflow execution: {e}")
            return
        
        print("\n--- EXECUTION COMPLETE ---")


def main():
    """Loads configuration and runs the Orchestrator."""
    try:
        config = load_config()
    except Exception:
        return
        
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()