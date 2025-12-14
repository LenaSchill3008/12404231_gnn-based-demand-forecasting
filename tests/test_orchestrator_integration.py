import pytest
import pandas as pd
import numpy as np
import torch # Required for mocking torch objects
from unittest.mock import patch, MagicMock
import sys
import os

# Add root directory to Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Orchestrator, load_config

# Mock-Konfiguration for fast tests
@pytest.fixture
def mock_config():
    """Provides a minimal config dictionary for Orchestrator initialization."""
    return {
        'DATA': {
            'FILE_PATHS': {
                'orders': 'mock_path', 
                'prior': 'mock_path', 
                'products': 'mock_path',
                'aisles': 'mock_path', 
                'departments': 'mock_path' 
            },
            'SAMPLING_FRACTION': 0.1,
            'TIME_SERIES_SLOTS': 10,
            'SPLIT_RATIO': 0.8,
            'TEMPORAL_AGG_FACTOR': 1,
            'EDGE_THRESHOLD': 1
        },
        'TRAINING': {
            'INPUT_SEQUENCE_LENGTH': 5,
            'OUTPUT_SEQUENCE_LENGTH': 1,
            'EPOCHS': 1,
            'BATCH_SIZE': 2,
            'CHECKPOINT_DIR': 'temp_checkpoints'
        },
        'MODEL_DEFAULTS': {
            'HIDDEN_DIM': 16,
            'GNN_LAYERS': 1,
            'LR': 0.01,
            'DROPOUT_RATE': 0.0
        },
        'RUN_GRID_SEARCH': False
    }

# Patching using 'main.ClassName' to target where objects are imported/used.
@patch('main.DataLoader')
@patch('main.DataPreprocessor')
@patch('main.ModelTrainer')
@patch('main.GNNForecastingModel')
@patch('main.load_config') 
def test_orchestrator_single_run_flow(
    mock_load_config, mock_GNNModel, mock_ModelTrainer, mock_DataPreprocessor, mock_DataLoader, mock_config, tmpdir):
    """Tests the complete single training and evaluation pipeline flow."""

    # 0. Set the mock config return value
    mock_load_config.return_value = mock_config 
    
    # 1. DataLoader Setup
    mock_data_instance = MagicMock()
    
    # Mock get_df to return required data structures
    def mock_get_df(name):
        if name == 'products':
            # Provides necessary columns for DataPreprocessor init/feature logic
            return pd.DataFrame({'product_id': [1, 2], 'aisle_id': [10, 20], 'department_id': [1, 1]}) 
        return pd.DataFrame() 
    
    mock_data_instance.get_df.side_effect = mock_get_df
    mock_DataLoader.return_value.load_data.return_value = mock_data_instance
    
    # 2. DataPreprocessor Setup 
    preprocessor_instance = mock_DataPreprocessor.return_value
    preprocessor_instance.product_nodes = [1, 2]
    preprocessor_instance.create_node_features.return_value = np.zeros((2, 5)) 

    preprocessor_instance.create_demand_time_series.return_value = (
        pd.DataFrame(np.ones((8, 2))), 
        pd.DataFrame(np.ones((2, 2)))  
    )
    preprocessor_instance.scale_time_series.side_effect = lambda x, y: (x, y) 
    
    # Graph Mock (must have required tensors for trainer)
    mock_graph_data = MagicMock(x=torch.zeros(2, 5), edge_index=torch.tensor([[0],[1]]))
    preprocessor_instance.build_graph_data.return_value = mock_graph_data

    # 3. ModelTrainer Setup 
    trainer_instance = mock_ModelTrainer.return_value
    trainer_instance.train_model.return_value = None 
    trainer_instance.evaluate_model.return_value = 2.5 

    # ACT
    orchestrator = Orchestrator(mock_load_config.return_value) 
    orchestrator.run()

    # ASSERT: Check method calls
    mock_DataLoader.return_value.load_data.assert_called_once()
    preprocessor_instance.create_node_features.assert_called_once()
    
    mock_ModelTrainer.assert_called_once()
    trainer_instance.train_model.assert_called_once()
    trainer_instance.evaluate_model.assert_called_once()
    
    assert trainer_instance.train_model.call_args[1]['epochs'] == 1