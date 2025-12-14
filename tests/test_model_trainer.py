import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os
import sys

# Add root directory to Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GNNForecastingModel, ModelTrainer 
from torch_geometric.data import Data 


@pytest.fixture
def mock_model():
    """Mocks the GNNForecastingModel for isolated trainer tests."""
    model = Mock(spec=GNNForecastingModel)
    model.input_len = 2
    model.output_len = 1
    model.criterion = nn.PoissonNLLLoss(log_input=False, reduction='mean')
    
    optimizer_mock = Mock(spec=torch.optim.Adam)
    model.optimizer = optimizer_mock 
    
    # Mock output must require grad to prevent RuntimeError during backward()
    mock_output_tensor = torch.tensor([[[5.0], [5.0], [5.0]]], dtype=torch.float, requires_grad=True)
    model.forward.return_value = mock_output_tensor
    model.to.return_value = model 
    
    return model

@pytest.fixture
def trainer(mock_model):
    """Instantiates the ModelTrainer with the mocked model."""
    return ModelTrainer(mock_model)

@pytest.fixture
def mock_timeseries_data():
    """Creates a simple time series matrix (4 slots, 3 products)."""
    data = np.array([
        [10.0, 20.0, 30.0], # T1
        [11.0, 21.0, 31.0], # T2
        [12.0, 22.0, 32.0], # T3
        [13.0, 23.0, 33.0]  # T4
    ])
    return pd.DataFrame(data) 

@pytest.fixture
def mock_graph_data():
    """Creates mock PyG Graph data, including edge_attr."""
    return Data(
        x=torch.zeros(3, 5), 
        edge_index=torch.tensor([[0, 1], [1, 0]]), 
        edge_attr=torch.ones(2, dtype=torch.float), 
        num_nodes=3
    )


class TestModelTrainer:

    def test_create_sequences(self, trainer, mock_timeseries_data):
        """Tests correct creation of input/target sequences (sliding window)."""
        X_train, Y_train = trainer._create_sequences(mock_timeseries_data)

        assert X_train.shape == (2, 2, 3) 
        assert Y_train.shape == (2, 1, 3) 
        
        # Convert the Pandas slice to a Tensor for assertion (prevents TypeError)
        target_tensor = torch.tensor(mock_timeseries_data.iloc[2:3].values, dtype=torch.float).reshape(1, 3)
        assert torch.allclose(Y_train[0], target_tensor)


    def test_train_model_call(self, trainer, mock_graph_data, mock_timeseries_data, mock_model, tmpdir):
        """Tests if the training loop successfully calls model and optimizer methods."""
        
        with patch.object(trainer, 'save_checkpoint') as mock_save:
            trainer.train_model(
                graph_data=mock_graph_data, 
                train_ts=mock_timeseries_data, 
                epochs=1, 
                checkpoint_dir=str(tmpdir), 
                batch_size=1
            )
            
            assert mock_model.forward.call_count == 2
            assert mock_model.optimizer.step.call_count == 2 
            
            assert mock_save.call_count == 1