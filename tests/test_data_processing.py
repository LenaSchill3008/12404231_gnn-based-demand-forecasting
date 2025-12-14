import pytest
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add root directory to Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader_preprocessor import DataPreprocessor, InstacartData
from torch_geometric.data import Data


@pytest.fixture
def mock_instacart_data():
    """Provides mock Instacart DataFrames for testing preprocessing steps."""
    mock_data = {
        'products': pd.DataFrame({
            'product_id': [1, 2, 3, 4], 
            'aisle_id': [10, 20, 10, 30],
            'department_id': [1, 1, 2, 2],
            'product_name': ['A', 'B', 'C', 'D']
        }),
        'aisles': pd.DataFrame({
            'aisle_id': [10, 20, 30],
            'aisle': ['A_10', 'A_20', 'A_30']
        }),
        'departments': pd.DataFrame({
            'department_id': [1, 2],
            'department': ['D_1', 'D_2']
        }),
        'prior': pd.DataFrame({
            'order_id': [1, 1, 2, 2, 3, 3, 3, 4],
            'product_id': [1, 2, 1, 3, 1, 2, 4, 1] 
        }),
        'orders': pd.DataFrame({
            'order_id': [1, 2, 3, 4], 
            'user_id': [1, 2, 1, 3]
        })
    }
    return InstacartData(mock_data)

@pytest.fixture
def preprocessor(mock_instacart_data):
    """Instantiates the DataPreprocessor."""
    return DataPreprocessor(mock_instacart_data)

class TestDataPreprocessor:

    def test_create_node_features_shape_and_content(self, preprocessor):
        """Tests the creation of GNN features (OHE + Global Freq)."""
        features = preprocessor.create_node_features()
        
        assert features.shape == (4, 6) 
        
        assert features[0, 5] == 1.0 # P1 Frequency (4/4)
        assert features[1, 5] == 0.5 # P2 Frequency (2/4)
        
        assert features[0, 0] == 1 # P1: aisle_10
        assert features[0, 3] == 1 # P1: dept_1


    def test_create_demand_time_series_shape_and_split(self, preprocessor):
        """Tests the creation and split of the time series matrix."""
        train_ts, test_ts = preprocessor.create_demand_time_series(
            num_weeks=4, 
            split_ratio=0.5, 
            temporal_agg_factor=1
        )
        
        assert train_ts.shape == (2, 4) 
        assert test_ts.shape == (2, 4)
        
        assert np.allclose(train_ts.values[0], [1., 1., 0., 0.])
        assert np.allclose(train_ts.values[1], [1., 0., 1., 0.])


    def test_build_graph_data_edges(self, preprocessor):
        """Tests co-purchase edge creation and thresholding."""
        preprocessor.create_node_features() 
        # Only P1-P2 pair meets threshold=2
        product_graph = preprocessor.build_graph_data(threshold=2) 
        
        assert isinstance(product_graph, Data)
        # 1 pair -> 2 directed edges
        assert product_graph.edge_index.shape == (2, 2) 
        
        assert torch.allclose(product_graph.edge_attr, torch.tensor([1.0, 1.0]))
        
        
    def test_build_graph_data_empty(self, preprocessor):
        """Tests graph creation with a threshold that yields no edges."""
        preprocessor.create_node_features() 
        product_graph = preprocessor.build_graph_data(threshold=10) 
        
        assert product_graph.edge_index.shape == (2, 0)