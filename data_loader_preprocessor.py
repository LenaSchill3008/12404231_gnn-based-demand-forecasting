import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Any, List, Tuple, Union
import os

# Base class for Instacart Data Operations 
class InstacartData:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    def get_df(self, name: str) -> pd.DataFrame:
        df = self.data.get(name)
        return df

    def get_product_nodes(self) -> List[int]:
        return self.get_df('products')['product_id'].unique().tolist()

class DataLoader:
    """
    Loads necessary Instacart CSV files and applies time-based sampling.
    """
    def __init__(self, file_paths: Dict[str, str], sampling_rate: float = 1.0):
        self.file_paths = file_paths
        self.sampling_rate = sampling_rate
        self.data: Dict[str, pd.DataFrame] = {}

    def _calculate_time_boundary(self, order_path: str) -> int:
        """
        Calculates the maximum Order ID to include based on the sampling rate.
        """
        try:
            orders_df = pd.read_csv(order_path, usecols=['order_id'])
        except FileNotFoundError:
            print(f"ERROR: Order file {order_path} not found.")
            return 0

        total_orders = len(orders_df)
        if self.sampling_rate >= 1.0:
            return orders_df['order_id'].max()
        if self.sampling_rate <= 0:
            return 0

        orders_df = orders_df.sort_values(by='order_id', ascending=True)
        boundary_index = int(total_orders * self.sampling_rate)
        
        if boundary_index >= total_orders:
            return orders_df['order_id'].max()
        
        return orders_df.iloc[boundary_index - 1]['order_id']

    def load_data(self) -> Union[InstacartData, None]:
        """Loads all DataFrames. Orders and Prior are filtered by time."""
        print(f"Loading Instacart data with time-based {self.sampling_rate * 100:.2f}% sampling...")

        time_boundary_id = 0
        order_path = self.file_paths.get('orders')

        if self.sampling_rate < 1.0 and order_path:
            time_boundary_id = self._calculate_time_boundary(order_path)
            print(f"Calculated Order ID boundary for {self.sampling_rate*100:.0f}%: {time_boundary_id}")

        for name, path in self.file_paths.items():
            try:
                if name in ['products', 'aisles', 'departments']:
                    self.data[name] = pd.read_csv(path)
                
                elif name == 'orders':
                    df = pd.read_csv(path)
                    if time_boundary_id > 0:
                        self.data[name] = df[df['order_id'] <= time_boundary_id].copy()
                    else:
                         self.data[name] = df
                         
                elif name == 'prior':
                    self.data[name] = pd.read_csv(path)
                
                else:
                    self.data[name] = pd.read_csv(path)

            except FileNotFoundError:
                print(f"ERROR: File {path} not found. Please ensure all files are in the path.")
                return None
        
        if 'prior' in self.data and 'orders' in self.data and time_boundary_id > 0:
            sampled_order_ids = self.data['orders']['order_id'].unique()
            initial_rows = self.data['prior'].shape[0]
            self.data['prior'] = self.data['prior'][self.data['prior']['order_id'].isin(sampled_order_ids)].copy()
            filtered_rows = self.data['prior'].shape[0]
            
            print(f"Prior data filtered to {filtered_rows} rows (from {initial_rows}) to match time boundary.")

        print("Loading complete.")
        return InstacartData(self.data)


class DataPreprocessor:
    """
    Handles feature engineering, time series creation, scaling, and graph construction.
    """
    def __init__(self, instacart_data: InstacartData):
        self._data = instacart_data
        self.product_nodes: List[int] = self._data.get_product_nodes()
        self.product_id_to_index: Dict[int, int] = {pid: i for i, pid in enumerate(self.product_nodes)}
        self.node_features_X: np.ndarray | None = None
        self.prior_data_for_graph = self._data.get_df('prior')[['order_id', 'product_id']].drop_duplicates()
        self.mean_ts: torch.Tensor | None = None 
        self.std_ts: torch.Tensor | None = None   

    def _create_metadata_features(self) -> pd.DataFrame:
        """
        Helper method to merge product, aisle, and department metadata, then one-hot encode.
        """
        products_df = self._data.get_df('products')
        aisles_df = self._data.get_df('aisles')
        departments_df = self._data.get_df('departments')

        product_features_df = pd.merge(products_df, aisles_df, on='aisle_id', how='left')
        product_features_df = pd.merge(product_features_df, departments_df, on='department_id', how='left')

        product_features_encoded = product_features_df[['product_id', 'aisle_id', 'department_id']].copy()
        
        product_features_encoded['aisle_id'] = product_features_encoded['aisle_id'].astype(str)
        product_features_encoded['department_id'] = product_features_encoded['department_id'].astype(str)

        product_features_encoded = pd.get_dummies(
            product_features_encoded,
            columns=['aisle_id', 'department_id'],
            prefix=['aisle', 'dept']
        ).set_index('product_id')
        
        product_features_encoded = product_features_encoded.reindex(self.product_nodes, fill_value=0)
        
        return product_features_encoded

    def create_node_features(self) -> np.ndarray:
        """
        Creates the initial product node features.
        """
        print("Creating GNN node features...")
        product_features_encoded = self._create_metadata_features()

        # Feature for global frequency 
        prior_df = self._data.get_df('prior')
        product_counts = prior_df.groupby('product_id').size().reset_index(name='global_frequency')
        product_counts = product_counts.set_index('product_id')
        
        # Merge and fill null values
        product_features_encoded = product_features_encoded.merge(
            product_counts, 
            left_index=True, 
            right_index=True, 
            how='left'
        ).fillna(0)
        
        # Normalize frequency to [0, 1]
        max_freq = product_features_encoded['global_frequency'].max()
        if max_freq > 0:
            product_features_encoded['global_frequency_norm'] = product_features_encoded['global_frequency'] / max_freq
        else:
            product_features_encoded['global_frequency_norm'] = 0
            
        # Select feature columns
        feature_cols = [
            col for col in product_features_encoded.columns 
            if col.startswith(('aisle_', 'dept_', 'global_frequency_norm'))
        ]
        
        # Store the feature matrix and ensure it is of type float32 for PyTorch compatibility
        self.node_features_X = product_features_encoded[feature_cols].values.astype(np.float32)
        
        print(f"Shape of Node Feature Matrix (X): {self.node_features_X.shape}")
        return self.node_features_X

    def create_demand_time_series(self, num_weeks: int = 100, split_ratio: float = 0.8, temporal_agg_factor: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates demand time series by aggregating order data into temporal slots.
        """
        prior_df = self._data.get_df('prior')
        orders_df = self._data.get_df('orders')
        
        num_initial_slots = num_weeks * temporal_agg_factor
        num_final_slots = num_weeks

        print(f"Creating demand time series (Fine slots: {num_initial_slots}, Final slots: {num_final_slots} aggregated slots)...")
        
        df_merged = pd.merge(prior_df, orders_df[['order_id']], on='order_id', how='inner')

        max_order_id = df_merged['order_id'].max()
        bins = np.linspace(df_merged['order_id'].min(), max_order_id, num_initial_slots + 1)
        
        df_merged['week_slot_fine'] = pd.cut(
            df_merged['order_id'],
            bins=bins,
            labels=False,
            include_lowest=True
        ).astype('Int64') + 1

        df_merged['week_slot'] = ((df_merged['week_slot_fine'] - 1) // temporal_agg_factor) + 1
        
        demand_ts = df_merged.groupby(['week_slot', 'product_id']).size().reset_index(name='demand')
        demand_matrix = demand_ts.pivot(index='week_slot', columns='product_id', values='demand').fillna(0)
        demand_matrix = demand_matrix.head(num_final_slots)
        demand_matrix = demand_matrix.reindex(columns=self.product_nodes, fill_value=0)

        split_point = int(demand_matrix.shape[0] * split_ratio)
        train_ts = demand_matrix.iloc[:split_point, :]
        test_ts = demand_matrix.iloc[split_point:, :]

        print(f"Time series matrix shape: {demand_matrix.shape}. Train/Test split: {train_ts.shape[0]}/{test_ts.shape[0]} final slots.")
        return train_ts, test_ts

    def scale_time_series(self, train_ts: pd.DataFrame, test_ts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the raw count time series without scaling.
        """
        self.mean_ts = torch.zeros(train_ts.shape[1], dtype=torch.float)
        self.std_ts = torch.ones(train_ts.shape[1], dtype=torch.float)

        return train_ts.copy(), test_ts.copy()

    def inverse_scale_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Since scaling is removed, this function only ensures non-negativity for count data.
        """
        unscaled_tensor = predictions
        unscaled_tensor = torch.relu(unscaled_tensor)
        
        return unscaled_tensor

    def _save_graph(self, graph_data: Data, path: str):
        """Saves the PyG Data object to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(graph_data, path)
        print(f"PyG Graph saved to: {path}")

    def _load_graph(self, path: str) -> Union[Data, None]:
        """Loads the PyG Data object from the cache if it exists."""
        if os.path.exists(path):
            try:
                # Ensure node features are created before loading to check consistency
                if self.node_features_X is None:
                    self.create_node_features()

                graph_data = torch.load(path)
                
                # Check for dimensional consistency
                if graph_data.num_nodes == len(self.product_nodes) and graph_data.x.shape[1] == self.node_features_X.shape[1]:
                    print(f"PyG Graph successfully loaded from cache: {path}")
                    self.node_features_X = graph_data.x.cpu().numpy()
                    return graph_data
                else:
                    print("WARNING: Loaded graph has inconsistent dimensions. Rebuilding.")
                    return None
            except Exception as e:
                print(f"Error loading graph cache: {e}. Rebuilding.")
                return None
        return None

    def build_graph_data(self, threshold: int = 10, cache_path: str = 'cache/product_graph.pt') -> Data:
        """
        Constructs the static product graph or loads it from the cache, if available.
        """
        # Attempt to load graph from cache
        if os.path.exists(cache_path):
             # Ensure node features are computed 
             if self.node_features_X is None:
                  self.create_node_features()
             cached_graph = self._load_graph(cache_path)
             if cached_graph is not None:
                  return cached_graph

        # If not in cache, perform the computationally expensive construction
        print(f"Calculating co-purchase edges (threshold >= {threshold})...")
        
        # Ensure node features are computed 
        if self.node_features_X is None:
             self.create_node_features()
             
        df_pairs = pd.merge(
            self.prior_data_for_graph,
            self.prior_data_for_graph,
            on='order_id',
            suffixes=('_A', '_B')
        )

        df_pairs = df_pairs[df_pairs['product_id_A'] < df_pairs['product_id_B']]
        co_purchase_counts = df_pairs.groupby(['product_id_A', 'product_id_B']).size().reset_index(name='co_frequency')
        co_purchase_edges = co_purchase_counts[co_purchase_counts['co_frequency'] >= threshold].copy()

        max_freq = co_purchase_edges['co_frequency'].max()
        if max_freq > 0:
            co_purchase_edges['normalized_weight'] = co_purchase_edges['co_frequency'] / max_freq
        else:
            co_purchase_edges['normalized_weight'] = 0 
            
        print(f"Found edge pairs (one-sided): {len(co_purchase_edges)}")
        
        edge_index_list = []
        edge_weight_list = []
        num_nodes = len(self.product_id_to_index)
        
        for _, row in co_purchase_edges.iterrows():
            u = self.product_id_to_index.get(row['product_id_A'])
            v = self.product_id_to_index.get(row['product_id_B'])
            weight = row['normalized_weight']
            
            if u is not None and v is not None:
                # Create undirected edges
                edge_index_list.append([u, v])
                edge_index_list.append([v, u])
                edge_weight_list.append(weight)
                edge_weight_list.append(weight)

        if not edge_index_list:
             # Handle case with no edges
             edge_index = torch.empty((2, 0), dtype=torch.long)
             edge_weight = torch.empty(0, dtype=torch.float)
        else:
             edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
             edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

        # Node features (X)
        x = torch.tensor(self.node_features_X, dtype=torch.float)

        product_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
        
        print(f"PyG Graph created: {product_graph}")
        
        # Save graph before returning
        self._save_graph(product_graph, cache_path)
        
        return product_graph