import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Any, List, Tuple
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
            # Only read the 'order_id' column for efficiency
            orders_df = pd.read_csv(order_path, usecols=['order_id'])
        except FileNotFoundError:
            print(f"ERROR: Order file {order_path} not found.")
            return 0

        total_orders = len(orders_df)
        if self.sampling_rate >= 1.0:
            # Return max ID if no sampling or full sampling is requested
            return orders_df['order_id'].max()
        if self.sampling_rate <= 0:
            return 0

        orders_df = orders_df.sort_values(by='order_id', ascending=True)

        # Calculate the index corresponding to the sampling rate
        boundary_index = int(total_orders * self.sampling_rate)

        # Handle edge cases for boundary index
        if boundary_index >= total_orders:
            return orders_df['order_id'].max()

        # The boundary ID is the order_id at the calculated index 
        return orders_df.iloc[boundary_index - 1]['order_id']

    def load_data(self) -> Union[InstacartData, None]:
        """Loads all DataFrames. Orders and Prior are filtered by time."""
        print(f"Loading Instacart data with time-based {self.sampling_rate * 100:.2f}% sampling...")

        time_boundary_id = 0
        order_path = self.file_paths.get('orders')

        if self.sampling_rate < 1.0 and order_path:
            # Calculate boundary if sampling is applied
            time_boundary_id = self._calculate_time_boundary(order_path)
            print(f"Calculated Order ID boundary for {self.sampling_rate*100:.0f}%: {time_boundary_id}")


        for name, path in self.file_paths.items():
            try:
                # Load metadata files completely
                if name in ['products', 'aisles', 'departments']:
                    self.data[name] = pd.read_csv(path)
                    print(f"Loaded {name} (full).")
                
                # Load orders and apply time-based filter
                elif name == 'orders':
                    df = pd.read_csv(path)
                    if time_boundary_id > 0:
                        # Filter all orders that chronologically exceed the boundary ID
                        self.data[name] = df[df['order_id'] <= time_boundary_id].copy()
                    else:
                         self.data[name] = df
                         
                # Load prior purchases
                elif name == 'prior':
                    # Load the large 'prior' dataset first
                    self.data[name] = pd.read_csv(path)

                # Default case for any other files
                else:
                    self.data[name] = pd.read_csv(path)

            except FileNotFoundError:
                print(f"ERROR: File {path} not found. Please ensure all files are in the path.")
                return None
        
        # Apply filter to 'prior' based on the sampled 'orders'
        if 'prior' in self.data and 'orders' in self.data and time_boundary_id > 0:
            sampled_order_ids = self.data['orders']['order_id'].unique()
            initial_rows = self.data['prior'].shape[0]
            # Filter 'prior' purchases to include only those from the sampled orders
            self.data['prior'] = self.data['prior'][self.data['prior']['order_id'].isin(sampled_order_ids)].copy()
            filtered_rows = self.data['prior'].shape[0]
            
            print(f"Loaded orders: {len(sampled_order_ids)} orders.")
            print(f"Prior data filtered to {filtered_rows} rows (from {initial_rows}) to match time boundary.")

        print("Loading complete.")
        return InstacartData(self.data)


class DataPreprocessor:
    """
    Handles feature engineering, time series creation, and graph construction.
    """
    def __init__(self, instacart_data: InstacartData):
        self._data = instacart_data
        self.product_nodes: List[int] = self._data.get_product_nodes()
        self.product_id_to_index: Dict[int, int] = {pid: i for i, pid in enumerate(self.product_nodes)}
        self.node_features_X: np.ndarray | None = None
        self.prior_data_for_graph = self._data.get_df('prior')[['order_id', 'product_id']].drop_duplicates()

    def _create_metadata_features(self) -> pd.DataFrame:
        """
        Helper method to merge product, aisle, and department metadata, then one-hot encode.
        """
        # Merge metadata
        products_df = self._data.get_df('products')
        aisles_df = self._data.get_df('aisles')
        departments_df = self._data.get_df('departments')

        product_features_df = pd.merge(products_df, aisles_df, on='aisle_id', how='left')
        product_features_df = pd.merge(product_features_df, departments_df, on='department_id', how='left')

        # Prepare for One-Hot-Encode
        product_features_encoded = product_features_df[['product_id', 'aisle_id', 'department_id']].copy()
        
        # Convert IDs to string for one-hot encoding
        product_features_encoded['aisle_id'] = product_features_encoded['aisle_id'].astype(str)
        product_features_encoded['department_id'] = product_features_encoded['department_id'].astype(str)

        # Apply one-hot encoding
        product_features_encoded = pd.get_dummies(
            product_features_encoded,
            columns=['aisle_id', 'department_id'],
            prefix=['aisle', 'dept']
        ).set_index('product_id')
        
        # Ensure all product nodes are present and features are correctly ordered
        product_features_encoded = product_features_encoded.reindex(self.product_nodes, fill_value=0)
        
        return product_features_encoded

    def create_node_features(self) -> np.ndarray:
        """
        Creates the initial product node features based on Aisle/Department metadata.
        """
        print("Creating GNN node features...")
        product_features_encoded = self._create_metadata_features()
        
        # Select the columns that were one-hot encoded
        feature_cols = [col for col in product_features_encoded.columns if col.startswith(('aisle_', 'dept_'))]
        
        # Save the Feature Matrix
        self.node_features_X = product_features_encoded[feature_cols].values
        
        print(f"Shape of Node Feature Matrix (X): {self.node_features_X.shape}")
        return self.node_features_X

    def create_demand_time_series(self, num_weeks: int = 100, split_ratio: float = 0.8, temporal_agg_factor: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates demand time series and performs the train/test split.
        `temporal_agg_factor` controls the aggregation (Standard: 4 for a monthly-like slot).
        """
        prior_df = self._data.get_df('prior')
        orders_df = self._data.get_df('orders')
        
        num_initial_slots = num_weeks * temporal_agg_factor
        num_final_slots = num_weeks

        print(f"Creating demand time series (Fine slots: {num_initial_slots}, Final slots: {num_final_slots} aggregated slots)...")
        
        # Only merge the orders that exist in the sampled set (already filtered in DataLoader)
        df_merged = pd.merge(prior_df, orders_df[['order_id']], on='order_id', how='inner')

        # Creation of the initial, finer slots based on Order ID
        max_order_id = df_merged['order_id'].max()
        # Define bins for discretization of order IDs into time slots
        bins = np.linspace(df_merged['order_id'].min(), max_order_id, num_initial_slots + 1)
        
        df_merged['week_slot_fine'] = pd.cut(
            df_merged['order_id'],
            bins=bins,
            labels=False,
            include_lowest=True
        ).astype('Int64') + 1

        # Creation of the aggregated slots
        df_merged['week_slot'] = ((df_merged['week_slot_fine'] - 1) // temporal_agg_factor) + 1
        
        # Count demand per product per aggregated slot
        demand_ts = df_merged.groupby(['week_slot', 'product_id']).size().reset_index(name='demand')
        # Pivot to create the Time Series matrix (Time Slots x Products)
        demand_matrix = demand_ts.pivot(index='week_slot', columns='product_id', values='demand').fillna(0)
        demand_matrix = demand_matrix.head(num_final_slots)
        demand_matrix = demand_matrix.reindex(columns=self.product_nodes, fill_value=0)

        split_point = int(demand_matrix.shape[0] * split_ratio)
        train_ts = demand_matrix.iloc[:split_point, :]
        test_ts = demand_matrix.iloc[split_point:, :]

        print(f"Time series matrix shape: {demand_matrix.shape}. Train/Test split: {train_ts.shape[0]}/{test_ts.shape[0]} final slots.")
        return train_ts, test_ts

    def build_graph_data(self, threshold: int = 10) -> Data:
        """
        Constructs the static product graph based on co-purchase patterns.
        """
        print(f"Calculating co-purchase edges (threshold >= {threshold})...")
        
        # Self-Join to generate all product pairs within each order
        df_pairs = pd.merge(
            self.prior_data_for_graph,
            self.prior_data_for_graph,
            on='order_id',
            suffixes=('_A', '_B')
        )

        # Exclude self-loops (product_id_A == product_id_B) and only keep unique direction (A < B). This ensures we count each unique co-purchase pair only once
        df_pairs = df_pairs[df_pairs['product_id_A'] < df_pairs['product_id_B']]

        # Count the frequency of each unique pair
        co_purchase_counts = df_pairs.groupby(['product_id_A', 'product_id_B']).size().reset_index(name='co_frequency')
        
        # Filter by co-purchase threshold
        co_purchase_edges = co_purchase_counts[co_purchase_counts['co_frequency'] >= threshold]
        
        print(f"Found edge pairs (one-sided): {len(co_purchase_edges)}")
        
        edge_index_list = []
        edge_weight_list = []
        num_nodes = len(self.product_id_to_index)
        
        for _, row in co_purchase_edges.iterrows():
            u = self.product_id_to_index.get(row['product_id_A'])
            v = self.product_id_to_index.get(row['product_id_B'])
            weight = row['co_frequency']
            
            if u is not None and v is not None:
                # Co-purchase graph is undirected, add both directions to the edge list
                edge_index_list.append([u, v])
                edge_index_list.append([v, u])
                
                # Assign weights for both directions
                edge_weight_list.append(weight)
                edge_weight_list.append(weight)

        if not edge_index_list:
             # Create empty tensors if no edges were found
             edge_index = torch.empty((2, 0), dtype=torch.long)
             edge_weight = torch.empty(0, dtype=torch.float)
        else:
             # Convert the list of [u, v] pairs to a (2, num_edges) tensor for PyG
             edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
             edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

        # Convert node features (X) to a torch tensor
        x = torch.tensor(self.node_features_X, dtype=torch.float)

        # Create the PyG Data object
        product_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
        
        print(f"PyG Graph created: {product_graph}")
        return product_graph