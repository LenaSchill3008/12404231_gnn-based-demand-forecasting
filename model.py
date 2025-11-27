import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Tuple, Dict, Any, List
import itertools

# Helper function to ensure consistency in batch size calculation
def _get_batch_size_and_num_nodes(tensor: torch.Tensor) -> Tuple[int, int, int]:
    """Extracts batch size (B), sequence length (L), and number of nodes (N) from the demand sequence tensor."""
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, L, N], got {tensor.dim()}D tensor.")
    B, L, N = tensor.shape
    return B, L, N

class GNNForecastingModel(nn.Module):
    """
    The complete GNN-LSTM hybrid model for time series forecasting on graphs.
    It uses GNN-derived features for initial state of the LSTM and for the final prediction step.
    """
    def __init__(self, node_features: int, num_nodes: int, input_len: int, output_len: int, hidden_dim: int = 64, gnn_layers: int = 2, lr: float = 0.001):
        super(GNNForecastingModel, self).__init__()

        # Store Hyperparameters
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gnn_layers = gnn_layers

        # Static GNN layers (Spatial component)
        self.gcn_layers = nn.ModuleList()
        input_dim = node_features
        for _ in range(self.gnn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # LSTM for temporal dependency modeling
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim,
            num_layers=1, 
            batch_first=True
        )

        # The final Output Layer processes concatenated features
        # Combined Features = LSTM Output (Hidden_Dim) + GNN Output (Hidden_Dim)
        COMBINED_DIM = hidden_dim * 2
        self.output_layer = nn.Linear(COMBINED_DIM, output_len)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_device(self) -> torch.device:
        """Determines the current device of the model (CPU/GPU)."""
        return next(self.parameters()).device

    def forward(self, x_static: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, demand_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete GNN-Forecasting model.
        """
        B, L, N = _get_batch_size_and_num_nodes(demand_seq)

        # 1. Static GNN Processing (Spatial Aggregation)
        h_static = x_static
        for gcn in self.gcn_layers:
            h_static = torch.relu(gcn(h_static, edge_index, edge_weight=edge_attr))
        # h_static shape: [N, Hidden_Dim]

        # 2. Sequence Preparation for LSTM
        demand_seq_final = demand_seq.permute(0, 2, 1).reshape(-1, L).unsqueeze(-1)

        # 3. LSTM Processing (Temporal Modeling)
        
        # Initial hidden state (h0) and cell state (c0) are initialized with GNN features
        h0_expanded = h_static.unsqueeze(0).expand(B, N, self.hidden_dim)
        h0 = h0_expanded.reshape(B * N, self.hidden_dim).unsqueeze(0).contiguous()
        c0 = torch.zeros_like(h0)

        # Pass through LSTM
        _, (hn, _) = self.lstm(demand_seq_final, (h0, c0))
        final_lstm_output = hn[-1] # Final hidden state [B*N, Hidden_Dim]

        # 4. Feature Concatenation and Prediction

        # Repeat GNN Output to match the batched LSTM output structure
        h_static_repeated = h_static.unsqueeze(0).expand(B, N, self.hidden_dim).reshape(B * N, self.hidden_dim)

        # Concatenation: LSTM output + GNN static feature
        combined_features = torch.cat((final_lstm_output, h_static_repeated), dim=1)

        # Final Linear Layer for multi-step prediction
        predictions_flat = self.output_layer(combined_features) # Shape [B*N, Output_Len (H)]
        # Reshape to the desired output format: [B, N, H] (Batch, Nodes, Prediction Length)
        predictions = predictions_flat.view(B, N, -1)

        return predictions

class ModelTrainer:
    """
    Handles training, evaluation, checkpointing, and data preparation (sequence creation).
    """
    def __init__(self, model: GNNForecastingModel):
        self.model = model
        self.input_len = model.input_len
        self.output_len = model.output_len
        self.criterion = model.criterion
        self.optimizer = model.optimizer
        # Check if CUDA is available for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Saves the current model and optimizer state to a checkpoint file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Loads a checkpoint and restores model and optimizer states."""
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded checkpoint from {path} (Epoch {epoch}, Loss {loss:.6f})")
            return epoch
        else:
            return 0

    def _create_sequences(self, data_ts: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates sliding window sequences (Input/Target) from the time series DataFrame."""
        T, N = data_ts.shape
        sequences_in: List[np.ndarray] = []
        sequences_target: List[np.ndarray] = []

        data_array = data_ts.values

        if T < self.input_len + self.output_len:
            print(f"Warning: Not enough time steps ({T}) to create sequences of length {self.input_len}+{self.output_len}.")
            return torch.empty((0, self.input_len, N), dtype=torch.float), torch.empty((0, self.output_len, N), dtype=torch.float)
            
        for i in range(T - self.input_len - self.output_len + 1):
            seq_in = data_array[i : i + self.input_len, :]
            seq_target = data_array[i + self.input_len : i + self.input_len + self.output_len, :]

            sequences_in.append(seq_in)
            sequences_target.append(seq_target)

        tensor_in = torch.tensor(sequences_in, dtype=torch.float)
        tensor_target = torch.tensor(sequences_target, dtype=torch.float)

        return tensor_in, tensor_target

    def _prepare_graph_data(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moves the static graph components to the correct device."""
        x_static = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        edge_attr = graph_data.edge_attr.to(self.device)
        return x_static, edge_index, edge_attr

    def _initialize_model(self, node_features: int, num_nodes: int, params: Dict[str, Any]) -> GNNForecastingModel:
        """Helper function to initialize a new model instance with specific parameters."""
        # The mandatory model initialization arguments must be passed separately
        model = GNNForecastingModel(
            node_features=node_features,
            num_nodes=num_nodes,
            input_len=self.input_len,
            output_len=self.output_len,
            # Use dictionary unpacking for optional hyperparameters
            hidden_dim=params.get('hidden_dim', 64),
            gnn_layers=params.get('gnn_layers', 2),
            lr=params.get('lr', 0.001)
        )
        model.to(self.device)
        return model

    def train_model(self, graph_data: Data, train_ts: pd.DataFrame, epochs: int = 50, checkpoint_dir: str = 'model_checkpoints', save_interval: int = 10, batch_size: int = 32):
        """Executes the training process and saves checkpoints using mini-batching."""
        
        print(f"\nStarting training for {epochs} epochs with batch size {batch_size} on device: {self.device} (Full GNN-LSTM)...")

        X_train, Y_train = self._create_sequences(train_ts)
        if X_train.shape[0] == 0: return

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        x_static, edge_index, edge_attr = self._prepare_graph_data(graph_data)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

            for batch_x, batch_y in epoch_iterator:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model.forward(x_static, edge_index, edge_attr, batch_x)

                target_y_permuted = batch_y.permute(0, 2, 1) # [B, H, N] -> [B, N, H]

                loss = self.criterion(predictions, target_y_permuted)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                epoch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)

            if epoch % save_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}], Training Loss (MSE): {avg_loss:.6f}")
                checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_e{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch, avg_loss)

        print("Training complete.")

    def evaluate_model(self, graph_data: Data, test_ts: pd.DataFrame, batch_size: int = 32) -> float:
        """Evaluates the model on the test time series data using mini-batching."""

        X_test, Y_test = self._create_sequences(test_ts)
        if X_test.shape[0] == 0: return 0.0

        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        x_static, edge_index, edge_attr = self._prepare_graph_data(graph_data)

        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                test_predictions = self.model.forward(x_static, edge_index, edge_attr, batch_x)
                target_y_permuted = batch_y.permute(0, 2, 1) # [B, H, N] -> [B, N, H]

                test_loss = self.criterion(test_predictions, target_y_permuted)
                total_loss += test_loss.item()

            avg_loss = total_loss / len(test_loader)

        print(f"Test Sequences created: {X_test.shape[0]} total, in {len(test_loader)} batches.")
        print(f"Test Loss (MSE): {avg_loss:.6f}")

        return avg_loss

    def grid_search(self, param_grid: Dict[str, List[Any]], graph_data: Data, train_ts: pd.DataFrame, test_ts: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Tuple[Dict[str, Any], float]:
        """
        Performs a grid search over specified hyperparameters to find the optimal combination.
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        # Generates all combinations of hyperparameters
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results: List[Dict[str, Any]] = []
        best_loss = float('inf')
        best_params: Dict[str, Any] = {}
        
        node_features = graph_data.x.shape[1]
        num_nodes = graph_data.num_nodes

        print(f"\n--- Starting Grid Search ({len(combinations)} total combinations) ---")
        
        for i, params in enumerate(combinations):
            print(f"\nConfiguration {i + 1}/{len(combinations)}: {params}")
            
            # 1. Initialize a new model and trainer instance for the current config
            current_model = self._initialize_model(node_features, num_nodes, params)
            current_trainer = ModelTrainer(current_model)
            
            # 2. Train the model (Note: We use the existing ModelTrainer train/eval methods)
            # Disable checkpointing during grid search to reduce I/O overhead
            current_trainer.train_model(graph_data, train_ts, epochs=epochs, batch_size=batch_size, checkpoint_dir='temp_grid_search', save_interval=epochs + 1)
            
            # 3. Evaluate the model on the test set
            test_loss = current_trainer.evaluate_model(graph_data, test_ts, batch_size=batch_size)
            
            # 4. Store result
            current_result = {'params': params, 'test_loss': test_loss}
            results.append(current_result)
            
            # 5. Check if this is the best result so far
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = params
                print(f"NEW BEST LOSS found: {best_loss:.6f}")

        print("\n--- Grid Search Complete ---")
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Best Parameters: {best_params}")
        
        return best_params, best_loss
