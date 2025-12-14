import torch
import torch.nn as nn
from torch_geometric.nn import GATConv 
import numpy as np
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Tuple, Dict, Any, List
import itertools
import matplotlib.pyplot as plt

def _get_batch_size_and_num_nodes(tensor: torch.Tensor) -> Tuple[int, int, int]:
    """Extracts batch size (B), sequence length (L), and number of nodes (N) from the demand sequence tensor."""
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, L, N], got {tensor.dim()}D tensor.")
    B, L, N = tensor.shape
    return B, L, N

class GNNForecastingModel(nn.Module):
    """
    The complete GNN-LSTM hybrid model for time series forecasting on graphs.
    """
    def __init__(self, node_features: int, num_nodes: int, input_len: int, output_len: int, hidden_dim: int = 64, gnn_layers: int = 2, lr: float = 0.001, dropout_rate: float = 0.3):
        super(GNNForecastingModel, self).__init__()

        # Hyperparameters
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gnn_layers = gnn_layers
        self.dropout_rate = dropout_rate

        # GNN (GATConv) layers
        self.gcn_layers = nn.ModuleList()
        input_dim = node_features
        for i in range(self.gnn_layers):
            if i < self.gnn_layers - 1:
                # Intermediate layers: Multi-Head GAT with concatenation
                self.gcn_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout_rate))
                input_dim = hidden_dim * 4
            else:
                # Last layer: Single-Head GAT, to project to hidden_dim for the LSTM
                self.gcn_layers.append(GATConv(input_dim, hidden_dim, heads=1, concat=False, dropout=dropout_rate))
                input_dim = hidden_dim

        # LSTM 
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim,
            num_layers=1, 
            batch_first=True
        )

        # The final Output Layer
        COMBINED_DIM = hidden_dim * 2
        self.output_layer = nn.Linear(COMBINED_DIM, output_len)
        
        # Poisson NLL Loss for count data
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction='mean') 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_device(self) -> torch.device:
        """Determines the current device of the model."""
        return next(self.parameters()).device

    def forward(self, x_static: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, demand_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete GNN-Forecasting model.
        """
        B, L, N = _get_batch_size_and_num_nodes(demand_seq)

        # Static GNN Processing 
        h_static = x_static
        for gnn_layer in self.gcn_layers:
            # GATConv does not take edge_attr/edge_weight
            h_static = torch.relu(gnn_layer(h_static, edge_index)) 

        # Sequence Preparation for LSTM
        # Shape: [B, L, N] -> [B, N, L] -> [B*N, L] -> [B*N, L, 1]
        demand_seq_final = demand_seq.permute(0, 2, 1).reshape(-1, L).unsqueeze(-1)

        # Initial hidden state (h0) and cell state (c0) are initialized with GNN features
        h0_expanded = h_static.unsqueeze(0).expand(B, N, self.hidden_dim)
        h0 = h0_expanded.reshape(B * N, self.hidden_dim).unsqueeze(0).contiguous()
        c0 = h0.clone() # c0 is initialized with GNN features as well

        # Pass through LSTM
        _, (hn, _) = self.lstm(demand_seq_final, (h0, c0))
        final_lstm_output = hn[-1] 

        # Feature Concatenation and Prediction
        # Shape h_static_repeated: [B*N, Hidden_Dim]
        h_static_repeated = h_static.unsqueeze(0).expand(B, N, self.hidden_dim).reshape(B * N, self.hidden_dim)
        combined_features = torch.cat((final_lstm_output, h_static_repeated), dim=1)

        # Final Linear Layer for multi-step prediction
        predictions_flat = self.output_layer(combined_features) 
        
        # ReLU for non-negative predictions (since we use Poisson NLL)
        predictions_flat = torch.relu(predictions_flat) 
        
        # Shape: [B*N, H] -> [B, N, H]
        predictions = predictions_flat.view(B, N, -1)

        return predictions


class ModelTrainer:
    """
    Handles training, evaluation, checkpointing, and data preparation.
    """
    def __init__(self, model: GNNForecastingModel, preprocessor=None):
        self.model = model
        self.input_len = model.input_len
        self.output_len = model.output_len
        self.criterion = model.criterion
        self.optimizer = model.optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.preprocessor = preprocessor 
        self.training_loss_history: List[float] = [] 

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
        """Creates sliding window sequences from the time series DataFrame."""
        T, N = data_ts.shape
        sequences_in: List[np.ndarray] = []
        sequences_target: List[np.ndarray] = []

        data_array = data_ts.values

        if T < self.input_len + self.output_len:
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
        model = GNNForecastingModel(
            node_features=node_features,
            num_nodes=num_nodes,
            input_len=self.input_len,
            output_len=self.output_len,
            hidden_dim=params.get('hidden_dim', 64),
            gnn_layers=params.get('gnn_layers', 2),
            lr=params.get('lr', 0.001),
            dropout_rate=params.get('dropout_rate', 0.3)
        )
        model.to(self.device)
        return model
        
    def _plot_training_loss(self, epochs: int, plot_dir: str = 'plots'):
        """
        Creates and saves a plot of the training loss history.
        """
        if not self.training_loss_history:
            return

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'training_loss_evolution.png')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), self.training_loss_history, marker='o', linestyle='-', color='blue')
        plt.title('Training Loss (Poisson NLL) Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Training loss plot saved to {plot_path}")

    def train_model(self, graph_data: Data, train_ts: pd.DataFrame, epochs: int = 50, checkpoint_dir: str = 'model_checkpoints', save_interval: int = 10, batch_size: int = 32):
        """
        Executes the training process using raw count data.
        """
        if graph_data.edge_index.shape[1] == 0:
            print("WARNING: Graph has no edges. Training skipped.")
            return

        print(f"\nStarting training for {epochs} epochs with batch size {batch_size} on device: {self.device} (Full GNN-LSTM with raw counts)...")

        X_train, Y_train = self._create_sequences(train_ts)
        if X_train.shape[0] == 0: return

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        x_static, edge_index, edge_attr = self._prepare_graph_data(graph_data)
        
        self.training_loss_history = [] 

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

            for batch_x, batch_y in epoch_iterator:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model.forward(x_static, edge_index, edge_attr, batch_x)
                target_y_permuted = batch_y.permute(0, 2, 1) # Targets permuten: [B, L_out, N] -> [B, N, H] 
                loss = self.criterion(predictions, target_y_permuted)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                epoch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.training_loss_history.append(avg_loss) 

            if epoch % save_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}], Training Loss (Poisson NLL): {avg_loss:.6f}")
                checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_e{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch, avg_loss)
        
        self._plot_training_loss(epochs)

        print("Training complete.")

    def evaluate_model(self, graph_data: Data, test_ts: pd.DataFrame, batch_size: int = 32) -> float:
        """
        Evaluates the model on the raw count test time series data.
        """

        X_test, Y_test = self._create_sequences(test_ts)
        if X_test.shape[0] == 0: 
            print("Not enough test data to create sequences. Evaluation skipped.")
            return 0.0

        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        x_static, edge_index, edge_attr = self._prepare_graph_data(graph_data)
        
        # Use MSE with reduction='sum' for manual averaging
        mean_squared_error_criterion = nn.MSELoss(reduction='sum') 
        total_final_mse = 0.0
        total_elements = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 1. Prediction (raw count)
                predictions = self.model.forward(x_static, edge_index, edge_attr, batch_x)

                # 2. Targets: [B, L_out, N] -> [B, N, H] 
                target_reshaped = batch_y.permute(0, 2, 1) 
                
                # 3. Targets and Predictions are already unscaled (raw counts)
                unscaled_predictions_tensor = predictions
                unscaled_target_tensor = target_reshaped
                
                # 4. Ensure non-negativity for RMSE (Predictions already have ReLU)
                unscaled_target_tensor = torch.relu(unscaled_target_tensor)
                
                # 5. Calculation of the Sum of Squared Errors (SSE)
                test_loss = mean_squared_error_criterion(unscaled_predictions_tensor, unscaled_target_tensor)
                total_final_mse += test_loss.item()
                total_elements += unscaled_predictions_tensor.numel()

        # Calculation of the AVG final MSE and RMSE
        average_final_mse = total_final_mse / total_elements
        # Convert to float/Tensor to calculate and return the pure Python float value
        average_final_rmse = torch.sqrt(torch.tensor(average_final_mse, dtype=torch.float)).item()
        
        print(f"Test Sequences created: {X_test.shape[0]} total, in {len(test_loader)} batches.")
        print(f"Test Loss (Final RMSE on unscaled data): {average_final_rmse:.6f}")

        return average_final_rmse

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

        print(f"\nStarting Grid Search ({len(combinations)} total combinations)")
        
        for i, params in tqdm(enumerate(combinations), total=len(combinations), desc="Grid Search Configurations"):
            print(f"\nConfiguration {i + 1}/{len(combinations)}: {params}")
            
            # Initialize a new model and trainer instance for the current config
            current_model = self._initialize_model(node_features, num_nodes, params)
            current_trainer = ModelTrainer(current_model) 
            
            # Train the model 
            current_trainer.train_model(graph_data, train_ts, epochs=epochs, batch_size=batch_size, checkpoint_dir='temp_grid_search', save_interval=epochs + 1)
            
            # Evaluate the model on the test set 
            test_loss = current_trainer.evaluate_model(graph_data, test_ts, batch_size=batch_size)
            
            # Store result
            current_result = {'params': params, 'test_loss': test_loss}
            results.append(current_result)
            
            # Check if this is the best result so far?
            if test_loss < best_loss and test_loss > 0.0:
                best_loss = test_loss
                best_params = params
                print(f"NEW BEST RMSE found: {best_loss:.6f}")

        print("\n--- Grid Search Complete ---")
        print(f"Best Loss (RMSE): {best_loss:.6f}")
        print(f"Best Parameters: {best_params}")
        
        return best_params, best_loss