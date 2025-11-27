# inferencing.py

import torch
import pandas as pd
import numpy as np 
import os
import sys
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader 
import plotly.express as px
import datetime

# Importiere die notwendigen Klassen aus deinen Dateien
from data_loader_preprocessor import DataLoader, DataPreprocessor 
from model import GNNForecastingModel as ForecastingModel 

# --- 1. KONFIGURATION ---

FILE_PATHS = {
    'orders': 'orders.csv',
    'prior': 'order_products__prior.csv',
    'products': 'products.csv',
    'aisles': 'aisles.csv',
    'departments': 'departments.csv'
}
CHECKPOINT_PATH = 'model_checkpoints/model_checkpoint_e50.pt' 

# Modell-Hyperparameter (müssen mit dem Training übereinstimmen)
INPUT_SEQUENCE_LENGTH = 12 
OUTPUT_SEQUENCE_LENGTH = 1 
GNN_HIDDEN_DIM = 64
GNN_LAYERS = 2
EDGE_THRESHOLD = 10
TIME_SERIES_WEEKS = 150
SPLIT_RATIO = 0.85
SAMPLING_FRACTION = 0.1 
TEMPORAL_AGG_FACTOR = 4 # Wichtig: 4 für monatliche Aggregation (Zeitachse)

# NEU: Konfiguration für die iterative Prognose
FORECAST_STEPS = 3 # Anzahl der Monate, die prognostiziert werden sollen


def load_model_and_data(device):
    """Lädt das trainierte Modell, den Graphen und die Zeitreihendaten."""
    print("--- 1. LOADING DATA AND PREPROCESSOR ---")
    
    loader = DataLoader(FILE_PATHS, sampling_rate=SAMPLING_FRACTION)
    if not loader.load_data():
        sys.exit(1)
    data = loader.get_data()

    preprocessor = DataPreprocessor(data)
    node_features_X = preprocessor.create_node_features()
    num_nodes = len(preprocessor.product_nodes)
    node_features_F = node_features_X.shape[1]
    
    # Der vollständige Zeitreihen-Matrix wird benötigt
    train_ts, test_ts = preprocessor.create_demand_time_series(
        num_weeks=TIME_SERIES_WEEKS, 
        split_ratio=SPLIT_RATIO,
        temporal_agg_factor=TEMPORAL_AGG_FACTOR
    )
    full_ts = pd.concat([train_ts, test_ts])
    
    product_graph = preprocessor.build_graph_data(threshold=EDGE_THRESHOLD)
    
    # 3. Modell initialisieren und Checkpoint laden
    print("\n--- 2. LOADING MODEL CHECKPOINT ---")
    model = ForecastingModel( 
        num_nodes=num_nodes, 
        node_features=node_features_F, 
        input_len=INPUT_SEQUENCE_LENGTH, 
        output_len=OUTPUT_SEQUENCE_LENGTH,
        hidden_dim=GNN_HIDDEN_DIM,
        num_layers=GNN_LAYERS
    ).to(device)

    model.load_checkpoint(CHECKPOINT_PATH)
    model.eval()
    
    # Graph-Tensoren auf das Gerät verschieben
    product_graph.x = product_graph.x.to(device)
    product_graph.edge_index = product_graph.edge_index.to(device)
    product_graph.edge_attr = product_graph.edge_attr.to(device)
    
    return model, product_graph, full_ts, test_ts, preprocessor.products_df, preprocessor.product_id_to_index


def get_product_input_data(product_name, full_ts, products_df, input_len):
    """
    Sucht Produkt-ID und extrahiert die letzte historische Sequenz.
    Gibt die Produkt-ID, den Index und die historische Nachfrage-Sequenz zurück.
    """
    # 1. Produkt-ID suchen
    product_search = products_df[
        products_df['product_name'].str.contains(product_name, case=False, na=False)
    ]

    if product_search.empty:
        print(f"ERROR: Produktname '{product_name}' nicht in der Datenbank gefunden.")
        return None, None, None, None, None

    pid = product_search.iloc[0]['product_id']
    name_found = product_search.iloc[0]['product_name']
    
    # 2. Index im Zeitreihen-Matrix (N-Achse) finden
    if pid not in full_ts.columns:
        print(f"ERROR: Produkt-ID {pid} ('{name_found}') ist nicht in der Zeitreihen-Matrix.")
        print("Dies geschieht oft, weil das Produkt in den gesampleten Trainingsdaten nie gekauft wurde.")
        return None, None, None, None, None

    product_index = full_ts.columns.get_loc(pid)

    # 3. Letzte Input-Sequenz extrahieren
    last_idx = full_ts.shape[0] 
    
    if last_idx < input_len:
        print(f"ERROR: Nicht genug Monate ({last_idx}) für Input-Länge {input_len}.")
        return None, None, None, None, None

    # Sequenz: [Input_Len, Num_Nodes]
    historic_sequence_df = full_ts.iloc[last_idx - input_len : last_idx, :] 

    # Wandelt den DataFrame in den initialen Tensor [B=1, L, N] um
    sequence_tensor = torch.tensor(
        historic_sequence_df.values, 
        dtype=torch.float
    ).unsqueeze(0) 

    return pid, name_found, product_index, sequence_tensor, historic_sequence_df


def plot_demand_forecast(historic_df, forecast_results, product_id, product_name, forecast_steps):
    """Erstellt den Plotly-Plot der historischen und prognostizierten Nachfragekurve."""
    
    # 1. Vorbereiten der historischen Daten
    demand_history = historic_df[product_id] 
    
    # KORRIGIERTE ZEITACHSEN-BERECHNUNG: Index als Offset zum Startdatum verwenden
    START_DATE = pd.to_datetime('2014-01-01')
    
    # Berechnung des Datums für die historischen Punkte
    time_offset_days = (demand_history.index.to_series() - 1) * TEMPORAL_AGG_FACTOR * 7
    dates = START_DATE + pd.to_timedelta(time_offset_days, unit='D')
    
    # Erstelle einen DataFrame für Plotly
    plot_df = pd.DataFrame({
        'Demand': demand_history.values,
        'Date': dates,
        'Type': 'Historical Demand'
    })
    
    # 2. Vorbereiten der prognostizierten Daten (Kurve)
    last_historical_date = plot_df['Date'].iloc[-1]
    
    forecast_dates = [
        last_historical_date + pd.Timedelta(days=(step + 1) * TEMPORAL_AGG_FACTOR * 7)
        for step in range(forecast_steps)
    ]

    forecast_df = pd.DataFrame({
        'Demand': forecast_results,
        'Date': forecast_dates,
        'Type': 'Predicted Demand'
    })
    
    # Füge den letzten historischen Punkt zum Prognose-Plot hinzu, um die Linie zu verbinden
    connector_df = pd.DataFrame({
        'Demand': [plot_df['Demand'].iloc[-1]],
        'Date': [last_historical_date],
        'Type': ['Predicted Demand']
    })
    
    # 3. Konsolidierung und Plot
    final_df = pd.concat([plot_df, connector_df, forecast_df]).sort_values(by='Date')
    
    fig = px.line(
        final_df, 
        x='Date', 
        y='Demand', 
        color='Type', 
        title=f'Nachfrageprognose ({forecast_steps} Monate): {product_name}',
        markers=True,
        line_dash='Type'
    )
    
    # Hebt den Prognosebereich hervor
    fig.add_scatter(
        x=forecast_df['Date'], y=forecast_df['Demand'], 
        mode='markers', marker=dict(size=10, color='red'), 
        name='Forecast Points'
    )
    
    fig.show()
    print(f"Plot für '{product_name}' wurde im Browser geöffnet.")


def run_inference(model, graph_data, product_name, full_ts, products_df, device, forecast_steps=FORECAST_STEPS):
    """Führt die Vorhersage iterativ für mehrere Schritte aus."""

    pid, name_found, p_index, input_tensor, historic_sequence_df = get_product_input_data(
        product_name, full_ts, products_df, model.input_len
    )
    
    if input_tensor is None:
        return

    print(f"\n--- INFERENCE START (Iterative Forecast) ---")
    print(f"Product: {name_found} (ID: {pid})")
    print(f"Using {model.input_len} months history to predict the next {forecast_steps} months.")

    current_input_tensor = input_tensor.to(device)
    forecast_results = []
    
    # Wir benötigen die konstanten Graphen-Features (bereits auf dem Gerät)
    x_static = graph_data.x
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr


    with torch.no_grad():
        for step in range(forecast_steps):
            
            # 1. Vorhersage des nächsten Schritts (t+1)
            predictions = model(x_static, edge_index, edge_attr, current_input_tensor)
            
            # 2. Ergebnis extrahieren und speichern
            predicted_demand_value = predictions[0, p_index, 0].item()
            predicted_demand_rounded = round(max(0, predicted_demand_value))
            
            forecast_results.append(predicted_demand_rounded)

            # 3. Den Output als Input für den nächsten Zeitschritt vorbereiten (Rekursiv)
            
            # Konvertiere den gesamten Vorhersage-Tensor [1, N, 1] in [1, 1, N]
            new_time_step_data = predictions.permute(0, 2, 1).cpu() 
            
            # Füge den neuen Schritt an und entferne den ältesten Schritt
            # current_input_tensor: [B=1, L, N]
            current_input_tensor = torch.cat(
                (current_input_tensor[:, 1:, :].cpu(), new_time_step_data), 
                dim=1
            ).to(device)


    # Ausgabe und Plotten
    print(f"\n[RESULT] Predicted demand for the next {forecast_steps} months:")
    print(f" -> {name_found}: {forecast_results} units")
    print("----------------------")
    
    # Plotten der Kurve
    plot_demand_forecast(
        historic_sequence_df, 
        forecast_results, 
        pid, 
        name_found,
        forecast_steps
    )


if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing on device: {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"FATAL ERROR: Checkpoint file not found at {CHECKPOINT_PATH}")
        print("Bitte führen Sie zuerst main.py aus, um das Modell zu trainieren und zu speichern.")
        sys.exit(1)

    # Lade Modell und Daten
    MODEL, GRAPH_DATA, FULL_TS, TEST_TS, PRODUCTS_DF, PRODUCT_ID_TO_INDEX = load_model_and_data(DEVICE)
    
    # Start der Inferenz-Schleife
    while True:
        user_input = input(f"\nGebe den Produktnamen ein (oder 'exit') und drücke Enter (z.B. {FORECAST_STEPS} Monate): ")
        if user_input.lower() == 'exit':
            break
        
        run_inference(MODEL, GRAPH_DATA, user_input, FULL_TS, PRODUCTS_DF, DEVICE)