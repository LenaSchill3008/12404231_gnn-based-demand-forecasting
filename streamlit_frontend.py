import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any

API_URL = "http://localhost:8000"
FORECAST_OPTIONS = [1, 3, 6]

def get_product_list() -> Dict[int, str]:
    """Fetches the list of available products (ID and name) from the API."""
    try:
        response = requests.get(f"{API_URL}/products")
        response.raise_for_status()
        
        # Converts {str_id: name} to {int_id: name}
        product_map = {int(k): v for k, v in response.json().items()}
        
        # Sorts products by name for better readability
        return dict(sorted(product_map.items(), key=lambda item: item[1]))
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving product list: {e}. Is the FastAPI server running?")
        return {}

def get_prediction(product_id: int, forecast_months: int) -> Dict[str, Any] | None:
    """Fetches the demand forecast for a product from the API."""
    try:
        response = requests.post(f"{API_URL}/predict/{product_id}", params={"forecast_months": forecast_months})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('detail', 'Unknown error')
        st.error(f"Error during prediction (HTTP {e.response.status_code}): {error_detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error to the API: {e}")
        return None

def plot_demand(historical_demand: List[float], prediction: List[float], input_len: int, slots_predicted: int):
    """
    Creates the plot of historical and forecasted demand.
    The X-axis is adjusted for better readability based on the assumed temporal aggregation factor (4).
    """
    historical_length = len(historical_demand)
    
    historical_df = pd.DataFrame({
        'Slot': range(historical_length),
        'Demand': historical_demand,
        'Type': 'Historical (Test Set)'
    })
    
    start_slot = historical_length
    prediction_df = pd.DataFrame({
        'Slot': range(start_slot, start_slot + len(prediction)),
        'Demand': prediction,
        'Type': 'Forecast'
    })
    
    total_slots = historical_length + len(prediction)
    
    AGG_FACTOR = 4 

    tick_slots = list(range(0, total_slots + AGG_FACTOR, AGG_FACTOR))
    
    tick_labels = [f"Slot {t}" if t != 0 else "Slot 0" for t in tick_slots]
    
    # Plotly Visualization
    fig = go.Figure()

    # Historical Line
    fig.add_trace(go.Scatter(
        x=historical_df['Slot'],
        y=historical_df['Demand'],
        mode='lines',
        name='Historical Demand',
        line=dict(color='blue')
    ))
    
    # Forecast Line
    fig.add_trace(go.Scatter(
        x=prediction_df['Slot'],
        y=prediction_df['Demand'],
        mode='lines+markers',
        name=f'Forecast ({slots_predicted} Slots)',
        line=dict(color='red', dash='dot')
    ))

    # Highlight Model Input Window
    input_start = start_slot - input_len
    if input_start >= 0:
        fig.add_vrect(
            x0=input_start, x1=start_slot, 
            fillcolor="lightblue", opacity=0.3, 
            layer="below", line_width=0,
            annotation_text=f"Model Input ({input_len} slots)",
            annotation_position="top left"
        )

    fig.update_layout(
        title='Product Demand: History and Forecast',
        
        # X-AXIS DEFINITION
        xaxis=dict(
            title='Time Slot (Aggregated Orders, 4 Slots ≈ 1 Month)',
            tickmode='array',
            tickvals=tick_slots, 
            ticktext=tick_labels,
            showgrid=True
        ),
        
        yaxis_title='Demand (Raw Count)',
        hovermode="x unified",
        legend=dict(y=1.05, x=0.05, orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit Application 
st.set_page_config(layout="wide", page_title="GNN Demand Forecast Demo")

st.title("GNN-LSTM Product Demand Forecasting")
st.markdown("Demonstration of the GNN hybrid model's forecasting capability.")

# Check API availability
api_status = requests.get(f"{API_URL}/health")
if api_status.status_code != 200:
    st.error(f"API connection error: Please ensure the FastAPI server is running at {API_URL} and the model/data are loaded.")
    st.code(f"API Status: {api_status.json().get('message', 'Unknown')}")
    st.stop()


product_map = get_product_list()
product_options = list(product_map.keys())

if product_options:
    
    col1, col2 = st.columns([1, 1])
    
    display_options = [f"{name} (ID: {pid})" for pid, name in product_map.items()]
    display_to_id = {f"{name} (ID: {pid})": pid for pid, name in product_map.items()}

    with col1:
        selected_display = st.selectbox(
            "Select a Product",
            options=display_options,
            index=0,
            help="Select a product for which demand is to be forecasted."
        )
        selected_product_id = display_to_id.get(selected_display)


    with col2:
        forecast_months = st.selectbox(
            "Forecast Length (Months/Slots)",
            options=FORECAST_OPTIONS,
            index=0,
            help="Defines the number of future months to be forecasted."
        )

    if st.button(f"Start Forecast for {forecast_months} Month(s)"):
        if selected_product_id is not None:
            with st.spinner(f"Calculating forecast for product {selected_product_id}..."):
                
                prediction_data = get_prediction(selected_product_id, forecast_months)
                
                if prediction_data:
                    
                    st.subheader(f"Results for {selected_display}")
                    
                    plot_demand(
                        historical_demand=prediction_data['historical_demand'],
                        prediction=prediction_data['prediction'],
                        input_len=prediction_data['input_sequence_length'],
                        slots_predicted=prediction_data['output_slots_predicted']
                    )
                    
                    total_forecast = sum(prediction_data['prediction'])
                    
                    st.metric(
                        label=f"Estimated total demand for the next {forecast_months} months ({prediction_data['output_slots_predicted']} slots)", 
                        value=f"{total_forecast:,.0f} units"
                    )

else:
    st.warning("No products available. Please check the API and data processing.")