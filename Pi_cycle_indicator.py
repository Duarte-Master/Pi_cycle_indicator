import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt          # <-- REMOVED
# import matplotlib.dates as mdates        # <-- REMOVED
import plotly.graph_objects as go          # <-- NEW: Import Plotly Graph Objects
from plotly.subplots import make_subplots  # <-- NEW: Import for Subplots

# ... (sma function remains unchanged) ...

# ... (plot_timeseries_data function header remains unchanged) ...
def plot_timeseries_data(filepath, start_date=None, end_date=None):
    """
    Loads data, calculates indicators, and plots the result using Plotly.
    """
    st.write(f"Loading data from: {filepath}...")

    # ... (Data Loading, Preprocessing, and Indicator Calculation remain unchanged) ...
    # This section calculates: price_data, ma_111, ma_350_double, zScore_valid
    # ...

    # --- NEW: Plotting Setup using Plotly ---
    
    # 1. Create a figure with two subplots: Price (row 1) and Z-Score (row 2)
    # The specs set the relative heights and ensure the second subplot shares the x-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[3, 1],
        subplot_titles=("BTC/USD Price Time Series (Log Scale) and Pi-Cycle Lines", "Pi-Cycle Z-Score")
    )
    
    # --- Price Chart (Row 1) ---

    # 2. Add Price Data (Primary Y-axis of the top plot)
    fig.add_trace(
        go.Scatter(
            x=price_data.index, y=price_data.values,
            mode='lines', name='Price',
            line=dict(color='white', width=1),
            hovertemplate='Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>' # Custom hover format
        ),
        row=1, col=1
    )

    # 3. Add Pi-Cycle Moving Averages
    
    # Yellow Line (111 SMA)
    fig.add_trace(
        go.Scatter(
            x=ma_111.index, y=ma_111.values,
            mode='lines', name='111 SMA',
            line=dict(color='yellow', width=2),
            hovertemplate='Date: %{x}<br>111 SMA: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Green Line (2 * 350 SMA)
    fig.add_trace(
        go.Scatter(
            x=ma_350_double.index, y=ma_350_double.values,
            mode='lines', name='2 * 350 SMA',
            line=dict(color='green', width=2),
            hovertemplate='Date: %{x}<br>2*350 SMA: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Set Y-axis to Log Scale for the Price Chart
    fig.update_yaxes(type="log", row=1, col=1, title_text="Price (USD) [Log Scale]", showgrid=True)


    # --- Z-Score Subplot (Row 2) ---

    # 4. Add Z-Score Line (Primary Y-axis of the bottom plot)
    fig.add_trace(
        go.Scatter(
            x=zScore_valid.index, y=zScore_valid.values,
            mode='lines', name='Pi-Cycle Z-Score',
            line=dict(color='red', width=2),
            hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # 5. Add Z-Score Horizontal Lines
    
    # 0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    
    # Key Buy/Sell Lines (Plotly does not support simple labels for hlines, so we omit them)
    fig.add_hline(y=3, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    fig.add_hline(y=-3, line_dash="dash", line_color="green", line_width=2, row=2, col=1)
    
    # Set Z-Score Y-axis limits
    fig.update_yaxes(
        title_text="Z-Score", 
        range=[-4.5, 4.5], # Fixed range for better visualization
        row=2, col=1, 
        showgrid=True
    )
    
    # --- Global Layout Customization (Sets size, background, and Range Slider) ---
    
    fig.update_layout(
        title_text='Bitcoin Pi-Cycle Indicator', # Main title
        height=800, # Increased height for better visibility of two charts
        # width=1200, # We will use Streamlit's container width instead of fixed width
        template="plotly_dark", # Use a dark theme template
        hovermode="x unified", # Shows all data for a given X-value on hover
        
        # Rangeslider is applied to the shared X-axis (the bottom one)
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.08,
            ),
            type="date",
            title_text="Date"
        ),
        
        # Apply the selected date range from the Streamlit slider
        xaxis2=dict(
            range=[start_date, end_date] if start_date and end_date else None,
        )
    )

    # 6. Display the interactive plot in Streamlit
    # use_container_width=True ensures the plot scales to the full Streamlit column size.
    st.plotly_chart(fig, use_container_width=True)


# ... (__main__ block remains unchanged) ...
if __name__ == '__main__':
    st.set_page_config(layout="wide") 
    st.title("Bitcoin Pi-Cycle Indicator Dashboard")
    
    file_to_plot = 'Bitcoin Historical Data_test.csv'
    
    # 1. Determine the Min/Max dates
    try:
        temp_df = pd.read_csv(file_to_plot, thousands=',')
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%m/%d/%Y', errors='coerce')
        min_date_available = temp_df['Date'].min().date()
        max_date_available = temp_df['Date'].max().date()
    except Exception:
        min_date_available = pd.to_datetime('2010-01-01').date()
        max_date_available = pd.to_datetime('today').date()
        
    st.sidebar.header("Date Range Selection")
    
    # 2. Create the Streamlit Date Slider
    date_range = st.sidebar.slider(
        "Select Time Period:",
        min_value=min_date_available,
        max_value=max_date_available,
        value=(min_date_available, max_date_available), # Default to full range
        format="YYYY-MM-DD"
    )
    
    selected_start_date = date_range[0] 
    selected_end_date = date_range[1]   

    # CRITICAL FIX: Convert Python date objects to Pandas Timestamps
    start_date_filter = pd.to_datetime(selected_start_date)
    end_date_filter = pd.to_datetime(selected_end_date)
    
    # 3. Call the plotting function with the CORRECTLY TYPED dates
    plot_timeseries_data(file_to_plot, start_date_filter, end_date_filter)
