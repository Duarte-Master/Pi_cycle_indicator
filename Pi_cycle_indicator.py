import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt          # <-- MUST REMOVE
# import matplotlib.dates as mdates        # <-- MUST REMOVE
import plotly.graph_objects as go          # <-- KEEP
from plotly.subplots import make_subplots  # <-- KEEP

# --- Custom SMA Function ---
def sma(series, period):
    """
    Calculates the Simple Moving Average (SMA) for a given series.
    Uses min_periods=period to return a value only when the full window is available.
    """
    # Use min_periods=period for standard indicator calculation.
    return series.rolling(window=period, min_periods=period).mean()
# --- End Custom SMA Function ---

# MODIFIED: Function accepts optional start_date and end_date (now expected to be Timestamps)
def plot_timeseries_data(filepath, start_date=None, end_date=None):
    """
    Loads data, calculates indicators, and plots the result using Plotly.
    """
    #st.write(f"Created by Gonçalo Duarte\n Loading data from: {filepath}...")
    st.write(f"Created by Gonçalo Duarte")
    st.write(f"Loading data from: {filepath}...")

    try:
        df = pd.read_csv(filepath, thousands=',')
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please check the path and filename.")
        return
    except Exception as e:
        st.error(f"An error occurred while reading the CSV: {e}")
        return

    # ... (Data Preprocessing and Indicator Calculation remain unchanged) ...
    date_column_name = 'Date'
    price_column_name = 'Price'

    if date_column_name not in df.columns or price_column_name not in df.columns:
        missing = [col for col in [date_column_name, price_column_name] if col not in df.columns]
        st.error(f"Error: Missing required columns: {', '.join(missing)}. Available columns: {df.columns.tolist()}")
        return

    try:
        df[date_column_name] = pd.to_datetime(df[date_column_name], format='%m/%d/%Y')
    except ValueError as e:
        st.error(f"Error parsing dates: {e}. Check if the date format is strictly 'MM/DD/YYYY'.")
        return

    df.set_index(date_column_name, inplace=True)
    price_data = df[price_column_name]
    price_data = pd.to_numeric(price_data, errors='coerce')
    price_data.dropna(inplace=True)

    # Reindex to fill calendar gaps
    if not price_data.empty:
        start_date_full = price_data.index.min()
        end_date_full = price_data.index.max()
        full_date_range = pd.date_range(start=start_date_full, end=end_date_full, freq='D')
        price_data = price_data.reindex(full_date_range).ffill()
    price_data.dropna(inplace=True)

    # Technical Analysis Calculation
    ma_111 = sma(price_data, period=111)
    ma_350_double = 2 * sma(price_data, period=350)
    Pi_cycle_ratio = 2 - (ma_350_double / ma_111).replace([np.inf, -np.inf], np.nan)

    thresholdTop = 1.05
    thresholdBottom = -1.10
    mean = (thresholdTop + thresholdBottom) / 2
    bands_range = thresholdTop - thresholdBottom
    stdDev = bands_range / 6 if bands_range != 0 else 0
    
    if stdDev != 0:
        zScore = (Pi_cycle_ratio - mean) / stdDev
    else:
        zScore = pd.Series(0, index=Pi_cycle_ratio.index)
    zScore_valid = zScore.dropna()
    
    # --- Plotting Setup using Plotly ---
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.18,
        row_heights=[3, 1],
        subplot_titles=("BTC/USD Price Time Series (Log Scale) and Pi-Cycle Lines", "Pi-Cycle Z-Score")
    )
    
    # --- Price Chart (Row 1) ---

    # 2. Add Price Data (This is the line that had the potential NameError)
    fig.add_trace(
        go.Scatter(
            x=price_data.index, y=price_data.values, # <-- CORRECTED: Ensure we use price_data.values
            mode='lines', name='Price',
            line=dict(color='white', width=1),
            hovertemplate='Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>' 
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

    fig.update_yaxes(type="log", row=1, col=1, title_text="Price (USD) [Log Scale]", showgrid=True)


    # --- Z-Score Subplot (Row 2) ---

    # 4. Add Z-Score Line
    fig.add_trace(
        go.Scatter(
            x=zScore_valid.index, y=zScore_valid.values,
            mode='lines', name='Pi-Cycle Z-Score',
            line=dict(color='red', width=2),
            hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

# 5. Add Z-Score Horizontal Lines with Gradient Colors
    
    # Neutral/Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    # POSITIVE (RED) LINES - Fading to transparent (lower Alpha) closer to zero
    
    # +1 Line (Very Transparent Red) -> Alpha 0.2
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(255, 0, 0, 0.2)", line_width=1, row=2, col=1)
    
    # +2 Line (Moderately Transparent Red) -> Alpha 0.4
    fig.add_hline(y=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.4)", line_width=1, row=2, col=1)
    
    # +3 Line (Solid Red) -> Alpha 1.0 (or just 'red')
    fig.add_hline(y=3, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    
    # NEGATIVE (GREEN) LINES - Fading to transparent (lower Alpha) closer to zero
    
    # -1 Line (Very Transparent Green) -> Alpha 0.2
    fig.add_hline(y=-1, line_dash="dash", line_color="rgba(0, 255, 0, 0.2)", line_width=1, row=2, col=1)
    
    # -2 Line (Moderately Transparent Green) -> Alpha 0.4
    fig.add_hline(y=-2, line_dash="dash", line_color="rgba(0, 255, 0, 0.4)", line_width=1, row=2, col=1)
    
    # -3 Line (Solid Green) -> Alpha 1.0 (or just 'green')
    fig.add_hline(y=-3, line_dash="dash", line_color="green", line_width=2, row=2, col=1)
    
    fig.update_yaxes(
        title_text="Z-Score", 
        range=[-4.5, 4.5], 
        row=2, col=1, 
        showgrid=True
    )
    
# --- Global Layout Customization ---
    fig.update_layout(
        title_text='Bitcoin Pi-Cycle Indicator', 
        height=820,
        template="plotly_dark", 
        hovermode="x unified", 
        
        # CRITICAL CHANGE: Significantly increase bottom margin to push the subplot up.
        margin=dict(b=150), 
        
        # Rangeslider is applied to the shared X-axis (the bottom one)
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.04, # Keep thickness minimal
            ),
            type="date",
            title_text="Date"
        ),
        
        # We no longer need this x-axis filter since the Plotly slider controls the view.
        xaxis2=dict(
            range=None,
        )
    )

    # 6. Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

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
        
    #st.sidebar.header("Date Range Selection")
    
    # 2. Create the Streamlit Date Slider
    #date_range = st.sidebar.slider(
    #    "Select Time Period:",
    #    min_value=min_date_available,
    #    max_value=max_date_available,
    #    value=(min_date_available, max_date_available), # Default to full range
    #    format="YYYY-MM-DD"
    #)
    
    selected_start_date = min_date_available 
    selected_end_date = max_date_available

    # CRITICAL FIX: Convert Python date objects to Pandas Timestamps
    start_date_filter = pd.to_datetime(selected_start_date)
    end_date_filter = pd.to_datetime(selected_end_date)
    
    # 3. Call the plotting function with the CORRECTLY TYPED dates
    plot_timeseries_data(file_to_plot, start_date_filter, end_date_filter)











