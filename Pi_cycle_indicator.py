import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Removed Matplotlib interactive imports (RangeSlider, etc.)

# --- Custom SMA Function ---
# ... (sma function remains unchanged) ...
def sma(series, period):
    # ... (function body remains unchanged) ...
    return series.rolling(window=period, min_periods=period).mean()
# --- End Custom SMA Function ---


# MODIFIED: Function accepts optional start_date and end_date (now expected to be Timestamps)
def plot_timeseries_data(filepath, start_date=None, end_date=None):
    """
    Loads time series data, calculates indicators, and plots the result,
    applying date limits if provided by Streamlit sliders.
    """
    st.write(f"Loading data from: {filepath}...")

    try:
        df = pd.read_csv(filepath, thousands=',')
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please check the path.")
        return
    except Exception as e:
        st.error(f"An error occurred while reading the CSV: {e}")
        return

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

    # --- Technical Analysis Calculation (Using Fixed Logic) ---
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
    
    # --- Plotting Setup ---

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax = ax1

    # Apply x-limits based on Streamlit slider input
    if start_date is not None and end_date is not None: # Check for not None
        ax1.set_xlim(start_date, end_date)
        
        # Filtering for dynamic Y-axis scaling (Uses the passed Timestamps)
        price_range = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
        ma_111_range = ma_111[(ma_111.index >= start_date) & (ma_111.index <= end_date)]
        ma_350_double_range = ma_350_double[(ma_350_double.index >= start_date) & (ma_350_double.index <= end_date)]
        all_y_values = pd.concat([price_range, ma_111_range, ma_350_double_range])
        all_y_values.dropna(inplace=True)

        if not all_y_values.empty:
            y_min = all_y_values.min() * 0.90
            y_max = all_y_values.max() * 1.10
            if y_min <= 0:
                y_min = all_y_values[all_y_values > 0].min() * 0.90 if not all_y_values[all_y_values > 0].empty else 1
            ax1.set_ylim(y_min, y_max)
    # End X/Y limit application

    # Initial plot of the entire time series
    ax.plot(price_data.index, price_data.values, color='black', linewidth=1, label=price_column_name)

    if not ma_111.empty:
        latest_ma_111 = ma_111.iloc[-1]
    else:
        latest_ma_111 = np.nan

    if not ma_350_double.empty:
        latest_ma_350_double = ma_350_double.iloc[-1]
    else:
        latest_ma_350_double = np.nan

    # Plot the Moving Averages on the PRICE chart (ax1)
    ax.plot(ma_111.index, ma_111.values, color='yellow', linewidth=2, label=f'111 SMA: {latest_ma_111:.0f}')
    ax.plot(ma_350_double.index, ma_350_double.values, color='green', linewidth=2, label=f'2 * 350 SMA: {latest_ma_350_double:.0f}')

    # Configure main plot aesthetics (ax1)
    ax.set_title(f'BTC/USD Price Time Series (Log Scale) and Pi-Cycle Z-Score') 
    ax.set_ylabel('Price (USD) [Log Scale]')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    date_formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)
    
    # --- Z-Score Subplot (ax2) ---
    if not zScore_valid.empty:
        latest_z_score = zScore_valid.iloc[-1]
    else:
        latest_z_score = np.nan

    ax2.plot(
        zScore_valid.index,
        zScore_valid.values,
        color='black',
        linewidth=2,
        label=f'Pi-Cycle Z-Score (Last): {latest_z_score:.2f}'
    )
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='0 line')
    ax2.axhline(1, color=(1, 0, 0, 0.4), linestyle='--', linewidth=1, label='+1 line')
    ax2.axhline(2, color=(1, 0, 0, 0.2), linestyle='--', linewidth=1, label='+2 line')
    ax2.axhline(3, color='red', linestyle='--', linewidth=1, label='+3 line')
    ax2.axhline(-1, color=(0, 1, 0, 0.4), linestyle='--', linewidth=1, label='-1 line')
    ax2.axhline(-2, color=(0, 1, 0, 0.2), linestyle='--', linewidth=1, label='-2 line')
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1, label='-3 line')

    if zScore_valid.empty or zScore_valid.max() > 4 or zScore_valid.min() < -4:
        ax2.set_ylim(-4, 4)
    else:
        ax2.set_ylim(zScore_valid.min() - 0.5, zScore_valid.max() + 0.5)

    ax2.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate(rotation=45)

    # Watermark
    watermark_text = 'Gonçalo Duarte'
    fig.text(
        0.5, 0.5, watermark_text,
        fontsize=50,
        color='gray',
        alpha=0.2,
        ha='center',
        va='center',
        rotation=10,
        zorder=1
    )

    # Show the plot using Streamlit
    st.pyplot(fig)


if __name__ == '__main__':
    st.set_page_config(layout="wide") 
    st.title("Bitcoin Pi-Cycle Indicator Dashboard")
    
    file_to_plot = 'Bitcoin Historical Data_test.csv'
    
    # --- NEW: Streamlit Sidebar for Date Selection ---
    
    # 1. Determine the Min/Max dates from the file for the slider
    try:
        # Load minimum data just to find the date range
        temp_df = pd.read_csv(file_to_plot, thousands=',')
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%m/%d/%Y', errors='coerce')
        min_date_available = temp_df['Date'].min().date()
        max_date_available = temp_df['Date'].max().date()
    except Exception:
        # Fallback in case of error
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
    
    # Get the selected Python date objects
    selected_start_date = date_range[0] 
    selected_end_date = date_range[1]   

    # --- CRITICAL FIX: Convert Python date objects to Pandas Timestamps ---
    # This resolves the TypeError by ensuring the filtering comparison is valid.
    start_date_filter = pd.to_datetime(selected_start_date)
    end_date_filter = pd.to_datetime(selected_end_date)
    # ----------------------------------------------------------------------
    
    # 3. Call the plotting function with the CORRECTLY TYPED dates
    plot_timeseries_data(file_to_plot, start_date_filter, end_date_filter)
