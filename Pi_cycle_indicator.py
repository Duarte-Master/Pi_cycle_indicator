import streamlit as st # <-- NEW: Import Streamlit for web display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from matplotlib.widgets import RangeSlider # <-- REMOVED: Matplotlib widgets won't work in Streamlit


# import pandas_ta as ta  <-- Removed pandas_ta import


# --- Custom SMA Function ---
def sma(series, period):
    """
    Calculates the Simple Moving Average (SMA) for a given series.
    Uses min_periods=period to return a value only when the full window is available.
    """
    # Use min_periods=period for standard indicator calculation.
    return series.rolling(window=period, min_periods=period).mean()


# --- End Custom SMA Function ---


def plot_timeseries_data(filepath):
    """
    Loads time series data from a CSV and plots the price with the Pi-Cycle Z-Score indicator.

    Args:
        filepath (str): The path to the CSV file (e.g., 'Bitcoin Historical Data.csv').
    """
    # Use st.write for user feedback in Streamlit
    st.write(f"Loading data from: {filepath}...")

    try:
        # 1. Load the CSV file into a DataFrame, handling comma thousands separators
        df = pd.read_csv(filepath, thousands=',')
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please check the path.")
        return
    except Exception as e:
        st.error(f"An error occurred while reading the CSV: {e}")
        return

    # --- Data Preprocessing ---

    date_column_name = 'Date'
    price_column_name = 'Price'

    # Basic column existence check
    if date_column_name not in df.columns or price_column_name not in df.columns:
        missing = [col for col in [date_column_name, price_column_name] if col not in df.columns]
        st.error(f"Error: Missing required columns: {', '.join(missing)}. Available columns: {df.columns.tolist()}")
        return

    # 2. Convert the Date column to proper datetime objects (MM/DD/YYYY)
    try:
        df[date_column_name] = pd.to_datetime(df[date_column_name], format='%m/%d/%Y')
    except ValueError as e:
        st.error(f"Error parsing dates: {e}. Check if the date format is strictly 'MM/DD/YYYY'.")
        return

    # Set the Date column as the index
    df.set_index(date_column_name, inplace=True)

    # Select the price column
    price_data = df[price_column_name]

    # Clean price data
    price_data = pd.to_numeric(price_data, errors='coerce')
    price_data.dropna(inplace=True)

    # --- CRITICAL FIX: Reindex to fill calendar gaps (New Addition) ---
    if not price_data.empty:
        # 1. Define a continuous calendar date range
        start_date = price_data.index.min()
        end_date = price_data.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # 2. Reindex and use forward-fill (ffill) to use the last known price for missing days.
        # This ensures the 350-day SMA covers 350 calendar days.
        price_data = price_data.reindex(full_date_range).ffill()

        # Drop any remaining NA values (safe cleanup)
    price_data.dropna(inplace=True)
    # ------------------------------------------------------------------

    # --- Technical Analysis Calculation (Using Fixed Logic) ---

    # 1. Calculate the required Moving Averages using the custom 'sma' function
    ma_111 = sma(price_data, period=111)
    ma_350_double = 2 * sma(price_data, period=350)

    # 2. Calculate the Pi-Cycle Ratio (The base value)
    Pi_cycle_ratio = 2 - (ma_350_double / ma_111).replace([np.inf, -np.inf], np.nan)

    # 3. Calculate Z-Score components
    thresholdTop = 1.05
    thresholdBottom = -1.10

    top_bands = thresholdTop
    bottom_bands = thresholdBottom

    mean = (top_bands + bottom_bands) / 2
    bands_range = top_bands - bottom_bands

    # Calculation of StdDev and Z-Score, handling division by zero (bands_range != 0)
    stdDev = bands_range / 6 if bands_range != 0 else 0

    # Calculate Z-Score
    if stdDev != 0:
        # Use the Pi_cycle_oscillator (centered value) for the Z-Score calculation
        zScore = (Pi_cycle_ratio - mean) / stdDev
    else:
        zScore = pd.Series(0, index=Pi_cycle_ratio.index)

    # Drop initial NaN values from the Z-Score series
    zScore_valid = zScore.dropna()

    # --- Plotting Setup ---

    # 1. Setup a figure with two subplots: Price (ax1) and Z-Score (ax2)
    # Adjust fig size for better viewing in Streamlit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig.subplots_adjust(bottom=0.25) # <-- REMOVED: No need for bottom space without slider

    # The main price axis is now ax1
    ax = ax1

    # Initial plot of the entire time series
    line, = ax.plot(price_data.index, price_data.values, color='black', linewidth=1, label=price_column_name)

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
    ax.set_title(f'BTC/USD Price Time Series (Log Scale) and Pi-Cycle Z-Score from {filepath}')
    ax.set_ylabel('Price (USD) [Log Scale]')

    # APPLY LOG SCALE TO Y-AXIS
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Apply date formatting to the main plot's x-axis
    date_formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)

    # --- Z-Score Subplot (ax2) ---

    # 1. Safely calculate the latest (last) Z-Score value
    if not zScore_valid.empty:
        latest_z_score = zScore_valid.iloc[-1]
    else:
        latest_z_score = np.nan

    # 2. Plot the Z-Score with the corrected label
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

    # Z-Score Horizontal Lines
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='0 line')

    # Red Lines (Positive Z-Scores)
    ax2.axhline(1, color=(1, 0, 0, 0.4), linestyle='--', linewidth=1, label='+1 line')
    ax2.axhline(2, color=(1, 0, 0, 0.2), linestyle='--', linewidth=1, label='+2 line')
    ax2.axhline(3, color='red', linestyle='--', linewidth=1, label='+3 line')

    # Green Lines (Negative Z-Scores)
    ax2.axhline(-1, color=(0, 1, 0, 0.4), linestyle='--', linewidth=1, label='-1 line')
    ax2.axhline(-2, color=(0, 1, 0, 0.2), linestyle='--', linewidth=1, label='-2 line')
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1, label='-3 line')

    # Ensure the Z-Score plot has a reasonable range
    if zScore_valid.empty or zScore_valid.max() > 4 or zScore_valid.min() < -4:
        ax2.set_ylim(-4, 4)
    else:
        ax2.set_ylim(zScore_valid.min() - 0.5, zScore_valid.max() + 0.5)

    # Apply date formatting and rotation to the lower plot (ax2)
    ax2.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate(rotation=45)

    # --- REMOVED INTERACTIVE WIDGET CODE (RangeSlider, Hover, Connects) ---
    # The following interactive Matplotlib code must be removed or commented out 
    # because it is not compatible with the Streamlit web environment.

    # 1. Convert dates to Matplotlib's internal numerical format, which the slider uses
    # dates_num = mdates.date2num(price_data.index.to_pydatetime())

    # 3. Setup Annotation (Tooltip) - REMOVED

    # 4. Define the hover event handler - REMOVED

    # 5. Setup Slider Axes - REMOVED

    # 6. Create the Range Slider - REMOVED

    # 7. Define the range update function - REMOVED

    # 8. Connect the update function to the slider's change event - REMOVED

    # 9. Connect the hover handler - REMOVED
    
    # Run the update function once to ensure the initial view is set correctly - REMOVED
    # update(slider.val)

    # Add the two-line Watermark using fig.text()
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
    st.pyplot(fig) # <-- CRITICAL CHANGE: Replace plt.show() with st.pyplot(fig)


if __name__ == '__main__':
    st.title("Bitcoin Pi-Cycle Indicator Dashboard") # <-- NEW: Streamlit Title
    
    # Define the file path from your uploaded data
    file_to_plot = 'Bitcoin Historical Data_test.csv'
    
    # Run the plotting function
    plot_timeseries_data(file_to_plot)
