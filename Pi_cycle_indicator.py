# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # fetch historical price data from Yahoo Finance
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots

# --- Custom SMA Function ---
def sma(series, period):
    """
    Calculates the Simple Moving Average (SMA) for a given series.
    Uses min_periods=period to return a value only when the full window is available.
    """
    return series.rolling(window=period, min_periods=period).mean()

# --- End Custom SMA Function ---

@st.cache_data(show_spinner=False)
def fetch_price_data(start_date="2013-01-01"):
    """Fetches historical Bitcoin price data using Yahoo Finance via yfinance.

    The function uses Streamlit's cache to avoid repeated network calls. Data is
    trimmed to begin at **start_date** (default January 1st 2013). The returned
    DataFrame has a datetime index and a single column named **Price**.
    """
    df = None
    for attempt in range(3):
        try:
            df = yf.download("BTC-USD", start=start_date, interval="1d", progress=False)
            break
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "too many requests" in msg:
                import time
                time.sleep(5)
                continue
            else:
                st.error(f"Error fetching data from Yahoo Finance: {e}")
                return pd.DataFrame(columns=["Price"])
    if df is None or df.empty:
        local_path = "Bitcoin Historical Data_test.csv"
        try:
            # apply thousands=',' so numeric values parse correctly
            df = pd.read_csv(local_path, parse_dates=True, index_col=0, thousands=',')
            st.warning("Using local CSV data because online download failed.")
        except Exception:
            if df is None or df.empty:
                st.error("Unable to obtain price data from Yahoo Finance or local file.")
                return pd.DataFrame(columns=["Price"])
    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        return pd.DataFrame(columns=["Price"])
    if "Close" not in df.columns:
        st.error("Expected 'Close' column in data from Yahoo Finance.")
        return pd.DataFrame(columns=["Price"])
    price_df = df[["Close"]].rename(columns={"Close": "Price"})
    price_df.index = pd.to_datetime(price_df.index)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    price_df = price_df[price_df.index >= pd.to_datetime(start_date)]
    return price_df


def plot_timeseries_data(df=None):
    """
    Calculates indicators and plots the result using Plotly.

    If *df* is omitted the function will automatically fetch data via API.
    """
    st.write(f"Created by Goncalo Duarte")

    # --- Data Loading and Preprocessing (Updated) ---
    if df is None:
        df = fetch_price_data()
    if df is None:
        st.error("Data fetch failed; received None from fetch_price_data.")
        return
    if df.empty:
        st.error("No data available to plot.")
        return

    st.write(f"Data from {df.index.min().date()} to {df.index.max().date()}")

    if "Price" not in df.columns:
        st.error("Error: Dataframe must contain a 'Price' column.")
        return

    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)

    price_data = pd.to_numeric(df["Price"], errors="coerce")
    price_data.dropna(inplace=True)

    # CRITICAL Fix: reindex to fill gaps
    if not price_data.empty:
        start_date = price_data.index.min()
        end_date = price_data.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        price_data = price_data.reindex(full_date_range).ffill()
    price_data.dropna(inplace=True)

    # Pi-cycle calculations
    ma_111 = sma(price_data, period=111)
    ma_350_double = 2 * sma(price_data, period=350)
    Pi_cycle_ratio = 2 - (ma_350_double / ma_111).replace([np.inf, -np.inf], pd.NA)

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
        row_heights=[3,1],
        subplot_titles=("BTC/USD Price Time Series (Log Scale) and Pi-Cycle Lines","Pi-Cycle Z-Score")
    )

    fig.add_trace(
        go.Scatter(x=price_data.index, y=price_data.values, mode="lines", name="Price",
                   line=dict(color="orange",width=2),
                   hovertemplate="Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=ma_111.index, y=ma_111.values, mode="lines", name="111 SMA",
                   line=dict(color="yellow",width=2),
                   hovertemplate="Date: %{x}<br>111 SMA: $%{y:,.0f}<extra></extra>"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=ma_350_double.index,y=ma_350_double.values,mode="lines",name="2 * 350 SMA",
                   line=dict(color="green",width=2),
                   hovertemplate="Date: %{x}<br>2*350 SMA: $%{y:,.0f}<extra></extra>"),
        row=1, col=1
    )

    fig.update_yaxes(type="log", row=1, col=1, title_text="Price (USD) [Log Scale]", showgrid=True)

    if not zScore_valid.empty:
        latest_z_score = zScore_valid.iloc[-1]
    else:
        latest_z_score = np.nan

    fig.add_trace(
        go.Scatter(x=zScore_valid.index, y=zScore_valid.values, mode="lines",
                   name=f"Pi Cycle Z-Score: {latest_z_score:.2f}",
                   line=dict(color="red",width=2),
                   hovertemplate="Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>"),
        row=2, col=1
    )

    # Neutral/Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    # Positive Lines (Red/Overvalued) - Using RGBA to mimic Matplotlib's alpha
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(255, 0, 0, 0.33)", line_width=1, row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.66)", line_width=1, row=2, col=1)
    fig.add_hline(y=3, line_dash="dash", line_color="rgba(255, 0, 0, 1.00)", line_width=1, row=2, col=1)

    # Negative Lines (Green/Undervalued)
    fig.add_hline(y=-1, line_dash="dash", line_color="rgba(0, 255, 0, 0.33)", line_width=1, row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="rgba(0, 255, 0, 0.66)", line_width=1, row=2, col=1)
    fig.add_hline(y=-3, line_dash="dash", line_color="rgba(0, 255, 0, 1.00)", line_width=1, row=2, col=1)

    # 6. Color Bar Background (Approximation using Scatter and color mapping)
    # Get Colormap from Matplotlib and convert to a Plotly colorscale
    # Matplotlib 3.7+ deprecates plt.cm.get_cmap
    cmap = plt.colormaps.get('RdYlGn_r')
    norm = mcolors.Normalize(vmin=-3, vmax=3)
    # Create the Plotly color scale (list of [relative value, color])
    plotly_colorscale = []
    for i in np.linspace(0, 1, 11): # Sample 11 points for gradient
        rgb = cmap(i)[:3]
        hex_color = mcolors.to_hex(rgb)
        plotly_colorscale.append([i, hex_color])

    # Plotly Scatter for gradient coloring (similar to your Matplotlib scatter)
    if not zScore_valid.empty:
        fig.add_trace(
            go.Scatter(
                x=zScore_valid.index,
                y=zScore_valid.values,
                mode='markers',
                marker=dict(
                    color=zScore_valid.values,
                    colorscale=plotly_colorscale,
                    cmin=-3,
                    cmax=3,
                    size=3, # Smaller size for a line-like appearance
                    showscale=False
                ),
                name='Z-Score Gradient',
                showlegend=False
            ),
            row=2, col=1
        )

    fig.update_yaxes(title_text="Z-Score", range=[-4.5,4.5], row=2,col=1, showgrid=True)

    fig.update_layout(
        title_text="Bitcoin Pi-Cycle Indicator",
        height=820, template="plotly_dark", hovermode="x unified",
        margin=dict(b=150),
        xaxis=dict(rangeslider=dict(visible=True,thickness=0.04), type="date", title_text="Date"),
        xaxis2=dict(range=None)
    )

    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Bitcoin Pi-Cycle Indicator Dashboard")
    plot_timeseries_data()
