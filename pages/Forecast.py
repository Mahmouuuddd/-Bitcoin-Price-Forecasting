import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore")

from utils import (
    parse_btc_csv, load_and_validate, run_prophet, run_arima, build_chart
)

st.set_page_config(page_title="Forecast", layout="wide")

st.title(" ₿ Bitcoin Price Forecasting")
st.markdown(
    '<p style="font-family:Space Mono,monospace;font-size:12px;color:#666;letter-spacing:1px;">'
    'TIME-SERIES ANALYSIS · PROPHET · ARIMA · INTERACTIVE VISUALIZATION</p>',
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ₿ BTC Forecaster")
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Kaggle BTC CSV", type=["csv"],
                                     help="Upload any Bitcoin CSV (daily or intraday).",
                                     key="forecast_uploader")

    st.markdown('<div class="section-header">Price Column</div>', unsafe_allow_html=True)
    price_col_select = st.selectbox("Price to forecast", ["Close", "Open", "High", "Low"],
                                    index=0)

    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio("Algorithm", ["Prophet", "ARIMA"],
                            help="Prophet: additive seasonality decomposition.\nARIMA: autoregressive integrated moving average.")

    st.markdown('<div class="section-header">Forecast Parameters</div>', unsafe_allow_html=True)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=180,
                        value=30, step=7)
    confidence = st.select_slider("Confidence interval", options=[0.80, 0.90, 0.95],
                                  value=0.95, format_func=lambda x: f"{int(x*100)}%")

    st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)
    show_sma = st.toggle("SMA-20", value=False)
    show_ema = st.toggle("EMA-20", value=False)

    st.markdown("---")
    run_btn = st.button("⚡ Generate Forecast", type="primary")

# --- Main area ---
st.markdown("# Bitcoin Price Forecasting Portal")
st.markdown(
    '<p style="font-family:Space Mono,monospace;font-size:12px;color:#666;letter-spacing:1px;">'
    'TIME-SERIES ANALYSIS · PROPHET · ARIMA · INTERACTIVE VISUALIZATION</p>',
    unsafe_allow_html=True
)

# Optional: auto-load pre-cleaned file (only if no uploaded file and no data in session)
CLEANED_PATH = os.path.join(os.path.dirname(__file__), '..', 'btc_cleaned.csv')
precleaned_available = os.path.exists(CLEANED_PATH)

# --- Data Loading Logic with Session State Persistence ---
if "df" in st.session_state:
    # Use existing data from session
    df = st.session_state.df
    n_dropped = st.session_state.get("n_dropped", 0)
    price_col_used = st.session_state.get("price_col_used", "Close")
    st.success(f"Using previously loaded dataset: **{st.session_state.get('uploaded_file_name', 'pre-cleaned file')}**")
else:
    # No data in session – try to load from uploaded file or pre-cleaned CSV
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            date_col, price_candidates = parse_btc_csv(raw_df)
            matched_col = next((c for c in price_candidates if c.lower() == price_col_select.lower()), price_candidates[0])
            uploaded_file.seek(0)
            df, n_dropped = load_and_validate(uploaded_file, matched_col, date_col)

            # Store in session
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.price_col_used = matched_col
            st.session_state.n_dropped = n_dropped
            price_col_used = matched_col
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ Data loading error: {e}</div>', unsafe_allow_html=True)
            st.stop()
    elif precleaned_available:
        try:
            df = pd.read_csv(CLEANED_PATH, parse_dates=['ds'])
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
            df = df.sort_values('ds').reset_index(drop=True)
            if 'y' not in df.columns:
                raise ValueError("Pre-cleaned CSV is missing the 'y' column.")
            if len(df) < 60:
                raise ValueError(f"Not enough rows in pre-cleaned CSV: {len(df)}.")

            # Store in session
            st.session_state.df = df
            st.session_state.uploaded_file_name = "btc_cleaned.csv"
            st.session_state.price_col_used = "y"
            st.session_state.n_dropped = 0
            price_col_used = "y"
            n_dropped = 0
            st.markdown(
                '<div class="info-box">⚡ Pre-cleaned dataset loaded automatically.</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ Failed to load pre-cleaned CSV: {e}</div>', unsafe_allow_html=True)
            st.stop()
    else:
        st.markdown("""
        <div class="upload-area">
            <p style="font-family:Space Mono,monospace;font-size:13px;color:#f7931a;letter-spacing:2px;margin-bottom:8px;">
            ₿ UPLOAD A KAGGLE BTC CSV TO BEGIN</p>
            <p style="font-size:12px;color:#555;">
            Supported: Any Bitcoin CSV with a date column and OHLCV/price column.<br>
            <b>Intraday data (1m, 1h) is automatically resampled to daily.</b><br><br>
            <b>Tip:</b> Place a <code>btc_cleaned.csv</code> file in the app folder to auto-load.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# --- Dataset Summary ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Data Points</div>
        <div class="metric-value">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Date Range</div>
        <div class="metric-value" style="font-size:14px;">{df['ds'].min().strftime('%Y-%m-%d')}<br>→ {df['ds'].max().strftime('%Y-%m-%d')}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Price</div>
        <div class="metric-value">${df['y'].iloc[-1]:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rows Dropped</div>
        <div class="metric-value">{n_dropped}</div>
    </div>""", unsafe_allow_html=True)

if n_dropped > 0:
    st.info(f"ℹ {n_dropped} rows were removed due to missing or unparseable values.")

# --- Show Historical Chart Until Forecast is Triggered ---
if not run_btn:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        line=dict(color='#f7931a', width=1.5),
        fill='tozeroy', fillcolor='rgba(247,147,26,0.06)',
        name='BTC Price',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,20,0.6)',
        font=dict(family='Space Mono, monospace', color='#e8e8f0'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   tickprefix='$', tickformat=',.0f'),
        height=480, margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<p style="text-align:center;font-family:Space Mono,monospace;font-size:12px;color:#555;">'
        'Configure parameters in the sidebar and click ⚡ Generate Forecast</p>',
        unsafe_allow_html=True
    )
    st.stop()

# --- Run Model ---
with st.spinner(f"Training {model_choice} model…"):
    try:
        if model_choice == "Prophet":
            forecast, mae, rmse, train_size = run_prophet(df, horizon, confidence)
        else:
            forecast, mae, rmse, train_size = run_arima(df, horizon, confidence)
    except ImportError as e:
        pkg = "prophet" if "prophet" in str(e).lower() else "statsmodels"
        st.markdown(f'<div class="error-box">⚠ Missing package: {pkg}. Run: pip install {pkg}</div>',
                    unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠ Forecasting error: {e}</div>', unsafe_allow_html=True)
        st.stop()

# --- Metrics ---
st.markdown('<div class="section-header">Backtest Performance (80/20 Split)</div>',
            unsafe_allow_html=True)
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAE (USD)</div>
        <div class="metric-value">${mae:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RMSE (USD)</div>
        <div class="metric-value">${rmse:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    fut = forecast[forecast['ds'] > df['ds'].max()]
    final_price = fut['yhat'].iloc[-1] if not fut.empty else df['y'].iloc[-1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecast ({horizon}d)</div>
        <div class="metric-value">${final_price:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with mc4:
    change_pct = (final_price - df['y'].iloc[-1]) / df['y'].iloc[-1] * 100
    color = "#4ade80" if change_pct >= 0 else "#f87171"
    arrow = "▲" if change_pct >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Change</div>
        <div class="metric-value" style="color:{color};">{arrow} {abs(change_pct):.1f}%</div>
    </div>""", unsafe_allow_html=True)

# --- Main Chart ---
st.markdown('<div class="section-header">Interactive Price Chart</div>', unsafe_allow_html=True)
fig = build_chart(df, forecast, train_size, horizon, model_choice, show_sma, show_ema, price_col_used)
st.plotly_chart(fig, use_container_width=True)

# --- Forecast Table ---
with st.expander("📋 Forecast Data Table"):
    fut_table = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    fut_table.columns = ['Date', 'Forecast (USD)', 'Lower Bound', 'Upper Bound']
    fut_table['Date'] = fut_table['Date'].dt.strftime('%Y-%m-%d')
    for c in ['Forecast (USD)', 'Lower Bound', 'Upper Bound']:
        fut_table[c] = fut_table[c].apply(lambda x: f"${x:,.2f}")
    st.dataframe(fut_table, use_container_width=True, hide_index=True)