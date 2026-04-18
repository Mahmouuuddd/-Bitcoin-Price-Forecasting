import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import streamlit as st

# ------------------------------------------------------------------
# Data parsing and validation
# ------------------------------------------------------------------
def parse_btc_csv(df: pd.DataFrame):
    """
    Auto-detect date and price columns from common Kaggle/Binance BTC formats.
    Returns (date_col, list_of_price_candidates).
    """
    df.columns = [c.strip() for c in df.columns]

    # 1. Detect timestamp column
    date_candidates = [
        c for c in df.columns
        if c.lower() in [
            'date', 'timestamp', 'time', 'datetime', 'open time', 'close time',
            'day', 'period', 'unix', 'opentime', 'closetime'
        ]
    ]
    if not date_candidates:
        date_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if not date_candidates:
        raise ValueError(
            "No date/timestamp column found. Expected columns like 'Date', 'Timestamp', 'Time', 'Open Time', or 'unix'."
        )
    date_col = date_candidates[0]

    # 2. Detect price columns
    price_candidates = [
        c for c in df.columns
        if c.lower() in [
            'close', 'open', 'high', 'low', 'price', 'value', 'last',
            'close/last', 'adj close', 'closing price'
        ]
    ]
    if not price_candidates:
        price_candidates = [
            c for c in df.columns
            if any(p in c.lower() for p in ['close', 'open', 'high', 'low', 'price'])
        ]
    if not price_candidates:
        raise ValueError(
            "No OHLC/price columns found. Expected 'Close', 'Open', 'High', 'Low', or 'Price'."
        )

    return date_col, price_candidates


def load_and_validate(uploaded_file, price_col: str, date_col: str):
    """
    Load CSV, parse dates, sort chronologically, handle missing values,
    and resample to daily frequency if needed.
    """
    # Try different encodings
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    df.columns = [c.strip() for c in df.columns]

    # Clean price column – remove non‑digit/period characters
    df[price_col] = df[price_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    # Parse dates – handle Unix timestamps and normal dates
    if date_col.lower() == 'unix':
        # Try seconds first, then milliseconds
        try:
            df[date_col] = pd.to_datetime(df[date_col], unit='s', errors='coerce')
        except:
            df[date_col] = pd.to_datetime(df[date_col], unit='ms', errors='coerce')
    else:
        # Try common formats without 'infer_datetime_format' (deprecated)
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors='coerce',
            dayfirst=False
        )
        # If many failures, try dayfirst=True
        if df[date_col].isna().mean() > 0.5:
            df[date_col] = pd.to_datetime(
                df[date_col],
                errors='coerce',
                dayfirst=True
            )

    # Drop rows with invalid dates or prices
    n_before = len(df)
    df = df.dropna(subset=[date_col, price_col])
    n_dropped = n_before - len(df)

    if len(df) == 0:
        raise ValueError("No valid data after cleaning. Check date and price columns.")

    df = df.rename(columns={date_col: 'ds', price_col: 'y'})
    df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    df['ds'] = df['ds'].dt.tz_localize(None)  # remove timezone if present

    # --- Resample to daily if data is intraday ---
    time_diff = df['ds'].diff().median()
    if time_diff < pd.Timedelta(days=1):
        st.info(f"⏱️ Intraday data detected (median interval: {time_diff}). Resampling to daily prices.")
        df = df.set_index('ds')
        # Use the last price of each day as the daily close
        df_daily = df['y'].resample('D').last().dropna()
        df = df_daily.reset_index()
        df.columns = ['ds', 'y']

    # Ensure daily frequency and forward‑fill any missing calendar days
    df = df.set_index('ds')
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_date_range)
    df['y'] = df['y'].ffill()
    df = df.reset_index().rename(columns={'index': 'ds'})

    if len(df) < 60:
        raise ValueError(f"Not enough daily data – need at least 60 rows, got {len(df)}.")

    return df, n_dropped


def add_technical_indicators(df, sma_window=20, ema_window=20):
    df = df.copy()
    df['SMA'] = df['y'].rolling(window=sma_window).mean()
    df['EMA'] = df['y'].ewm(span=ema_window, adjust=False).mean()
    return df



# ------------------------------------------------------------------
# Forecasting models
# ------------------------------------------------------------------
def run_prophet(df, horizon, interval_width):
    from prophet import Prophet
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size][['ds', 'y']]
    test_df  = df.iloc[train_size:][['ds', 'y']]

    m = Prophet(interval_width=interval_width, daily_seasonality=False,
                weekly_seasonality=True, yearly_seasonality=True)
    m.fit(train_df)

    # Backtest
    test_forecast = m.predict(test_df[['ds']])
    mae  = np.mean(np.abs(test_df['y'].values - test_forecast['yhat'].values))
    rmse = np.sqrt(np.mean((test_df['y'].values - test_forecast['yhat'].values)**2))

    # Full model
    m2 = Prophet(interval_width=interval_width, daily_seasonality=False,
                 weekly_seasonality=True, yearly_seasonality=True)
    m2.fit(df[['ds', 'y']])
    future = m2.make_future_dataframe(periods=horizon)
    forecast = m2.predict(future)

    return forecast, mae, rmse, train_size


def run_arima(df, horizon, interval_width):
    from statsmodels.tsa.arima.model import ARIMA

    train_size = int(len(df) * 0.8)
    train = df['y'].iloc[:train_size].values
    test  = df['y'].iloc[train_size:].values

    # Backtest
    model_bt = ARIMA(train, order=(5, 1, 0))
    res_bt   = model_bt.fit()
    pred_bt  = res_bt.forecast(steps=len(test))
    mae  = np.mean(np.abs(test - pred_bt))
    rmse = np.sqrt(np.mean((test - pred_bt)**2))

    # Full forecast
    model = ARIMA(df['y'].values, order=(5, 1, 0))
    res   = model.fit()
    forecast_res = res.get_forecast(steps=horizon)
    pred_mean = forecast_res.predicted_mean
    pred_ci   = forecast_res.conf_int(alpha=1 - interval_width)

    last_date = df['ds'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]

    forecast_df = pd.DataFrame({
        'ds': df['ds'].tolist() + future_dates,
        'yhat': np.concatenate([df['y'].values, pred_mean]),
        'yhat_lower': np.concatenate([df['y'].values, pred_ci[:, 0]]),
        'yhat_upper': np.concatenate([df['y'].values, pred_ci[:, 1]]),
    })

    return forecast_df, mae, rmse, train_size


def build_chart(df, forecast, train_size, horizon, model_name, show_sma, show_ema, price_col):
    fig = go.Figure()

    forecast_start_idx = len(df)
    fut_forecast  = forecast.iloc[forecast_start_idx - 1:]

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fut_forecast['ds'], fut_forecast['ds'][::-1]]),
        y=pd.concat([fut_forecast['yhat_upper'], fut_forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(247, 147, 26, 0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        showlegend=True,
        hoverinfo='skip',
    ))

    # Historical price
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        line=dict(color='#e8e8f0', width=1.5),
        name='Historical Price',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:,.0f}<extra></extra>',
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        line=dict(color='#f7931a', width=2, dash='solid'),
        name=f'{model_name} Forecast',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast: $%{y:,.0f}<extra></extra>',
    ))

    # Forecast start marker
    start_date = df['ds'].iloc[-1]
    start_price = forecast.loc[forecast['ds'] == start_date, 'yhat']
    if not start_price.empty:
        fig.add_trace(go.Scatter(
            x=[start_date], y=[start_price.values[0]],
            mode='markers',
            marker=dict(color='#f7931a', size=12, symbol='circle',
                        line=dict(color='white', width=2)),
            name='Forecast Start',
            hovertemplate='<b>Forecast Start</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>',
        ))

    fig.add_vline(x=start_date, line_width=1, line_dash="dash",
                  line_color="rgba(247,147,26,0.5)")

    # Technical indicators
    if show_sma or show_ema:
        df_ind = add_technical_indicators(df)
        if show_sma:
            fig.add_trace(go.Scatter(
                x=df_ind['ds'], y=df_ind['SMA'],
                line=dict(color='#6ee7f7', width=1.2, dash='dot'),
                name='SMA-20',
                hovertemplate='SMA-20: $%{y:,.0f}<extra></extra>',
            ))
        if show_ema:
            fig.add_trace(go.Scatter(
                x=df_ind['ds'], y=df_ind['EMA'],
                line=dict(color='#b47fff', width=1.2, dash='dot'),
                name='EMA-20',
                hovertemplate='EMA-20: $%{y:,.0f}<extra></extra>',
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,20,0.6)',
        font=dict(family='Space Mono, monospace', color='#e8e8f0'),
        xaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            zeroline=False, title='Date',
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            zeroline=False, title=f'BTC Price ({price_col}) — USD',
            tickprefix='$', tickformat=',.0f',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)', font=dict(size=11),
        ),
        hovermode='x unified',
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def compute_returns(prices):
    return prices.pct_change().dropna() * 100


def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi