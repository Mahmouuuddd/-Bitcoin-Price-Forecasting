import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from utils import compute_returns, compute_rsi

st.set_page_config(page_title="Statistics", layout="wide")
st.title(" ₿ Bitcoin Exploratory Data Analysis")
st.markdown(
    '<p style="font-family:Space Mono,monospace;font-size:12px;color:#666;letter-spacing:1px;">'
    'KEY METRICS · DISTRIBUTIONS · VOLATILITY · TECHNICAL INDICATORS</p>',
    unsafe_allow_html=True
)

# Load data from session state
if "df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the **Forecast** page and upload a CSV file first.")
    st.stop()

df = st.session_state.df.copy()
st.success(f" Using dataset: **{st.session_state.get('uploaded_file_name', 'uploaded file')}** ({len(df)} days)")

# ------------------------------------------------------------------
# Key Metrics
# ------------------------------------------------------------------
returns = compute_returns(df['y'])
volatility = returns.std() * np.sqrt(365)

st.markdown('<div class="section-header"> Key Statistics</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Price</div><div class="metric-value">${df["y"].mean():,.2f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Max Price</div><div class="metric-value">${df["y"].max():,.2f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Min Price</div><div class="metric-value">${df["y"].min():,.2f}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Daily Vol</div><div class="metric-value">{returns.std():.2f}%</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Annual Vol</div><div class="metric-value">{volatility:.2f}%</div></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Skewness</div><div class="metric-value">{returns.skew():.3f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Kurtosis</div><div class="metric-value">{returns.kurtosis():.3f}</div></div>', unsafe_allow_html=True)
with col3:
    sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
    st.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.3f}</div></div>', unsafe_allow_html=True)
with col4:
    total_ret = (df['y'].iloc[-1] / df['y'].iloc[0] - 1) * 100
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Return</div><div class="metric-value">{total_ret:.2f}%</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Price Chart with Indicators
# ------------------------------------------------------------------
st.markdown('<div class="section-header"> Price & Technical Indicators</div>', unsafe_allow_html=True)
df['SMA20'] = df['y'].rolling(20).mean()
df['SMA50'] = df['y'].rolling(50).mean()
df['BB_mid'] = df['y'].rolling(20).mean()
df['BB_upper'] = df['BB_mid'] + 2 * df['y'].rolling(20).std()
df['BB_lower'] = df['BB_mid'] - 2 * df['y'].rolling(20).std()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Close', line=dict(color='#f7931a', width=1.5)))
fig1.add_trace(go.Scatter(x=df['ds'], y=df['SMA20'], name='SMA 20', line=dict(color='#6ee7f7', width=1, dash='dot')))
fig1.add_trace(go.Scatter(x=df['ds'], y=df['SMA50'], name='SMA 50', line=dict(color='#b47fff', width=1, dash='dot')))
fig1.add_trace(go.Scatter(x=df['ds'], y=df['BB_upper'], name='BB Upper', line=dict(color='gray', width=0.5), showlegend=False))
fig1.add_trace(go.Scatter(x=df['ds'], y=df['BB_lower'], name='BB Lower', line=dict(color='gray', width=0.5),
                           fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False))
fig1.update_layout(template='plotly_dark', height=450, hovermode='x unified',
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,20,0.6)',
                   font=dict(family='Space Mono, monospace', color='#e8e8f0'),
                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Date'),
                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Price (USD)', tickprefix='$', tickformat=',.0f'))
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------
# Returns Distribution
# ------------------------------------------------------------------
st.markdown('<div class="section-header"> Daily Returns Distribution</div>', unsafe_allow_html=True)
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Q-Q Plot"))
fig2.add_trace(go.Histogram(x=returns, nbinsx=80, marker_color='#f7931a',
                            histnorm='probability density'), row=1, col=1)
x_range = np.linspace(returns.min(), returns.max(), 100)
norm_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
fig2.add_trace(go.Scatter(x=x_range, y=norm_pdf, mode='lines', name='Normal fit',
                          line=dict(color='white')), row=1, col=1)

qq = stats.probplot(returns, dist="norm")
fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='#f7931a', size=3)), row=1, col=2)
fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * qq[0][0], mode='lines',
                          line=dict(color='white')), row=1, col=2)
fig2.update_layout(template='plotly_dark', showlegend=False, height=400,
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,20,0.6)',
                   font=dict(family='Space Mono, monospace', color='#e8e8f0'))
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Rolling Volatility
# ------------------------------------------------------------------
st.markdown('<div class="section-header">📉 Rolling Volatility (30‑Day)</div>', unsafe_allow_html=True)
df['rolling_vol'] = returns.rolling(30).std() * np.sqrt(365)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['ds'], y=df['rolling_vol'], fill='tozeroy', line=dict(color='#f7931a')))
fig3.update_layout(template='plotly_dark', height=350,
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,20,0.6)',
                   font=dict(family='Space Mono, monospace', color='#e8e8f0'),
                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Date'),
                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Annualized Volatility (%)'))
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# RSI
# ------------------------------------------------------------------
st.markdown('<div class="section-header">📊 Relative Strength Index (RSI)</div>', unsafe_allow_html=True)
df['RSI'] = compute_rsi(df['y'])
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df['ds'], y=df['RSI'], line=dict(color='#f7931a')))
fig4.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
fig4.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
fig4.update_layout(template='plotly_dark', height=350, yaxis=dict(range=[0, 100]),
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,20,0.6)',
                   font=dict(family='Space Mono, monospace', color='#e8e8f0'),
                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Date'),
                   yaxis_title='RSI')  # <-- FIXED: use yaxis_title instead of duplicate yaxis
st.plotly_chart(fig4, use_container_width=True)