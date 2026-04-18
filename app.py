import streamlit as st

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(
    page_title="BTC Analytics Suite",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Page Config ---
st.set_page_config(
    page_title="BTC Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (unchanged) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.metric-card {
    background: rgba(247, 147, 26, 0.08);
    border: 1px solid rgba(247, 147, 26, 0.25);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #f7931a;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #f7931a;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(247, 147, 26, 0.3);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.stButton>button {
    background: linear-gradient(135deg, #f7931a, #e8640a) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.stButton>button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

.stSelectbox label, .stSlider label, .stRadio label {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    color: #888 !important;
    text-transform: uppercase !important;
}

.stSidebar {
    background: rgba(15, 15, 26, 0.95) !important;
    border-right: 1px solid rgba(247, 147, 26, 0.15) !important;
}

.upload-area {
    border: 2px dashed rgba(247, 147, 26, 0.4);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    background: rgba(247, 147, 26, 0.03);
    margin-bottom: 24px;
}

.error-box {
    background: rgba(220, 50, 50, 0.1);
    border: 1px solid rgba(220, 50, 50, 0.4);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #ff6b6b;
}

.info-box {
    background: rgba(247, 147, 26, 0.07);
    border: 1px solid rgba(247, 147, 26, 0.2);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #f7931a;
}
</style>
""", unsafe_allow_html=True)



# --- Define Pages ---
forecast_page = st.Page("pages/Forecast.py", title="Forecast")
stats_page = st.Page("pages/Statistics.py", title="Statistics")

# --- Navigation ---
pg = st.navigation([forecast_page, stats_page])
pg.run()