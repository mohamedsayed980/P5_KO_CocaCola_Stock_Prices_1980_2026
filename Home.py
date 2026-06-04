# ─────────────────────────────────────────────
#  Repo_5 —P5_KO_CocaCola_Stock_Prices_1980_2026
#  Home.py  |  Multipage Launcher
#  Author : Mohamed · M3
# ─────────────────────────────────────────────
# =============================================================================
## path = streamlit run "E:\FINAL PROJECTS\P5_KO_CocaCola_Stock_Prices_1980_2026\Home.py"
# ================================================================#
import streamlit as st
import pathlib

st.set_page_config(
    page_title  = "KO CocaCola Stock Analysis 1980–2026 · M3",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

CLR = {
    "primary"   : "#1565c0",
    "success"   : "#2e7d32",
    "warning"   : "#e65100",
    "danger"    : "#c62828",
    "teal"      : "#00695c",
    "secondary" : "#455a64",
    "light"     : "#e3f2fd",
    "dark"      : "#1a237e",
    "purple"    : "#6a1b9a",
    "amber"     : "#f57f17",
    "pink"      : "#ad1457",
    "grey"      : "#546e7a",
    "white"     : "#ffffff",
    "black"     : "#212121",
    "teal2"     : "#00695c",
    "green2"    : "#1b5e20",
}

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 📊  KO CocaCola Stock Analysis 1980–2026")
    st.markdown("**Author:** Mohamed · M3")
    st.markdown("---")
    st.markdown("#### 📂 Navigation")
    st.markdown("""
- **🏠 Home** ← You are here
- **📊 EDA Dashboard** → Stage 1
- **🤖 ML Models** → Stage 2
""")
    st.markdown("---")
    st.markdown("#### 📁 Dataset Info")
    st.markdown("""
- **Source:** Tableau KO_CocaCola_Stock_Prices_1980_2026 (Kaggle)
- **Trading Days:** 11613 Days
- **Close range:** 0.1651 → 73.55
- **Price Up %:** 76.16
- **Cumulative Return:** 39377.57


""")
    st.markdown("---")
    st.caption("© M3 · Data Analysis Portfolio")

# ── Header
st.markdown(f"""
<div style='background: linear-gradient(135deg, {CLR["green2"]} 0%, {CLR["teal2"]} 60%, {CLR["primary"]} 100%);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.4rem;'>
        📊  KO_CocaCola_Stock_Prices_1980_2026 Data Analysis
    </h1>
    <p style='color: #b2dfdb; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        End-to-End Business Intelligence & Machine Learning · Tableau KO_CocaCola_Stock_Prices_1980_2026 Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards
col1, col2, col3, col4 = st.columns(4)

for col_w, label, val, color in zip(
    [col1, col2, col3, col4],
    [" Trading Days", " Close range ", "Price Up %", " Cumulative Return"  ],
    ["11,613 " , " 0.1651→ 73.55 ",  " 76.16", "39377.57"],
    
    [CLR["primary"], CLR["success"], CLR["teal2"], CLR["amber"] ]
    
):
    col_w.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {color}; text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,.06);'>
        <h2 style='color:{color}; margin:0;'>{val}</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>{label}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Stages
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["primary"]}; margin-top:0;'>📊 Stage 1 — EDA Dashboard</h3>
        <p style='color:{CLR["secondary"]};'>Deep business intelligence across 12 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Data Overview & Correlation</li>
            <li>Variables Analysis & Distributions</li>
            <li>IQR Cleaning & Outlier Detection</li>
            <li>Missing Values & Multicollinearity</li>
            <li>Insights & Recommendations</li>
            <li>📈 Stock KPI Dashboard <b>(NEW)</b></li>
            <li>💹 Price & Returns Analysis <b>(NEW)</b></li>
            <li>📉 Technical Indicators — MA, RSI, MACD, BB <b>(NEW)</b></li>
            <li>🔄 Time Series Decomposition <b>(NEW)</b></li>
            <li>🧪 Statistical Tests + ADF Stationarity</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["success"]}; margin-top:0;'>🤖 Stage 2 — ML Models</h3>
        <p style='color:{CLR["secondary"]};'>Full machine learning pipeline across 5 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
#------------------------------------------------------------------------------        
            <li>Regression Models (6) → predict <b>next Close price</b></li>
            <li>Classification Models (6) → predict <b>price direction</b> (Up/Down)</li>
            <li>Model Comparison & Report</li>
            <li>Predict Next Trading Day</li>
            <li>Final Insights & Report (PDF + Word)</li>
        </ul>
        <br>
        <p style='color:{CLR["secondary"]}; font-size:0.85rem;'>
            ✅ 12 models · Parallel training · 50/50 balanced data · AUC evaluation
#------------------------------------------------------------------------------
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Business Context
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;
            border-left:5px solid {CLR["teal2"]}; margin-bottom:1rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🎯 Why This Project Matters</h4>
    <p style='color:{CLR["black"]}; margin:0; line-height:1.8;'>
        KO CocaCola (NYSE: KO) has delivered <b>39,377% cumulative return</b> since 1980 —
        one of the greatest long-term wealth creators in history. This project analyzes
        <b>46 years of daily stock data</b> using technical indicators (RSI, MACD, Bollinger Bands)
        to identify trends, volatility patterns, and predict future price direction.
        A perfect demonstration of Time Series Analysis applied to real financial data.
    </p>
</div>
""", unsafe_allow_html=True)

# ── How to Use
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🚀 How to Use</h4>
    <ol style='color:{CLR["black"]}; line-height:2;'>
        <li>Go to <b>📊 EDA Dashboard</b> → upload <code>KO_CocaCola_Stock_Prices_1980_2026_clean.csv</code></li>
        <li>Explore all 12 EDA tabs — insights auto-saved to session</li>
        <li>Go to <b>🤖 ML Models</b> → data flows automatically from Stage 1</li>
        <li>Train models, compare results, export the final report</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align:center; color:{CLR["grey"]}; font-size:0.8rem; padding:1rem;'>
    Built with ❤️ by Mohamed · M3 · Data Analysis Portfolio · Tableau KO_CocaCola_Stock_Prices_1980_2026 Dataset
</div>
""", unsafe_allow_html=True)
