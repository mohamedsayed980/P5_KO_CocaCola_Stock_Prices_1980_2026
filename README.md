# 📈 KO CocaCola Stock Analysis — End-to-End Data Analysis & ML
> **Author:** Mohamed · M3 · Data Analysis Portfolio  
> **Dataset:** KO CocaCola Stock Prices 1980–2026  
> **Stack:** Python · Streamlit · Scikit-learn · Statsmodels · Pandas · Matplotlib

---

## 📌 Project Overview

A full end-to-end Time Series Analysis and Machine Learning project on **46 years of KO CocaCola daily stock data** — from $0.17 in 1980 to $73.55 in 2026, delivering a cumulative return of **39,377%**.

This project demonstrates professional financial data analysis using technical indicators (RSI, MACD, Bollinger Bands), time series decomposition, stationarity testing, and ML-based price direction prediction.

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| Ticker | KO (The Coca-Cola Company) |
| Exchange | NYSE |
| Period | 1980-01-02 → 2026-01-30 |
| Trading Days | 11,613 |
| Columns | 27 (OHLCV + technical indicators) |
| Price Range | $0.1651 → $73.55 |
| Cumulative Return | 39,377% |

### Pre-built Technical Indicators
| Indicator | Description |
|-----------|-------------|
| MA_20/50/200 | Moving averages — trend direction |
| EMA_12/26 | Exponential moving averages |
| MACD + Signal | Momentum crossover signals |
| BB_Upper/Lower/Width | Bollinger Bands — volatility |
| RSI_14 | Relative Strength Index (0–100) |
| Daily_Return_Pct | Daily percentage change |
| Cumulative_Return_Pct | Total return since 1980 |

### Dataset Prep (Jupyter)
```python
df = pd.read_csv("KO_CocaCola_Stock_Prices_1980_2026.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["price_up"]   = (df["Close"] > df["Close"].shift(1)).astype(int)
df["next_close"] = df["Close"].shift(-1)
df_clean = df.dropna(subset=["next_close","Daily_Return_Pct"]).copy()
df_clean.to_csv("ko_stock_clean.csv", index=False)
```

---

## 🏗️ Architecture

```
📁 Repo_5_CocaCola_Stock/
├── Home.py                    ← Multipage launcher
├── M3_logo.png
├── requirements.txt
├── README.md
├── pages/
│   ├── EDA_dashboard.py       ← Stage 1 (Tabs 1–13)
│   └── ML_Models.py           ← Stage 2 (Tabs 9–13)
└── data/
    └── ko_stock_clean.csv
```

**Run:** `streamlit run Home.py`

---

## 📊 Stage 1 — EDA Dashboard (13 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 1 | Data Overview & Correlation | Shape, stats, correlations |
| 2 | Variables Analysis | Distributions, histograms |
| 3 | IQR Cleaning | Outlier detection |
| 4 | Outliers Lab | Z-score, box plots |
| 5 | Dashboard Summary | Visual summary |
| 6 | Missing Values | Heatmap, strategy |
| 7 | Multicollinearity | VIF analysis |
| 8 | Insights | Auto-generated insights |
| 9 | **Stock KPI Dashboard** ⭐ | Full price history · annual avg · KPI table |
| 10 | **Price & Returns Analysis** ⭐ | Daily returns · cumulative return · annual table |
| 11 | **Technical Indicators** ⭐ | MA · RSI · MACD · Bollinger Bands |
| 12 | **Time Series Decomposition** ⭐ | Trend + Seasonality + Residuals + Volatility |
| 13 | **Statistical Tests + ADF** ⭐ | Stationarity · T-Test · ANOVA by quarter |

---

## 🤖 Stage 2 — ML Models (5 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 9 | Regression Models (6) | Predict **next day Close price** |
| 10 | Classification Models (6) | Predict **price direction** (Up/Down) |
| 11 | Comparison & Report | Model comparison |
| 12 | Predict Next Trading Day | Input features → prediction |
| 13 | Final Insights & Report | PDF + Word export |

### Models
**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · KNN · Decision Tree · Random Forest · Gradient Boosting · SVM

### ML Notes
- **price_up** is perfectly balanced (50/50) → no `class_weight` needed
- **next_close** regression will show high R² due to autocorrelation — this is expected for stock data
- Drop `Date` and `Day_of_Week` from features automatically

---

## 🔑 Key Insights

- **39,377% cumulative return** over 46 years — $1 invested in 1980 = $394 today
- **Price is non-stationary** (ADF p > 0.05) — expected for stock prices
- **Daily returns ARE stationary** (ADF p < 0.05) — good for ML features
- **RSI > 70** (overbought) occurred frequently in 2020–2026 bull run
- **MACD crossovers** accurately signal trend reversals across all decades
- **Volatility spikes** visible in 1987 crash, 2008 crisis, 2020 COVID

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/Repo_5_CocaCola_Stock.git
cd Repo_5_CocaCola_Stock
pip install -r requirements.txt
streamlit run Home.py
```

## 📥 Dataset Setup
1. Download KO stock data from Kaggle or Yahoo Finance
2. Run Jupyter prep → produces `ko_stock_clean.csv`
3. Upload via app file uploader

---

## 🗺️ Portfolio Roadmap

| # | Project | Domain | Status |
|---|---------|--------|--------|
| P1 | House Prices | Real Estate | ✅ Complete |
| P2 | Olist E-Commerce | Retail / BI | ✅ Complete |
| P3 | HR Attrition | Human Resources | ✅ Complete |
| P4 | Superstore Sales | Business Intel | ✅ Complete |
| P5 | KO CocaCola Stock | Finance / Time Series | ✅ Complete |
| P6–P7 | Coming Soon... | Environment · Fraud | 🔜 |

---

## 👤 Author
**Mohamed · M3** — Mechanical Engineer → Data Analyst  
📊 Building a professional DA/DS portfolio — one dataset at a time.

*Built with ❤️ using Python & Streamlit*
