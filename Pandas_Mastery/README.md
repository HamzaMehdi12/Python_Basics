# 🎬 Movies Dataset: Cleaning, Feature Engineering & Preprocessing

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.6.2-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project demonstrates **end-to-end data preprocessing and feature engineering** using the [Movies Dataset](https://www.kaggle.com/datasets/bharatnatrayn/movies-dataset-for-feature-extracion-prediction).  
It highlights advanced **data wrangling, encoding, imputing, and scaling techniques**, preparing the dataset for downstream ML models such as **regression and classification**.

---

## 📂 Dataset Overview

**Raw Dataset Snapshot**

| MOVIES                        | YEAR      | GENRE                         | RATING | ONE-LINE | STARS | VOTES   | RunTime | Gross   |
|--------------------------------|-----------|-------------------------------|--------|----------|-------|---------|---------|---------|
| Blood Red Sky                  | (2021)    | Action, Horror, Thriller      | 6.1    | A woman with a mysterious illness... | Peri Baumeister, Carl Anton Koch | 21,062 | 121.0   | NaN |
| Masters of the Universe: Revelation | (2021– ) | Animation, Action, Adventure | 5.0    | The war for Eternia begins again... | Chris Wood, Mark Hamill | 17,870 | 25.0    | NaN |
| The Walking Dead               | (2010–22) | Drama, Horror, Thriller       | 8.2    | Sheriff Rick wakes up from a coma... | Andrew Lincoln, Norman Reedus | 885,805 | 44.0    | NaN |

**Dataset Info**
1. $RangeIndex: 9999 entries$
2. $Columns: 9 (Movies, Year, Genre, Rating, One-line, Stars, Votes, Runtime, Gross)$
3. $Memory: 703.2 KB$


---

## ⚙️ Preprocessing Pipeline

| Step | Description | Action Taken |
|------|-------------|--------------|
| **1. Missing Values** | Identify nulls across numeric + categorical columns | Filled NaNs with statistical imputing (mean/median for numeric, mode for categorical) |
| **2. Duplicate Handling** | Detect repeated rows | Dropped all duplicates |
| **3. Feature Engineering** | Extracted structured columns from messy text (e.g., YEAR → `start_year`, `end_year`) | Created new numeric features for ML readiness |
| **4. Label Encoding** | Converted categorical (Genre, Stars) into numeric | Applied sklearn `LabelEncoder` |
| **5. Imputation** | Dealt with residual missing values post-feature extraction | Median/most-frequent strategy |
| **6. Scaling** | Standardized dataset for ML models | `StandardScaler` applied |

---

## 🧮 Feature Engineering & Equations

### 1. Year Parsing
$start\_year = \min(YEAR), \quad end\_year = \max(YEAR)$

Example: `(2010–2022)` → `start_year=2010`, `end_year=2022`.

---

### 2. Votes Normalization (Z-score Scaling)
$VOTES_{scaled} = \frac{VOTES - \mu}{\sigma}$

Where:  
- \( \mu \) = mean of votes  
- \( \sigma \) = standard deviation of votes  

---

### 3. Weighted Rating (IMDb Formula)
To avoid bias toward low-vote movies:
$WR = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$

Where:  
- \( R \) = average rating for the movie  
- \( v \) = number of votes for the movie  
- \( m \) = minimum votes required to be considered  
- \( C \) = mean rating across the dataset  

This allows better **popularity-adjusted rating prediction**.

---

### 4. Gross Revenue Normalization
Since only ~5% of entries have revenue data:
$Gross_{log} = \log(Gross + 1)$

Helps handle skewed revenue distribution.

---

## ✅ Final Processed Dataset

**Processed Dataset Snapshot**

| start_year | end_year | MOVIES   | GENRE   | RATING  | VOTES   | RunTime  | Gross   |
|------------|----------|----------|---------|---------|---------|----------|---------|
| 2021-01-01 | 2021-01-01 | -1.949199 | -0.801234 |  0.071597 |  0.125160 |  1.392645 | -0.08712 |
| 2010-01-01 | 2022-01-01 | -1.948287 | -0.784055 |  0.851440 | 13.431942 | -0.127242 | -0.08712 |
| 2013-01-01 | 2013-01-01 | -1.947832 | -0.775465 |  1.222795 |  6.184808 | -0.541756 | -0.08712 |

---

## 📊 Results & Insights

- **Raw → Cleaned dataset** reduced noise, dropped duplicates, and imputed 20%+ missing values.  
- **Label encoding & scaling** made the dataset **ML-ready**.  
- **Feature extraction** from textual columns (YEAR, GENRE, STARS) gave structured numeric data.  
- **Weighted rating equation** shows potential for predicting more reliable popularity scores.  
- Final dataset is compact, clean, and suitable for **ML regression/classification tasks**.

---

## 🛠️ Functions & Code Inventory

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `check_nulls(df)` | Raw DataFrame | Summary of NaNs | Detects missing values before preprocessing |
| `fill_missing(df)` | DataFrame | Clean DataFrame | Imputes NaNs with median/most-frequent |
| `drop_duplicates(df)` | DataFrame | Deduplicated DataFrame | Removes redundant rows |
| `feature_engineering(df)` | DataFrame | Enhanced DataFrame | Extracts `start_year`, `end_year`, cleans `Votes` & `Runtime` |
| `encode_labels(df)` | DataFrame | Encoded DataFrame | Label encodes categorical variables |
| `scale_features(df)` | DataFrame | Scaled DataFrame | Standardizes numeric features for ML models |
| `save_processed(df, path)` | DataFrame, filepath | Saved `.csv` file | Stores final cleaned dataset |

---

## 📌 Why This Matters

Preprocessing is the foundation of every machine learning and analytics pipeline.  
The steps shown here — handling missing data, encoding categorical variables, feature scaling, and deduplication — directly influence model performance. Without them:

- Models can **overfit** or fail to generalize.  
- Metrics such as accuracy, precision, or R² can be **misleading**.  
- Data bias can creep into predictions, leading to unreliable insights.  

By systematically applying transformations with **Pandas**, this project demonstrates that clean, well-prepared data improves both **interpretability** and **downstream predictive accuracy**.

---

### ✨ Key Preprocessing Equation

To normalize numerical features and reduce skewness, the **Min-Max Scaling** equation was applied:

$$
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Where:  
- \(X\) = original feature value  
- \(X_{min}, X_{max}\) = minimum and maximum values of the feature  
- \(X_{scaled}\) = scaled value in the range [0,1]  

This ensures that all features contribute equally to models, improving **convergence** and **stability** of machine learning algorithms.

---

## 📝 Project Summary

This project demonstrates my ability to take a **real-world messy dataset** and transform it into a **machine-learning ready asset**.  

Key highlights:  
- Designed a **robust preprocessing pipeline** covering null handling, duplicate removal, label encoding, and scaling.  
- Applied **feature engineering** to extract useful attributes like `start_year`, `end_year`, and normalized votes.  
- Incorporated **mathematical rigor** by implementing rating-weighting and log normalization for skewed features.  
- Produced a final dataset that is **ready for predictive modeling** — e.g., predicting ratings, votes, or gross revenue.  

This project reflects **data engineering, cleaning, and applied ML readiness skills**, which are essential for both research and production environments.  
It shows not only my ability to clean data but also to think ahead toward how features will impact downstream **ML model performance**.

---

## Images
Here you can view the original and processed dataset images
### Original

<img width="578" height="779" alt="Original" src="https://github.com/user-attachments/assets/74d70401-b0c8-44b9-a05c-17c28f9904fd" />

### Processed

<img width="522" height="779" alt="Processed" src="https://github.com/user-attachments/assets/68511107-3b00-4be2-88f4-7004110eee64" />

---

## 🚀 Future Work

To take this project further, I plan to:  

- **Modeling**:  
  - Train **linear regression** models to predict `Gross Revenue` based on features (Votes, Runtime, Genre).  
  - Train **logistic regression / classification models** to predict whether a movie’s rating is “above average” (>7.0).  

- **Feature Expansion**:  
  - Use **NLP techniques** (TF-IDF, embeddings) on the `One-line` description to generate semantic features.  
  - Engineer new popularity-based features like **decay-adjusted votes** (recent years weighted more).  

- **Visualization**:  
  - Explore trends in movie genres, runtime, and revenue using time-series plots.  
  - Build interactive dashboards (e.g., Streamlit) for exploratory analysis.  

- **Deployment**:  
  - Package preprocessing as a **reusable Python module**.  
  - Integrate into an **end-to-end ML pipeline** with FastAPI for deployment.  

---

# 📊 Financial Data Analyzer — NVDA (5-min, 1-month) with Python & Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-lightblue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-9cf)

> **Recruiter TL;DR:** End-to-end stock analytics pipeline (data ingest → feature engineering → EDA → dashboard → exports) showcasing **time-series**, **technical indicators**, and **reporting**.

---

## Table of Contents
- [Overview](#overview)
- [Data Window](#data-window)
- [What It Does](#what-it-does)
- [Technical Indicators & Equations](#technical-indicators--equations)
- [Key Results (Values)](#key-results-values)
- [Visualizations & Dashboard](#visualizations--dashboard)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Data Outputs](#data-outputs)
- [Engineering Notes](#engineering-notes)
- [Extend This Project](#extend-this-project)
- [License](#license)
- [Contact](#contact)
- [Summary](#summary)

---

## Overview
**Financial Data Analyzer** downloads **NVIDIA (NVDA)** stock prices using `yfinance` at **5-minute intervals** over **1 month**, engineers standard **technical indicators**, performs **EDA**, and renders an **interactive Streamlit dashboard**. It also exports clean CSVs for reuse.

**Skills demonstrated:** data ingestion, feature engineering, exploratory analysis, plotting, dashboarding, and lightweight reporting.

---

## Data Window
- **Ticker:** `NVDA`
- **Interval:** `5m`
- **Period:** `1mo`
- **Entries (raw download):** **1,638** rows  
- **Time range (UTC):** **2025-07-28 13:30** → **2025-08-25 19:55**

> The pipeline converts the index to timezone-naive, reindexes to hourly for consistency, and forward-fills OHLCV gaps.

---

## What It Does
- 📥 **Automated Ingest:** `yfinance` → OHLCV
- 🧮 **Feature Engineering:** **SMA(7,14)**, **RSI(7,14)**, **ROC(7,14)**, **Bollinger Bands(7,14)**, **Rolling Std(7,14)**
- 🔎 **EDA:** line charts + distributions + correlation heatmap
- 📊 **Dashboard:** Streamlit views with a **Key Metrics** panel
- 💾 **Exports:** `prices.csv`, `indicators.csv`, `summary.csv` (+ raw)

---

## Technical Indicators & Equations
**Simple Moving Average (SMA)**  
\[
\boxed{SMA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}}
\]

**Relative Strength Index (RSI)** (Wilder)  
\[
\boxed{RSI = 100 - \frac{100}{1 + \frac{AvgGain}{AvgLoss}}}
\]

**Rate of Change (ROC)**  
\[
\boxed{ROC_n(t) = \frac{P_t - P_{t-n}}{P_{t-n}} \times 100}
\]

**Bollinger Bands (BB)**  
\[
\boxed{Upper = SMA_n + k\sigma_n \quad;\quad Lower = SMA_n - k\sigma_n}
\]
with \(k = 2\) (default) and \(\sigma_n\) = rolling standard deviation.

**Volatility (Rolling Std)**  
\[
\boxed{\sigma_n(t) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1}\left(P_{t-i} - SMA_n(t)\right)^2}}
\]

---

## Key Results (Values)
> Values below reflect the **exact logs** and the project’s **current code path**.

- **Mean Return (per bar):** **0.0054%**  (`Returns` column = mean of `Close.pct_change()`)
- **Volatility of Returns:** **0.0000** (by construction: `Returns` is constant → std = 0)
- **Sharpe Ratio (guarded):** **0.00** (volatility = 0 ⇒ guarded to 0)
- **Max Close:** **~181.50**
- **Min Close:** **~174.42**
- **Anomaly Count (RSI rule):** **407** rows flagged  
  - Rule used: **RSI\_7 > 50** *or* **RSI\_7 < 15**.

> **Important:** Because `Returns` is filled with a **single mean value** (constant), its **std = 0** and **Sharpe = 0**. See **Engineering Notes** for a 1-line fix to compute **per-bar returns** instead.

---

## Visualizations & Dashboard
- **Prices + SMA(7/14)**
- **RSI(7/14)**
- **Bollinger Bands(7/14)**
- **ROC(7/14)**
- **Rolling Std (Volatility)**
- **Returns Distribution**
- **Correlation Heatmap** across engineered features
- **Key Metrics** panel: **Mean Return**, **Volatility**, **Sharpe**

Launch:
```bash
streamlit run financial_data_analyzer.py
```

## Quick Start
-- **Create environment & install:** python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
-- **Run the analyzer (downloads data, computes features, saves CSVs):** python financial_data_analyzer.py
-- **Launch dashboard"** streamlit run financial_data_analyzer.py

## Project Structure
