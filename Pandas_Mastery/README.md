# 🎬 Movies Dataset: Cleaning, Feature Engineering & Preprocessing

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
-> RangeIndex: 9999 entries
-> Columns: 9 (Movies, Year, Genre, Rating, One-line, Stars, Votes, Runtime, Gross)
-> Memory: 703.2 KB

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

## 🧮 Feature Engineering Equations

1. **Year Parsing**

-> start_year = min(YEAR)
-> end_year = max(YEAR)

Example: `(2010–2022)` → `start_year=2010`, `end_year=2022`

2. **Votes Normalization**
-> $VOTES_scaled = (VOTES - μ) / σ$

3. **Runtime Imputation**
-> $Runtime_filled = median(Runtime)$

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
- This cleaned dataset can now power **ML pipelines for regression/classification tasks** (e.g., predicting ratings, votes, or gross revenue).

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

## 🚀 Key Takeaways

- Strong **data wrangling & feature engineering** workflow.  
- Dealt with **real-world messy text columns** (YEAR ranges, comma-separated genres, stars).  
- Produced a **scalable preprocessing pipeline** ready for ML training.  
- Showcases ability to go from **raw unstructured Kaggle data → clean, usable dataset**.

---
