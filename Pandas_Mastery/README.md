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

## 📝 Project Summary (For Recruiters)

This project demonstrates my ability to take a **real-world messy dataset** and transform it into a **machine-learning ready asset**.  

Key highlights:  
- Designed a **robust preprocessing pipeline** covering null handling, duplicate removal, label encoding, and scaling.  
- Applied **feature engineering** to extract useful attributes like `start_year`, `end_year`, and normalized votes.  
- Incorporated **mathematical rigor** by implementing rating-weighting and log normalization for skewed features.  
- Produced a final dataset that is **ready for predictive modeling** — e.g., predicting ratings, votes, or gross revenue.  

This project reflects **data engineering, cleaning, and applied ML readiness skills**, which are essential for both research and production environments.  
It shows not only my ability to clean data but also to think ahead toward how features will impact downstream **ML model performance**.

---
