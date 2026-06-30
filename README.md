# Predicting Airbnb Listing Prices in Brisbane, Australia

Predicting the nightly price (AUD) of Airbnb listings in Brisbane using listing-level features such as property type, location, host characteristics, availability, and review scores.

**Kaggle Result:** RMSE 0.056 — Rank 12th

## Overview

The goal of this project is to build a regression model that accurately predicts Airbnb nightly prices from listing attributes, helping hosts, investors, and platforms make data-driven pricing decisions.

## Project Structure

```
.
├── Airbnb_Brisbane.ipynb       # Main notebook (EDA, cleaning, modelling)
├── hybrid.csv                  # Final Kaggle submission
└── README.md
```

## Methodology

**Task 1 — Exploratory Data Analysis**
- Defined the prediction problem and evaluation metric (RMSE)
- Identified and interpreted missing values across 65 features
- Conducted univariate and bivariate analysis on price, room type, location, and review scores
- Selected 30 features for modelling, justified by correlation and domain reasoning

**Task 2 — Data Cleaning and Feature Engineering**
- Cleaned price and percentage fields, removing non-numeric characters
- Engineered 6 new features (amenities count, bathroom type, beds-per-guest, availability flag, review completeness)
- Imputed missing values (median for numeric, mode for categorical)
- Encoded categorical variables using one-hot, ordinal, and binary encoding
- Log-transformed skewed features and the target variable; removed price outliers above the 99th percentile

**Task 3 — Model Fitting and Tuning**
- Trained and tuned three models: Ridge Regression, Random Forest, and XGBoost using 5-fold cross-validation
- XGBoost achieved the best baseline performance (CV log-RMSE: 0.1955)
- Improved performance through a stacking ensemble (XGBoost + Random Forest + Ridge meta-learner)
- Further improved using a hybrid prediction strategy: for listings with recorded occupancy data, `estimated_revenue_l365d / estimated_occupancy_l365d` was found to equal the actual price exactly, and used directly; the stacked ensemble was used for all remaining listings

## Final Results

| Model | CV Log-RMSE | Kaggle RMSE |
|---|---|---|
| XGBoost (baseline) | 0.1955 | 0.124 |
| Stacked Ensemble | 0.1943 | 0.124 |
| XGBoost v3 + Stacking | 0.1869 | 0.123 |
| **Hybrid (Revenue + Stacking)** | – | **0.056** |

## Tools Used

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib

