# King County House Price Prediction

A comprehensive machine learning project predicting residential property prices in King County (2014-2015).
Here is the link to the dataset: https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa?resource=download 

## 📊 Project Overview

This project builds and compares multiple regression models to predict house prices based on property characteristics:
- **Dataset:** 21,613 house sales (King County, WA)
- **Time Period:** 2014-2015
- **Target Variable:** Sale Price
- **Features:** 41 (20 original + 21 engineered)
- **Best Model:** XGBoost (regularized) – Test R² ≈ 0.878, RMSE ≈ $135,693 (≈2.47% of price range)

This model is intended as a decision-support tool for real-estate analysts and potential home buyers/sellers in King County. It helps estimate a reasonable price range given the property’s characteristics, and highlights which features most strongly drive value (e.g. grade, living area, waterfront). It is not a replacement for professional appraisal, but a complementary analytical tool.

## 🎯 Key Findings

### Model Performance Comparison

| Model                          | Type             | R² Train | R² Test | Test RMSE  | RMSE % of range |
|--------------------------------|------------------|---------:|--------:|-----------:|----------------:|
| **XGBoost (regularized)**      | Boosting         | 0.9387   | **0.8782** | **$135,693** | **≈2.47%** |
| Gradient Boosting (baseline)   | Boosting         | 0.915    | 0.864   | $143,483  | 2.61% |
| Gradient Boosting (GridSearch) | Boosting         | 0.9725   | 0.8551  | $147,999  | 2.70% |
| Random Forest (tuned)          | Tree ensemble    | 0.865    | 0.820   | $164,841  | 3.80% |
| KNN Regression                 | Instance-based   | 0.848    | 0.750   | $194,536  | 3.55% |
| AdaBoost                       | Boosting         | 0.657    | 0.611   | $242,463  | 4.42% |
| Ridge Regression (opt)         | Linear           | 0.714    | 0.713   | $208,357  | 3.80% |
| Lasso Regression (opt)         | Linear           | 0.713    | 0.711   | $209,145  | 3.81% |
| Linear Regression (baseline)   | Linear           | 0.714    | 0.713   | $208,349  | 3.77% |


### Top 10 Most Important Features

1. `grade` - House quality (27.7%)
2. `sqft_living` - Living area (13.7%)
3. `waterfront` - Waterfront presence (9.8%)
4. `total_sqft` - Living area including basement (9.7%)
5. `lat` - Latitude (5.3%)
6. `sqft_living15` - Neighbors' living area area (4.5%)
7. `long` - Longitude (3.4%)
8. `view` - View quality (3.3%)
9. `house_age` - Age at sale (2.9%)
10.`bathrooms` - Number of bathrooms (2.0%)


## 📁 Project Structure

```
kingCountryHouses/
├── king_county_house_price_prediction.ipynb    # Main analysis notebook
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
├── KC_House_Price_Prediction.pdf               # Presentation slides summarizing findings
├── models/
│   └── best_model.pkl                          # Serialized best model: regularized XGBoost model
├── data/
│   └── king-country-houses-aa.csv              # Link to dataset provided within this file
└── outputs/
    ├── model_comparison                        # Performance metrics: part of the notebook
    └── feature_importance                      # Top features: part of the notebook
```

## 🔍 Methodology

### 1. Exploratory Data Analysis
- Statistical information on 21 features
- Assessment of data quality (missing values, duplicates, strings, zeros)
- Distribution analysis of 21 features
- Spearman correlation with price and between all the features
- Outlier detection (IQR method)
- Location and temporal patterns

### 2. Feature Engineering
- **Living-area:** total_sqft, living_to_lot_ratio, bath_per_bed, basement_share
- **Age/Renovation:** house_age, since_renovation, was_renovated, renovation_period
- **Density:** lot_per_living
- **Log transforms:** log_price, log_sqft_living, log_sqft_lot

### 3. Model Development & Comparison
- **Linear Models:** Linear Regression, Ridge (α=10), Lasso (α=100)
- **Tree Models:** Random Forest, Gradient Boosting, XGBoost
- **Hyperparameter Tuning:** GridSearchCV for optimal parameters
- **Cross-validation:** 5-fold CV, 80/20 train-test split

### 4. Regularization & Overfitting Control
- Ridge L2: Shrinks coefficients toward zero
- Lasso L1: Performs feature selection (shrinks to exactly zero)
- Tree constraints: max_depth, min_samples_split, subsample

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Make a Prediction
In the notebook we show how to use the trained model for prediction.

## 📈 Results & Insights

**Best Model:** XGB boosting (regularized)
- **Test R²:** 0.878 (explains 87.8% of variance)
- **Test RMSE:** $135,693 (2.47% of price range)
- **Overfitting Gap:** 6.05% (good generalization)

### Why XGB Boosting Won:
1. **Non-linear relationships:** Captures complex price patterns better than linear models
2. **Low overfitting:** Train-test R² gap of only 6.05% (vs 8.1% for Random Forest)
3. **Balanced performance:** Good accuracy without sacrificing generalization
4. **Interpretability:** Feature importances show which factors drive prices

### Model Ranking:
1. XGBoost (regularized) – Test R² ≈ 0.878, RMSE ≈  $135,693 (2.47% of price range) ✓ - Best balance
2. XGBoost (baseline, R² 0.874) - Slightly lower R2, slightly more overfitting
3. Gradient Boosting (R² 0.864)  - Slightly lower R2, but slightly less overfitting
3. Random Forest (R² 0.842) - Good but overfits more
4. Linear Models (R² ~0.71) - Limited by linear assumption, no overfitting

## 🔑 Key Discoveries

1. **Grade is King:** House quality is the #1 predictor (27.8% importance)
2. **Location Matters:** Latitude and longitude account for 8.7% of importance
3. **Living Space Drives Price:** sqft_living is 2nd most important (13.7%)
4. **Age & Condition:** Newer houses, number of bathrooms, view and waterfront increase value
5. **Linear Models Fail:** Price has strong non-linear relationships with features

## Limitations/ Future Work:
- The model uses sale data only from 2014–2015 and may not generalize to other market conditions.
- Calendar features (year/month/day of sale) are derived from historical sale dates, which are known in hindsight; in a real deployment, a similar model would use listing date instead.
- No explicit handling of inflation or macro-economic variables.
- Future work: add spatial encodings (distance to downtown, schools), try CatBoost/LightGBM, or add uncertainty estimates (prediction intervals).

## 🛠️ Technologies

- **Python 3.8+** with Pandas, NumPy, Scikit-learn, XGBoost
- **Hyperparameter Tuning:** GridSearchCV with 5-fold CV
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** Kaggle (King County House Sales)

## 📊 Visualizations Included

- Price distribution (right-skewed, mean $540K)
- Features' distribution
- Scatter plots for correlation of each feature with price
- Feature correlation heatmap
- Box plots by grade, condition, view
- Scatter plots with trend lines
- Feature importance bar charts
- Residual analysis (mean ≈ 0, normally distributed)
- Actual vs predicted scatter plot

## 📝 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
kagglehub
```

## 📖 Educational Value

This project demonstrates:
- Complete ML pipeline (EDA → FE → Modeling → Evaluation)
- Feature engineering best practices
- Hyperparameter tuning with GridSearchCV
- Model comparison framework
- Regularization techniques (Ridge/Lasso)
- Overfitting detection and control
- Python best practices



## 👤 Authors

Julia Parnis, Helena Gomez, Miquel de Tolledo, Gail Marechal- Data Science & ML Students

---

**Last Updated:** December 2025
**Best Model:** XGB boosting (regularized)
**Dataset:** King County House Sales (21,613 rows, 2014-2015)
