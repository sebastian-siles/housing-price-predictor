# California Housing Price Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zrknQ1aawsJKXWb_00y6RR2SrZCTE-j5)

A supervised machine learning project implementing multivariate linear regression and ensemble methods to forecast median house values in California. Developed as a core asset for academic portfolio transfer.

## Project Overview
This project transitions from deterministic logic into probabilistic machine learning. Utilizing Scikit-Learn's California Housing dataset (20,640 records), the pipeline analyzes economic and geographic features to map non-linear correlations and predict housing costs.

## Key Architecture & Features
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and computed correlation matrices, establishing Median Income (`MedInc`) as the primary linear driver ($r \approx 0.69$).
* **Feature Engineering:** Developed synthetic domain-specific attributes, including `RoomsPerHousehold` and `BedroomsPerRoom`, to capture spatial density.
* **Data Preprocessing:** Implemented Z-score normalization (`StandardScaler`) on an 80/20 train-test split to optimize Gradient Descent convergence and eliminate data leakage.

## 📈 Model Performance & Evaluation
We established a statistical baseline using Linear Regression and subsequently optimized performance using an ensemble Random Forest framework:

| Model | R-Squared ($R^2$) | RMSE (USD) | Performance Shift |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.5942 | \$72,920 | Baseline Model |
| **Random Forest Regressor** | **0.8049** | **\$50,566** | **+35.45% Accuracy Gain** |

### Insights
The **Random Forest Regressor** drastically minimized error by capturing complex non-linear geographic interactions (Latitude/Longitude boundaries) and economic ceilings that traditional linear formulas fail to map.

## Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebooks
