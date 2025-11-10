# Ensemble Learning for Bike Share Demand Prediction

A comprehensive implementation and comparison of ensemble learning techniques for predicting hourly bike rental demand using the Bike Sharing Dataset.

## 📋 Project Overview

This project demonstrates the application of three primary ensemble learning techniques—*Bagging, **Boosting, and **Stacking*—to solve a complex time-series regression problem. The goal is to accurately forecast hourly bike rental counts by leveraging multiple ensemble methods to address model variance and bias.

*Student:* Venkata Sai Vishwesvar SV  
*Roll Number:* BE22B042  
*Course:* DA5401 - Data Analytics

## 🎯 Business Context

As a data scientist for a city's bike-sharing program, accurate demand forecasting is critical for:
- *Inventory Management*: Ensuring bikes are available where and when needed
- *Logistics Planning*: Optimizing bike redistribution strategies  
- *Maintenance Scheduling*: Planning service and repair schedules
- *User Satisfaction*: Improving overall user experience

## 🔬 Methodology

### Ensemble Techniques Implemented

1. *Bagging (Bootstrap Aggregating)*
   - *Purpose*: Reduce model variance
   - *Approach*: Train multiple decision trees on bootstrap samples and average predictions
   - *Implementation*: BaggingRegressor with 60 DecisionTreeRegressors

2. *Boosting (Gradient Boosting)*
   - *Purpose*: Reduce model bias
   - *Approach*: Sequential model building where each model corrects predecessor errors
   - *Implementation*: GradientBoostingRegressor with optimized hyperparameters

3. *Stacking (Stacked Generalization)*
   - *Purpose*: Optimal combination of diverse models
   - *Approach*: Two-level architecture with base learners and meta-learner
   - *Implementation*: StackingRegressor with KNN, Bagging, and GBR as base learners

## 📊 Dataset Information

- *Source*: [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- *Citation*: Fanaee-T, Hadi, and Gamper, H. (2014). Bikeshare Data Set. UCI Machine Learning Repository
- *Features*: Weather conditions, temporal patterns, seasonal factors
- *Target*: Total hourly bike rental count (cnt)
- *Size*: 17,379 hourly records

## 🚀 Key Features

### Data Preprocessing
- *Feature Engineering*: One-hot encoding for categorical variables
- *Data Cleaning*: Removal of irrelevant columns and prevention of data leakage
- *Train/Test Split*: 80/20 split with stratified sampling

### Advanced Analysis
- *Exploratory Data Analysis (EDA)*: Weather impact, temperature-humidity interactions, temporal patterns
- *Feature Engineering*: Interaction features based on EDA insights
- *Advanced Models*: XGBoost implementation with engineered features
- *Model Validation*: Comprehensive residual analysis

## 📈 Results Summary

| Model | RMSE | Performance Improvement |
|-------|------|------------------------|
| Linear Regression (Baseline) | 103.30 | - |
| Decision Tree (Baseline) | 122.03 | - |
| Bagging Regressor | 101.39 | 1.8% improvement |
| Gradient Boosting | 56.46 | 45.3% improvement |
| Stacking Regressor | 54.27 | *47.5% improvement* |
| XGBoost (Enhanced) | 47.65 | *53.9% improvement* |

### Key Insights
- *Best Performer*: XGBoost with engineered features (RMSE: 47.65)
- *Ensemble Winner*: Stacking Regressor achieved the best performance among standard ensemble methods
- *Primary Challenge*: Model bias rather than variance (boosting methods excelled)
- *Critical Features*: Hour of day, temperature, working day status, and their interactions

## 🛠 Installation & Setup

### Prerequisites
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost


### Required Libraries
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


## 📁 Project Structure
```

├── be22b042.ipynb          # Main Jupyter notebook with complete analysis
├── be22b042.md             # Detailed markdown report
├── hour.csv                # Bike sharing dataset
├── README.md               # This file
└── be22b042_files/         # Generated plots and visualizations
    ├── be22b042_21_3.png   # Model comparison chart
    ├── be22b042_23_0.png   # Weather impact analysis
    ├── be22b042_25_0.png   # Temperature-humidity scatter plot
    ├── be22b042_27_0.png   # Monthly trends
    ├── be22b042_29_1.png   # Hourly patterns (working vs non-working days)
    ├── be22b042_35_1.png   # Feature importance analysis
    └── be22b042_37_0.png   # Residual analysis plots
```

## 🔧 Usage

1. *Clone the repository*
   bash
   git clone <repository-url>
   cd bike-share-ensemble-learning
   

2. *Run the analysis*
   bash
   jupyter notebook be22b042.ipynb
   

3. *Execute specific sections*
   - Part A: Data Preprocessing and Baseline
   - Part B: Bagging Implementation
   - Part C: Boosting Implementation  
   - Part D: Stacking Implementation
   - Part E: Model Comparison
   - Part F: Advanced EDA and Feature Engineering

## 📋 Analysis Workflow

1. *Data Loading & Preprocessing*
   - Load hourly bike sharing data
   - Clean and prepare features
   - Handle categorical variables with one-hot encoding

2. *Baseline Model Development*
   - Implement Linear Regression and Decision Tree
   - Establish performance benchmarks

3. *Ensemble Implementation*
   - Bagging: Bootstrap aggregating with decision trees
   - Boosting: Gradient boosting for bias reduction
   - Stacking: Multi-level ensemble architecture

4. *Advanced Modeling*
   - Feature engineering based on EDA insights
   - XGBoost implementation with hyperparameter tuning
   - Model validation and residual analysis

## 🎨 Visualizations

The project includes comprehensive visualizations:
- *Model Performance Comparison*: RMSE comparison across all models
- *Weather Impact Analysis*: Boxplots showing rental patterns by weather conditions
- *Temperature-Humidity Interactions*: Scatter plots with color-coded humidity levels
- *Temporal Patterns*: Monthly trends and hourly patterns for working vs. non-working days
- *Feature Importance*: Top contributing features from XGBoost model
- *Residual Analysis*: Comprehensive diagnostic plots for model validation

## 🔍 Key Findings

1. *Ensemble Superiority*: All ensemble methods outperformed single baseline models
2. *Boosting Excellence*: Gradient-based methods showed the most significant improvements
3. *Feature Interactions*: Engineered interaction features substantially improved model performance
4. *Temporal Importance*: Hour of day and working day status are the most predictive features
5. *Weather Impact*: Clear negative correlation between weather severity and bike rentals

## 🚦 Model Performance Insights

- *Bagging*: Modest improvement through variance reduction
- *Boosting*: Substantial improvement through bias reduction  
- *Stacking*: Best ensemble performance by combining diverse model strengths
- *XGBoost*: Ultimate performance with advanced boosting and feature engineering

## 🤝 Contributing

This is an academic project. For questions or discussions about the methodology:
1. Review the detailed analysis in [be22b042.md](be22b042.md)
2. Examine the implementation in [be22b042.ipynb](be22b042.ipynb)
3. Check the comprehensive visualizations in the be22b042_files/ directory

## 📄 License

This project is for educational purposes as part of the DA5401 Data Analytics course.

## 🙏 Acknowledgments

- *Dataset*: UCI Machine Learning Repository
- *Original Authors*: Fanaee-T, Hadi, and Gamper, H. (2014)
- *Course*: DA5401 Data Analytics
- *Institution*: Academic institution providing the data analytics curriculum

---

*Note*: This analysis demonstrates the power of ensemble learning for complex regression problems, showing how combining multiple models can achieve significantly better performance than individual approaches.
