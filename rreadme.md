🏠 California Housing Price Prediction
📌 Project Overview

This project focuses on predicting median house values in California districts using regression-based machine learning models.
The objective is to build a robust, interpretable model while carefully handling data skewness, multicollinearity, and target censoring.

📂 Dataset Description

Source: California Housing Dataset

Rows: ~20,640 (before preprocessing)

Features: 8 numerical predictors

Target: median_house_value

Key Features

median_income

housing_median_age

total_rooms

total_bedrooms

population

households

latitude

longitude

🔍 Exploratory Data Analysis (EDA)
Key Findings

No missing values in most features

total_bedrooms contained missing values → mean imputation applied

Target variable is right-skewed

Strong price cap at $500,000, causing artificial clustering

median_income shows the strongest correlation with house price (~0.67)

Several features are highly correlated, indicating multicollinearity

Visual Insights

Histogram revealed skewed distribution and capped values

Scatter plots confirmed strong linear relationship between income and price

Geographic clustering observed around major urban areas

⚙️ Data Preprocessing
1️⃣ Target Censoring

Rows with median_house_value == 500000 were removed

Reason: These values represent a capped ceiling, not true prices

2️⃣ Target Transformation

Log transformation applied:

log(median_house_value)


Reduced skewness

Improved stability for linear models

3️⃣ Missing Value Handling

total_bedrooms filled using mean imputation

4️⃣ Feature Scaling

StandardScaler applied

Required for linear and regularized models

🤖 Models Implemented
🔹 Linear Regression

Used as a baseline

Performance limited by multicollinearity

🔹 Ridge Regression ✅ (Final Model)

Addresses multicollinearity

Provides stable coefficients

Best generalization performance

Configuration:

Alpha: 0.01

Target: log-transformed

Scaled features

Performance:

R² ≈ 0.62

RMSE (log scale) ≈ 0.33

🔹 Random Forest Regression ❌

Tested to capture non-linear relationships

Underperformed due to:

Small dataset size

Predominantly linear relationships

Weak interaction effects

📊 Model Comparison
Model	Target	R²
Linear Regression	log	~0.60
Ridge Regression	log	~0.62
Random Forest	raw	~0.61
🧠 Key Learnings

Model complexity does not guarantee better performance

Regularization is critical when features are correlated

Log transformation improves stability for skewed targets

Tree-based models are not always optimal for small, linear datasets

Proper EDA directly informs model choice

✅ Final Conclusion

Ridge Regression with a log-transformed target provided the most stable and interpretable results for California housing price prediction, outperforming both basic linear regression and tree-based models.

🛠 Tech Stack

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

🚀 Future Improvements

RidgeCV for optimal alpha selection

Polynomial and interaction features

Residual diagnostics and assumption checks

Gradient Boosting on engineered features

👏 Closing Note

This project demonstrates an end-to-end machine learning workflow:
EDA → Feature Engineering → Model Selection → Evaluation → Documentation