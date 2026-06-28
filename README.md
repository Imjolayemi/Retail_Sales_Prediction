# 📊 Retail Sales Prediction: End-to-End Regression Pipeline

## 📝 Project Overview

This project is an end-to-end Machine Learning pipeline designed to predict `total_sales` for a retail business. Built strictly following a robust data science workflow, this project transforms a messy, real-world dataset (5,050 rows, 30 columns) into a highly accurate **Gradient Boosting Regressor** capable of forecasting revenue on unseen data with **~93.6% accuracy (R²)**.

The primary goal of this project was not just model training, but mastering the critical steps of data cleaning, leakage prevention, automated preprocessing pipelines, and business-driven feature engineering.

## 🧹 Phase 1: Data Cleaning & Wrangling

Real-world data is rarely clean. The foundation of this model relies on strict data sanitization (reducing the dataset to 5,000 pristine rows):

* **Currency Parsing:** Cleaned the `unit_price` column by stripping mixed currency symbols (₦, NGN) and converting to strict floats.
* **Date Parsing:** Standardized the `order_date` column across four different rotating datetime formats.
* **Categorical Standardization:** Fixed inconsistent casing, stray whitespace, and underscores across high-cardinality text columns (e.g., `state`, `product_category`).
* **Boolean Normalization:** Unified `Yes/No`, `Y/N`, `1/0`, and `True/False` inputs into standard pandas booleans.
* **Impossible Value Handling:** Identified and removed logically impossible data entry errors (negative ages, quantities <= 0, and negative targets) instead of blindly imputing them.
* **Outlier Capping:** Handled extreme "unicorn" orders by mathematically capping `total_sales` at the 99th percentile (winsorizing) to stabilize the regression model.

## 🧠 Phase 2: EDA & Feature Engineering

Before modeling, exploratory data analysis was used to guide the creation of custom business logic features:

* **Temporal Features:** Extracted `order_month`, `order_dayofweek`, and `is_weekend` from the raw order dates to capture shopping seasonality.
* **Interaction Features:** Engineered `price_after_discount` and `price_vs_competitor` to mathematically represent the exact dollar advantage over rival stores.
* **Dimensionality Reduction:** Dropped high-cardinality identifiers (`order_id`, `customer_id`, `product_name`) to prevent the model from memorizing the data and blowing up the feature space.

## ⚙️ Phase 3: The Preprocessing Pipeline

To prevent **Data Leakage**, the dataset was strictly split into Training and Testing sets *before* any transformations. A Scikit-Learn `ColumnTransformer` was built to automate the preprocessing:

* **Numeric:** Median imputation for missing values + `StandardScaler`.
* **Categorical:** Most frequent imputation + `OneHotEncoder(handle_unknown='ignore')` to safely manage new categories in the future.
* **Boolean:** Separated into its own pipeline to avoid redundant encoding.

## 🚀 Phase 4: Model Selection & Tuning

Models were evaluated against a simple **Linear Regression Baseline** to ensure complexity was actually adding value.

* **Evaluated Models:** Ridge Regression, Random Forest, and Gradient Boosting.
* **Hyperparameter Tuning:** Selected Gradient Boosting as the leading architecture and utilized `GridSearchCV` (with 3-fold Cross Validation) to systematically find the optimal parameters: `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}`.

## 🏆 Final Evaluation & Business Impact

The model was evaluated once on the locked-away test set, yielding exceptional results:

| **Metric** | **Score** | **Business Interpretation** | 
| :--- | :--- | :--- | 
| **R² Score** | `0.936` | Captures ~94% of the variance in sales. Proves the model generalizes well and is not overfitted. | 
| **MAE** | `$9,333` | Average prediction error is highly precise in the context of large-scale B2B invoices. | 
| **RMSE** | `$16,377` | Penalizes the model for extreme outlier orders, providing a conservative reliability metric. | 

### Feature Importance (Sanity Check)

Extracted via `get_feature_names_out()`, the model successfully identified `unit_price`, `quantity`, and our custom `price_vs_competitor` features as the top drivers of revenue, proving the business logic was sound.

## 🛡️ Technical Pitfalls Avoided

This project successfully navigated several classic Machine Learning traps:

1. **Data Leakage:** Scalers and encoders were strictly fit *after* the train/test split.
2. **Target Imputation:** Invalid `total_sales` rows were dropped; the target was never fabricated.
3. **Cardinality Explosions:** Identifiers were dropped to prevent One-Hot Encoding from destroying memory.
4. **Production Crashes:** Used `handle_unknown="ignore"` to ensure the model doesn't break when encountering a brand new product category in production.

## 🔮 Stretch Goals & Future Iterations

To push the accuracy even further, the following techniques are slated for V2:

1. **Missing Indicators:** Add `add_indicator=True` to the competitor price imputer to analyze if a monopoly (missing competitor) acts as a sales signal.
2. **Advanced Algorithms:** Refactor the pipeline to utilize **XGBoost** or **LightGBM**.
3. **Target Encoding:** Replace One-Hot Encoding for the `state` column with historical average sales data using the `category_encoders` library.
4. **Interaction Variables:** Engineer a `discount_percent × loyalty_program_member` feature.