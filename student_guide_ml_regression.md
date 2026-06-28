# Student Guide: Building a Sales Regression Model

**Dataset:** `sales_dataset.csv` (5,050 rows × 30 columns)
**Goal:** Predict `total_sales` from order, customer, product, and operational features
**Prerequisites:** Python, pandas, scikit-learn basics (you've already covered `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, and feature selection — this guide builds directly on that)

Work through the steps in order. Each one has a clear deliverable so you can check your progress before moving on. Don't skip the cleaning steps — the dataset is deliberately messy, and a model trained on unclean data will give you misleading results.

---

## Step 0 — Set up

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("sales_dataset.csv")
print(df.shape)
df.head()
```

**Checkpoint:** confirm you see 5,050 rows and 30 columns.

---

## Step 1 — First look at the data

Before touching anything, profile it:

```python
df.info()
df.describe(include="all")
df.isna().sum().sort_values(ascending=False)
df.duplicated().sum()
```

Write down (in a notebook markdown cell, not just in your head):
- Which columns are missing values, and roughly what %
- Which columns *look* numeric but aren't (`dtype: object`)
- Whether you can see duplicate rows

This step matters because it's the difference between cleaning deliberately and cleaning by trial and error.

---

## Step 2 — Remove duplicates

```python
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)
```

**Checkpoint:** row count should drop by exactly 50, to 5,000.

---

## Step 3 — Fix `unit_price` (mixed currency formatting)

This column has plain numbers mixed with strings like `"₦24,579.50"` and `"NGN 37,906"`. You need every value as a clean float.

```python
def clean_price(val):
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace("₦", "").replace("NGN", "").replace(",", "").strip()
    return float(val)

df["unit_price"] = df["unit_price"].apply(clean_price)
df["unit_price"].dtype  # should be float64 now
```

---

## Step 4 — Parse `order_date` (mixed date formats)

The dates rotate through 4 formats. Let pandas infer per-row:

```python
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
df["order_date"].dtype  # datetime64[ns]
```

If you get parsing errors, inspect the offending rows with `errors="coerce"` first, then decide whether to drop or fix them — don't silently lose rows without checking how many you'd lose.

---

## Step 5 — Standardize messy categoricals

`state`, `product_category`, and `payment_method` have inconsistent casing, stray whitespace, or underscores.

```python
text_cols = ["state", "product_category", "payment_method", "region",
             "customer_segment", "store_type", "marketing_channel"]

for col in text_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace("_", " ")
        .str.title()
    )

df["state"].unique()        # check casing is now consistent
df["payment_method"].unique()
```

**Checkpoint:** `df["state"].nunique()` should match the number of real states, not be inflated by casing variants.

---

## Step 6 — Standardize boolean columns

`loyalty_program_member` and `is_holiday` mix `Yes/No`, `Y/N`, `1/0`, `True/False`.

```python
true_values = {"yes", "y", "1", "true"}

def to_bool(val):
    if pd.isna(val):
        return np.nan
    return str(val).strip().lower() in true_values

df["loyalty_program_member"] = df["loyalty_program_member"].apply(to_bool)
df["is_holiday"] = df["is_holiday"].apply(to_bool)
```

These stay as nullable booleans for now — you'll decide how to handle the missing ones in Step 8.

---

## Step 7 — Handle invalid / impossible values

Some values are outside plausible ranges (data entry errors, not real outliers). Before doing statistical outlier detection, fix the impossible ones:

| Column | Invalid condition | Suggested fix |
|---|---|---|
| `customer_age` | `< 0` or `> 100` | set to `NaN` |
| `quantity` | `<= 0` | set to `NaN` (can't order zero or negative units) |
| `discount_percent` | `< 0` or `> 100` | clip to `[0, 100]` or set to `NaN` |
| `customer_satisfaction_score` | `< 1` or `> 5` | clip to `[1, 5]` or set to `NaN` |
| `delivery_days` | `< 0` or `> 60` | set to `NaN` |
| `total_sales` | `< 0` | set to `NaN` (target can't be negative) |

```python
df.loc[(df["customer_age"] < 0) | (df["customer_age"] > 100), "customer_age"] = np.nan
df.loc[df["quantity"] <= 0, "quantity"] = np.nan
df["discount_percent"] = df["discount_percent"].clip(0, 100)
df["customer_satisfaction_score"] = df["customer_satisfaction_score"].clip(1, 5)
df.loc[(df["delivery_days"] < 0) | (df["delivery_days"] > 60), "delivery_days"] = np.nan
df = df[df["total_sales"] >= 0]  # drop rows with impossible negative target
```

For `total_sales`, since it's your target, rows with invalid values should be **dropped**, not imputed — you don't want to predict a fabricated number.

---

## Step 8 — Decide on rare, extreme `total_sales` outliers

A few rows have `total_sales` 8–15x larger than typical. These are real (not data errors), but extreme values can distort a regression model.

```python
df["total_sales"].describe()
sns.boxplot(x=df["total_sales"])
```

Two reasonable options — pick one and justify it in your notes:
1. **Cap (winsorize)** at the 99th percentile.
2. **Keep them**, but use a model less sensitive to outliers (tree-based) and/or evaluate with both RMSE and MAE (MAE is less outlier-sensitive).

```python
# Option 1: capping
cap = df["total_sales"].quantile(0.99)
df["total_sales"] = df["total_sales"].clip(upper=cap)
```

---

## Step 9 — Exploratory Data Analysis

Before modeling, understand what's actually predictive.

```python
# Numeric correlations with the target
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols].corr()["total_sales"].sort_values(ascending=False)

# Visualize key relationships
sns.scatterplot(data=df, x="unit_price", y="total_sales", alpha=0.3)
sns.boxplot(data=df, x="product_category", y="total_sales")
sns.boxplot(data=df, x="loyalty_program_member", y="total_sales")
```

Look for: which numeric features actually correlate with `total_sales`, and which categories show visibly different sales distributions. This shapes your feature engineering in the next step.

---

## Step 10 — Feature engineering

```python
df["order_month"] = df["order_date"].dt.month
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["is_weekend"] = df["order_dayofweek"].isin([5, 6])

df["price_after_discount"] = df["unit_price"] * (1 - df["discount_percent"] / 100)
df["price_vs_competitor"] = df["unit_price"] - df["competitor_price"]
```

**Drop columns that won't help (or would leak):**

```python
drop_cols = ["order_id", "customer_id", "product_name", "sales_rep_id",
             "order_date", "day_of_week"]  # day_of_week duplicates order_dayofweek
df = df.drop(columns=drop_cols)
```

`order_id`, `customer_id`, `product_name`, and `sales_rep_id` are high-cardinality identifiers — they won't generalize and can cause your model to memorize rather than learn patterns.

---

## Step 11 — Train/test split

**Do this before any imputation or scaling.** Fitting imputers/scalers on the full dataset (including test data) leaks information from the test set into training — the same leakage issue you worked through with `ColumnTransformer` before.

```python
X = df.drop(columns=["total_sales"])
y = df["total_sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 12 — Build the preprocessing pipeline

Separate your columns by type, then build a `ColumnTransformer` exactly like your earlier workflow:

```python
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()
boolean_features = ["loyalty_program_member", "is_holiday"]

# remove booleans from numeric/categorical lists if they were caught there
numeric_features = [c for c in numeric_features if c not in boolean_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

boolean_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("bool", boolean_transformer, boolean_features),
])
```

---

## Step 13 — Baseline model

Always start simple. A baseline tells you whether a complex model is actually earning its complexity.

```python
baseline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

baseline.fit(X_train, y_train)
preds = baseline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("R²:", r2_score(y_test, preds))
```

---

## Step 14 — Try stronger models

```python
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
    results[name] = scores.mean()
    print(f"{name}: R² = {scores.mean():.3f} (+/- {scores.std():.3f})")
```

Tree-based models (`RandomForest`, `GradientBoosting`) typically handle non-linear relationships and categorical splits better than linear models for this kind of data — see how much they actually improve over the baseline.

---

## Step 15 — Tune your best model

Pick whichever model performed best in Step 14, then tune it:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
}

pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))])

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)
```

---

## Step 16 — Final evaluation on the test set

Only touch the test set once, at the very end:

```python
best_model = grid_search.best_estimator_
final_preds = best_model.predict(X_test)

print("Final MAE:", mean_absolute_error(y_test, final_preds))
print("Final RMSE:", mean_squared_error(y_test, final_preds) ** 0.5)
print("Final R²:", r2_score(y_test, final_preds))

# Residual plot — look for patterns that suggest a missing feature or non-linearity
residuals = y_test - final_preds
sns.scatterplot(x=final_preds, y=residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
```

---

## Step 17 — Feature importance

Get back the feature names from your `ColumnTransformer` (this is the same `get_feature_names_out()` pattern from your earlier sklearn work):

```python
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

importance_df.head(15)
```


Use this to sanity-check the model: do `unit_price`, `quantity`, and `discount_percent` come out on top? If something unexpected dominates, it's worth investigating whether it's a real signal or a leftover leakage issue.

---

## Common pitfalls to avoid

- **Fitting scalers/encoders before the train/test split.** Always split first.
- **Imputing the target column.** Drop invalid target rows instead of filling them in.
- **One-hot encoding before checking cardinality.** High-cardinality columns (like `product_name` or `customer_id`) blow up your feature space — drop or target-encode them instead.
- **Judging a model only on R².** Check MAE/RMSE too, and always look at a residual plot — R² alone can hide systematic bias.
- **Forgetting `handle_unknown="ignore"` in `OneHotEncoder`.** Without it, any category in the test set unseen during training will crash prediction.

---

## Stretch goals (optional)

1. Try target-encoding `state` or `product_subcategory` instead of one-hot encoding, and compare performance.
2. Engineer an interaction feature (e.g. `discount_percent × loyalty_program_member`) and see if it improves the model.
3. Compare model performance with and without capping the `total_sales` outliers from Step 8.
4. Build a `MissingIndicator` feature for `competitor_price` (it's missing ~20% of the time) instead of just imputing — does "missingness itself" carry signal?
5. Try `XGBoost` or `LightGBM` if you have them installed, and compare against `GradientBoostingRegressor`.
