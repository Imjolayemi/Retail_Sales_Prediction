# Sales Dataset — Data Dictionary

**File:** `sales_dataset.csv`
**Rows:** 5,050 (5,000 unique + 50 intentional duplicates)
**Columns:** 30
**Target variable:** `total_sales` (regression)

This dataset is synthetic and deliberately "dirty" so it can be used to
practice a full data science pipeline: EDA, cleaning, type coercion,
encoding, outlier handling, feature engineering, and model training.

## Columns

| Column | Type (intended) | Description | Known issues |
|---|---|---|---|
| `order_id` | string | Unique order identifier | None |
| `customer_id` | string | Customer identifier (repeats across orders) | None |
| `order_date` | date | Date of order | **4 inconsistent formats** (`YYYY-MM-DD`, `DD/MM/YYYY`, `YYYY/MM/DD`, `Month DD, YYYY`) |
| `customer_age` | float | Customer age in years | ~7% missing; outliers (negative, >140) |
| `customer_gender` | category | Male / Female / Other | ~4% missing |
| `customer_segment` | category | Consumer / Corporate / Home Office | None |
| `region` | category | Nigerian geopolitical zone | None |
| `state` | category | State/city within region | Inconsistent casing & stray whitespace (`" lagos "`, `"LAGOS"`, `"Lagos"`) |
| `store_type` | category | Online / Retail / Wholesale | None |
| `product_category` | category | High-level product category | Some inconsistent casing |
| `product_subcategory` | category | Subcategory within `product_category` | None |
| `product_name` | string | Generated product name | None |
| `unit_price` | float | Price per unit | **Stored as mixed object dtype** — some values are plain floats, some prefixed `₦` with thousands separators, some `"NGN ####"` |
| `quantity` | int | Units ordered | Outliers: negative values, values up to 800 |
| `discount_percent` | float | Discount applied (%) | ~10% missing; invalid values (>100%, negative) |
| `shipping_cost` | float | Shipping cost | ~6% missing |
| `payment_method` | category | Payment method used | Inconsistent formatting (`"Credit Card"` vs `"CREDIT_CARD"`) |
| `marketing_channel` | category | Acquisition channel | None |
| `customer_tenure_months` | int | Months since customer's first order | None |
| `loyalty_program_member` | boolean | Loyalty program membership | **Mixed boolean encodings** (`Yes/No`, `Y/N`, `1/0`, `True/False`); ~8% missing |
| `competitor_price` | float | Nearby competitor's price for similar product | ~20% missing (largest gap in the dataset) |
| `economic_index` | float | Synthetic macroeconomic indicator | None |
| `season` | category | Rainy / Dry (derived from order month) | None |
| `day_of_week` | category | Day name derived from `order_date` | None |
| `is_holiday` | boolean | Whether order date was a holiday | Mixed boolean encodings, same pattern as `loyalty_program_member` |
| `sales_rep_id` | string | Sales rep identifier | None |
| `sales_rep_experience_years` | float | Rep's years of experience | ~3% missing |
| `customer_satisfaction_score` | float | Self-reported score, intended range 1–5 | ~12% missing; some values outside valid range (0, negative, >5) |
| `delivery_days` | float | Days to deliver | ~5% missing; outliers (negative, up to 365) |
| `total_sales` | float | **Target.** Final sale amount | A few extreme outliers (8–15x normal) and a few invalid negative values |

## Known dataset-level issues

- **50 fully duplicated rows** mixed into the data (not flagged — find via `.duplicated()`).
- **Row order is shuffled**, so no positional patterns exist by design.
- Relationships to `total_sales` are real but noisy: `unit_price`, `quantity`,
  and `discount_percent` are the strongest drivers; `sales_rep_experience_years`,
  `loyalty_program_member`, and `marketing_channel` have smaller, genuine effects.
  This keeps the regression task non-trivial rather than a deterministic formula.

## Suggested pipeline steps

1. Parse `order_date` with mixed formats (`pd.to_datetime(..., format='mixed')` or manual format detection).
2. Clean `unit_price` (strip `₦`/`NGN`, remove commas, cast to float).
3. Standardize categorical casing/whitespace (`.str.strip().str.lower()` or title-case).
4. Normalize boolean columns to a single consistent representation.
5. Drop or investigate duplicate rows.
6. Handle missing values (imputation strategy varies by column — e.g. `competitor_price` at 20% missing may warrant a "missing" indicator rather than naive imputation).
7. Clip or flag invalid/outlier values (negative quantities, satisfaction scores outside 1–5, age outside plausible range, negative `total_sales`).
8. Feature engineer from `order_date` (month, day-of-week already provided, is_weekend, etc.).
9. Encode categoricals (OneHotEncoder / target encoding) within a `ColumnTransformer` + `Pipeline`, consistent with your usual scikit-learn workflow.
10. Train/test split, then fit a regression model (start with a tree-based baseline since relationships are non-linear and there are many categoricals).
