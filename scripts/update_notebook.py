#!/usr/bin/env python3
"""Apply improvement updates to airbnb_predictor.ipynb and strip outputs."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "airbnb_predictor.ipynb"


def set_source(cell, text: str):
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]]
    if lines:
        cell["source"].append(lines[-1])


def find_cell(cells, marker: str) -> int:
    for i, c in enumerate(cells):
        if marker in "".join(c.get("source", [])):
            return i
    raise ValueError(f"Cell not found: {marker}")


def clear_outputs(nb):
    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None


def main():
    with NOTEBOOK.open() as f:
        nb = json.load(f)

    cells = nb["cells"]

    set_source(
        cells[0],
        """# What Makes an Airbnb Worth the Price?
## Predicting Nightly Listing Prices Across New York City

---

This notebook builds regression models to predict the nightly price of Airbnb listings in New York City:

1. **Data Loading & Cleaning** — Inside Airbnb data, outlier filtering, and imputation
2. **Exploratory Visualizations** — price distributions and feature relationships
3. **Predictive Modeling** — naive baseline, Ridge regression, and tuned Random Forest
4. **Geographic Visualization** — median prices across NYC neighborhoods""",
    )

    set_source(
        cells[1],
        """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import json
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")

DATA_DIR = Path(".").resolve()
LISTINGS_PATH = DATA_DIR / "listings.csv.gz"
GEOJSON_PATH = DATA_DIR / "neighbourhoods.geojson"
""",
    )

    set_source(
        cells[3],
        """df = pd.read_csv(LISTINGS_PATH)
print(f"Loaded {len(df):,} listings from {LISTINGS_PATH.name}")""",
    )

    set_source(
        cells[7],
        """## 2. Feature Selection & Cleaning

We narrow the dataset to the columns most relevant to price prediction, then perform cleaning:

- **Price**: stored as a string like `"$120.00"` — strip `$` and commas and cast to float
- **Bathrooms**: extract the numeric value from text like `"1.5 baths"`
- **Outliers**: keep listings with nightly price between **$10 and $2,000**
- **Missing values**: impute where reasonable instead of dropping ~60% of rows (see audit below)

`neighbourhood_group_cleansed`, `latitude`, and `longitude` are kept for EDA but **dropped before modeling** — neighborhood dummies capture location; borough is redundant given neighborhood.""",
    )

    set_source(
        cells[9],
        """# Clean the price column — remove $ and commas, convert to float
if df_clean['price'].dtype == 'O':
    df_clean['price'] = df_clean['price'].str.replace('[\\$,]', '', regex=True).astype(float)

# Extract the number from the bathrooms text field (e.g. '1.5 baths' -> 1.5)
df_clean['bathrooms'] = df_clean['bathrooms_text'].str.extract(r'([0-9.]+)', expand=False).astype(float)
df_clean = df_clean.drop('bathrooms_text', axis=1)

# Require a listed price, then filter unrealistic nightly rates
rows_before_price = len(df_clean)
df_clean = df_clean.dropna(subset=['price'])
df_clean = df_clean[(df_clean['price'] >= 10) & (df_clean['price'] <= 2000)].copy()
print(f"Rows with valid price ($10–$2,000): {len(df_clean):,} (dropped {rows_before_price - len(df_clean):,})")""",
    )

    set_source(
        cells[10],
        """# Audit missing values BEFORE imputation
print("Missing values per column (before imputation):")
print(df_clean.isnull().sum().sort_values(ascending=False))
print(f"\\nRows before imputation: {len(df_clean):,}")

# Impute numeric features by room type where it makes sense
for col in ['bedrooms', 'bathrooms']:
    median_by_room = df_clean.groupby('room_type')[col].transform('median')
    global_median = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_by_room).fillna(global_median)

# Review score: only meaningful when there are reviews; otherwise use global median
review_median = df_clean.loc[df_clean['number_of_reviews'] > 0, 'review_scores_rating'].median()
df_clean['review_scores_rating'] = df_clean['review_scores_rating'].fillna(review_median)

# Superhost: treat missing as its own category (often newer hosts)
df_clean['host_is_superhost'] = df_clean['host_is_superhost'].fillna('unknown')

print(f"\\nRows after imputation: {len(df_clean):,}")
print(f"Rows retained vs raw upload: {len(df_clean)/len(df)*100:.1f}%")""",
    )

    # Insert imputation takeaway if missing
    if not any(
        c.get("cell_type") == "markdown" and "strict `dropna()`" in "".join(c.get("source", []))
        for c in cells
    ):
        cells.insert(
            11,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Takeaway:** Listings without a price are excluded. Other fields are imputed so we retain far more listings than a strict `dropna()` (~15k vs ~20k+ with price). Review scores for zero-review listings use the cohort median — a simplification that favors established listings; interpret coefficients accordingly.\n"
                ],
            },
        )

    cells = nb["cells"]

    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "Model 1" in src and "Linear Regression" in src:
            set_source(
                c,
                """## 5. Predictive Modeling

### Naive baseline — median price by neighborhood

We predict each test listing using the **median log price in its neighborhood**, using medians computed on the **training set only**.

### Model 1 — Ridge Regression

With ~200 neighborhood dummies, unregularized OLS is unstable. **Ridge regression** (`alpha=10`) handles multicollinearity better than plain linear regression.

### Model 2 — Random Forest Regressor

A non-linear ensemble with light hyperparameter search. We compare train vs test R², report **MAE and RMSE in dollars**, and run **5-fold CV on the training set only**.""",
            )
            break

    split_idx = find_cell(cells, "train_test_split")
    set_source(
        cells[split_idx],
        """# Train / Test Split
y = df_final['log_price']
X = df_final.drop('log_price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive baseline: median log_price by neighborhood (train only)
nbhd_cols = [c for c in X_train.columns if c.startswith('neighbourhood_cleansed_')]
train_nbhd = X_train[nbhd_cols].idxmax(axis=1).str.replace('neighbourhood_cleansed_', '', regex=False)
train_log_median = (
    pd.DataFrame({'nbhd': train_nbhd, 'log_price': y_train.values})
    .groupby('nbhd')['log_price']
    .median()
)
test_nbhd = X_test[nbhd_cols].idxmax(axis=1).str.replace('neighbourhood_cleansed_', '', regex=False)
naive_log_pred = test_nbhd.map(train_log_median).fillna(y_train.median()).values

naive_dollars_pred = np.expm1(naive_log_pred)
y_test_dollars = np.expm1(y_test)
naive_rmse = np.sqrt(mean_squared_error(y_test_dollars, naive_dollars_pred))
naive_mae = mean_absolute_error(y_test_dollars, naive_dollars_pred)
naive_r2 = r2_score(y_test, naive_log_pred)

print("Naive baseline (train median by neighborhood):")
print(f"  Test R² (log scale): {naive_r2:.4f}")
print(f"  RMSE (dollars):      ${naive_rmse:.2f}/night")
print(f"  MAE (dollars):       ${naive_mae:.2f}/night")""",
    )

    ridge_idx = find_cell(cells, "BASELINE MODEL")
    set_source(
        cells[ridge_idx],
        """# ---------------------------------------------------------
# MODEL 1: Ridge Regression
# ---------------------------------------------------------
ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
y_pred_dollars = np.expm1(y_pred)
ridge_rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
ridge_mae_dollars = mean_absolute_error(y_test_dollars, y_pred_dollars)

print("Ridge Regression Results:")
print(f"  R-squared (R², log scale): {r2:.4f}")
print(f"  RMSE (log scale):          {rmse:.4f}")
print(f"  RMSE (dollars):            ${ridge_rmse_dollars:.2f}/night")
print(f"  MAE (dollars):             ${ridge_mae_dollars:.2f}/night")""",
    )

    for i, c in enumerate(cells):
        if "VISUALIZATION 6" in "".join(c.get("source", [])):
            src = "".join(c.get("source", []))
            set_source(
                c,
                src.replace("Linear Regression", "Ridge Regression").replace(
                    "lr_model", "ridge_model"
                ),
            )
            break

    rf_idx = find_cell(cells, "IMPROVED MODEL: Random Forest")
    set_source(
        cells[rf_idx],
        """# ---------------------------------------------------------
# MODEL 2: Random Forest (hyperparameter search on train set)
# ---------------------------------------------------------
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [15, 25, None],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 0.3],
}
search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=12,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
rf_model = search.best_estimator_
print("Best RF params:", search.best_params_)

rf_y_pred = rf_model.predict(X_test)
rf_train_pred = rf_model.predict(X_train)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))

rf_y_pred_dollars = np.expm1(rf_y_pred)
rf_rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, rf_y_pred_dollars))
rf_mae_dollars = mean_absolute_error(y_test_dollars, rf_y_pred_dollars)

print("\\n--- MODEL COMPARISON ---")
print(f"  Naive (nbhd median)     R²: {naive_r2:.4f}")
print(f"  Ridge Regression        R²: {r2:.4f}")
print(f"  Random Forest (train)   R²: {rf_train_r2:.4f}")
print(f"  Random Forest (test)    R²: {rf_r2:.4f}")
print(f"  Overfitting gap:              {rf_train_r2 - rf_r2:.4f}")
print("-" * 40)
print(f"  Naive      RMSE/MAE ($): ${naive_rmse:.2f} / ${naive_mae:.2f}")
print(f"  Ridge      RMSE/MAE ($): ${ridge_rmse_dollars:.2f} / ${ridge_mae_dollars:.2f}")
print(f"  RF         RMSE/MAE ($): ${rf_rmse_dollars:.2f} / ${rf_mae_dollars:.2f}")""",
    )

    for i, c in enumerate(cells):
        if "RMSE in actual dollar terms" in "".join(c.get("source", [])):
            set_source(
                c,
                """# Dollar metrics reported above for all models.
print(f"Random Forest RMSE (log scale): {rf_rmse:.4f}")""",
            )
            break

    cv_idx = find_cell(cells, "cross_val_score")
    set_source(
        cells[cv_idx],
        """# 5-fold CV on TRAINING data only (avoids test-set leakage)
cv_scores = cross_val_score(
    clone(rf_model),
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1,
)

print(f"5-Fold CV R² (train set only): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Individual folds: {[round(s, 4) for s in cv_scores]}")""",
    )

    fi_idx = find_cell(cells, "VISUALIZATION 8: Feature Importance")
    set_source(
        cells[fi_idx],
        """# ---------------------------------------------------------
# VISUALIZATION 8: Feature Importance (grouped neighborhoods)
# ---------------------------------------------------------
importances = rf_model.feature_importances_
feature_names = X.columns

def readable_feature(name):
    if name.startswith('neighbourhood_cleansed_'):
        return 'Neighborhood: ' + name.replace('neighbourhood_cleansed_', '')
    labels = {
        'accommodates': 'Guest Capacity',
        'minimum_nights': 'Minimum Nights',
        'room_type_Private room': 'Room Type: Private Room',
        'bathrooms': 'Number of Bathrooms',
        'number_of_reviews': 'Total Reviews',
        'room_type_Hotel room': 'Room Type: Hotel Room',
        'review_scores_rating': 'Review Rating',
        'bedrooms': 'Number of Bedrooms',
        'host_is_superhost_t': 'Superhost Status',
        'host_is_superhost_unknown': 'Superhost: Unknown',
        'room_type_Shared room': 'Room Type: Shared Room',
    }
    return labels.get(name, name)

raw_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
raw_df['Group'] = raw_df['Feature'].apply(
    lambda x: 'Neighborhood (combined)' if x.startswith('neighbourhood_cleansed_') else readable_feature(x)
)
grouped = raw_df.groupby('Group', as_index=False)['Importance'].sum()
grouped = grouped.sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Group', data=grouped, palette='magma')
plt.title('Top 15 Drivers of Airbnb Prices in NYC (Random Forest)', fontsize=14)
plt.xlabel('Importance Score (neighborhoods summed)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()""",
    )

    if not any("VISUALIZATION 8b" in "".join(c.get("source", [])) for c in cells):
        choropleth_idx = find_cell(cells, "VISUALIZATION 9: Choropleth")
        cells.insert(
            choropleth_idx,
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [
                    line + ("\n" if i < 17 else "")
                    for i, line in enumerate(
                        """# VISUALIZATION 8b: Residuals in dollars (Random Forest)
residuals = y_test_dollars - rf_y_pred_dollars
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(residuals, bins=50, kde=True, ax=axes[0], color='teal')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title('Distribution of Prediction Errors ($)')
axes[0].set_xlabel('Actual − Predicted ($)')
axes[1].scatter(rf_y_pred_dollars, residuals, alpha=0.25, s=8, color='teal')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted Price ($)')
axes[1].set_ylabel('Residual ($)')
axes[1].set_title('Residuals vs Predicted Price')
plt.tight_layout()
plt.show()""".split(
                            "\n"
                        )
                    )
                ],
            },
        )

    cells = nb["cells"]
    choropleth_idx = find_cell(cells, "VISUALIZATION 9: Choropleth")
    set_source(
        cells[choropleth_idx],
        """# ---------------------------------------------------------
# VISUALIZATION 9: Choropleth Map — Median Price by Neighborhood
# ---------------------------------------------------------
neighborhood_median = (
    df_clean.groupby('neighbourhood_cleansed')['price']
    .median()
    .reset_index()
    .rename(columns={'price': 'median_price'})
)

with open(GEOJSON_PATH, 'r') as f:
    nyc_geo = json.load(f)

geo_neighborhoods = {f['properties']['neighbourhood'] for f in nyc_geo['features']}
data_neighborhoods = set(neighborhood_median['neighbourhood_cleansed'])
matched = data_neighborhoods & geo_neighborhoods
print(f"Matched neighborhoods: {len(matched)} / {len(data_neighborhoods)} in data")
if data_neighborhoods - geo_neighborhoods:
    print("  In data only (sample):", sorted(data_neighborhoods - geo_neighborhoods)[:5])

m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=nyc_geo,
    name='Median Nightly Price',
    data=neighborhood_median,
    columns=['neighbourhood_cleansed', 'median_price'],
    key_on='feature.properties.neighbourhood',
    fill_color='YlOrRd',
    fill_opacity=0.75,
    line_opacity=0.3,
    legend_name='Median Nightly Price (USD)',
    nan_fill_color='lightgrey',
    highlight=True,
).add_to(m)

neighborhood_dict = neighborhood_median.set_index('neighbourhood_cleansed')['median_price'].to_dict()

for feature in nyc_geo['features']:
    name = feature['properties']['neighbourhood']
    price = neighborhood_dict.get(name)
    feature['properties']['median_price'] = f"${price:.0f}" if price else "No data"

folium.GeoJson(
    nyc_geo,
    style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=['neighbourhood', 'median_price'],
        aliases=['Neighborhood:', 'Median Price:'],
        localize=True,
        sticky=True,
    ),
).add_to(m)

folium.LayerControl().add_to(m)
out_path = DATA_DIR / 'nyc_airbnb_choropleth.html'
m.save(str(out_path))
print(f"Map saved to {out_path}")
m""",
    )

    concl_idx = find_cell(cells, "Dynamic conclusion table")
    set_source(
        cells[concl_idx],
        """# Dynamic conclusion table
print("=" * 72)
print(f"{'Model':<36} {'R² (log)':>10} {'RMSE $':>10} {'MAE $':>10}")
print("-" * 72)
print(f"{'Naive (nbhd median)':<36} {naive_r2:>10.4f} {naive_rmse:>10.2f} {naive_mae:>10.2f}")
print(f"{'Ridge Regression':<36} {r2:>10.4f} {ridge_rmse_dollars:>10.2f} {ridge_mae_dollars:>10.2f}")
print(f"{'Random Forest (test)':<36} {rf_r2:>10.4f} {rf_rmse_dollars:>10.2f} {rf_mae_dollars:>10.2f}")
print(f"{'Random Forest (train)':<36} {rf_train_r2:>10.4f} {'—':>10} {'—':>10}")
print("=" * 72)
print(f"\\nRandom Forest 5-Fold CV R² (train only): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Overfitting gap (train - test R²):         {rf_train_r2 - rf_r2:.4f}")""",
    )

    # Update conclusion markdown
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "**Key findings:**" in "".join(c.get("source", [])):
            set_source(
                c,
                """**Key findings:**
- Listing size (`accommodates`, `bedrooms`, `bathrooms`) and neighborhood drive most of the price signal.
- Random Forest beats Ridge and the naive neighborhood median on test R² and dollar RMSE/MAE.
- Imputation retains more listings than listwise deletion; review-score imputation introduces mild bias toward reviewed listings.

**Limitations:**
- Snapshot data (not forecasting); prices reflect one Inside Airbnb scrape.
- Models predict association, not causal “what if” effects of host choices.
- Luxury outliers above $2,000/night are excluded by design.""",
            )
            break

    clear_outputs(nb)

    with NOTEBOOK.open("w") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
