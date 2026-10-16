"""
Used Car Price Prediction - Model Training and Evaluation Pipeline
==================================================================
This script loads the raw CarDekho dataset, performs data cleaning,
handles missing values and outliers, builds an scikit-learn Pipeline
with ColumnTransformer and TransformedTargetRegressor, benchmarks multiple
algorithms, tunes the top model, evaluates on a held-out test set, and saves
the production artifacts to disk.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cardekho_dataset.csv"
PIPELINE_PATH = MODELS_DIR / "car_price_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "app_metadata.joblib"


def format_inr(amount: float) -> str:
    """Format a monetary amount in Indian Rupee style (e.g. ₹4,25,000)."""
    val = int(round(amount))
    if val < 0:
        return f"-{format_inr(abs(val))}"
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    comma_part = ",".join(groups)
    return f"₹{comma_part},{last_three}"


def load_and_clean_data(data_path: Path):
    """
    Load raw CSV data and perform thorough cleaning:
    - Drop index artifacts and duplicate records
    - Standardize brand and category string representations
    - Impute physical impossibilities (seats = 0 -> 5)
    - Remove data corruption anomalies and extreme outliers
    """
    print("=" * 70)
    print("STEP 1: LOADING AND CLEANING DATASET")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}!")

    df = pd.read_csv(data_path)
    initial_rows, initial_cols = df.shape
    print(f"Initial raw shape: {initial_rows} rows x {initial_cols} columns")

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped redundant index column: 'Unnamed: 0'")

    # Drop duplicate rows
    dups_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped duplicate records: {dups_before} rows (Remaining: {len(df)})")

    # Clean text columns
    text_cols = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Brand standardization
    brand_replacements = {"ISUZU": "Isuzu", "Mercedes-AMG": "Mercedes-Benz"}
    df["brand"] = df["brand"].replace(brand_replacements)
    print(f"Standardized brand casing (e.g. ISUZU -> Isuzu)")

    # Data correction: Seats == 0 is impossible; mode/median is 5
    zero_seats = (df["seats"] == 0).sum()
    if zero_seats > 0:
        df["seats"] = df["seats"].replace({0: 5})
        print(f"Corrected {zero_seats} record(s) where seats == 0 to 5")

    # Remove corrupt records where km_driven was mistakenly recorded identical to selling_price
    corrupt_km_price = (df["km_driven"] == df["selling_price"]) & (df["km_driven"] > 100000)
    num_corrupt = corrupt_km_price.sum()
    df = df[~corrupt_km_price].copy().reset_index(drop=True)
    print(f"Removed corrupt records (km_driven entered identical to price > 1L): {num_corrupt}")

    # Remove extreme outliers that distort generalization
    # 1. km_driven: realistic bounds [100 km, 350,000 km]
    # 2. vehicle_age: realistic bounds [0 years, 20 years]
    # 3. selling_price: realistic bounds [₹30,000, ₹1.2 Crore]
    km_outliers = (df["km_driven"] < 100) | (df["km_driven"] > 350000)
    age_outliers = (df["vehicle_age"] < 0) | (df["vehicle_age"] > 20)
    price_outliers = (df["selling_price"] < 30000) | (df["selling_price"] > 12000000)

    total_outliers = (km_outliers | age_outliers | price_outliers).sum()
    df = df[~(km_outliers | age_outliers | price_outliers)].copy().reset_index(drop=True)
    print(f"Removed extreme outliers (km > 350k: {km_outliers.sum()}, age > 20: {age_outliers.sum()}, price > 1.2Cr: {price_outliers.sum()})")
    print(f"Total outliers removed: {total_outliers}")
    print(f"Final clean dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    return df


def prepare_features_and_split(df: pd.DataFrame):
    """
    Prepare feature columns and target, perform 80/20 train/test split,
    and group rare categories strictly on training split to prevent data leakage.
    """
    feature_cols = [
        "brand",
        "model",
        "vehicle_age",
        "km_driven",
        "seller_type",
        "fuel_type",
        "transmission_type",
        "mileage",
        "engine",
        "max_power",
        "seats",
    ]
    target_col = "selling_price"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 80/20 split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    print(f"Train/Test Split: {len(X_train)} train samples (80%), {len(X_test)} test samples (20%)")

    # Group rare categories into 'Other' based STRICTLY on X_train frequencies
    brand_counts = X_train["brand"].value_counts()
    frequent_brands = set(brand_counts[brand_counts >= 10].index.tolist())
    rare_brands = set(brand_counts[brand_counts < 10].index.tolist())

    X_train["brand"] = X_train["brand"].apply(lambda b: b if b in frequent_brands else "Other")
    X_test["brand"] = X_test["brand"].apply(lambda b: b if b in frequent_brands else "Other")

    model_counts = X_train["model"].value_counts()
    frequent_models = set(model_counts[model_counts >= 10].index.tolist())
    rare_models = set(model_counts[model_counts < 10].index.tolist())

    X_train["model"] = X_train["model"].apply(lambda m: m if m in frequent_models else "Other")
    X_test["model"] = X_test["model"].apply(lambda m: m if m in frequent_models else "Other")

    print(f"Grouped {len(rare_brands)} rare brand(s) and {len(rare_models)} rare model(s) with <10 train occurrences into 'Other'")

    return X_train, X_test, y_train, y_test, feature_cols, frequent_brands, frequent_models


def build_preprocessor():
    """Build scikit-learn ColumnTransformer for numeric and categorical pipelines."""
    num_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    cat_cols = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def evaluate_model(y_true, y_pred):
    """Calculate R2, MAE, RMSE, and MAPE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def benchmark_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train and compare LinearRegression, Ridge, RandomForest, and GradientBoosting.
    All target values are trained with TransformedTargetRegressor using np.log1p.
    """
    print("\n" + "=" * 70)
    print("STEP 2: MODEL BENCHMARKING (5-Fold CV + Test Evaluation)")
    print("=" * 70)

    candidate_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, random_state=RANDOM_STATE),
    }

    results = []
    trained_pipelines = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in candidate_models.items():
        t0 = time.time()
        # Wrap target in log1p / expm1 transformation
        ttr = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", ttr),
        ])

        # 5-Fold Cross Validation on Training Split
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_r2_mean = float(np.mean(cv_scores))
        cv_r2_std = float(np.std(cv_scores))

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        elapsed = time.time() - t0

        trained_pipelines[name] = pipeline
        results.append({
            "Model": name,
            "CV R² (Mean±Std)": f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
            "Test R²": metrics["R2"],
            "Test MAE": metrics["MAE"],
            "Test RMSE": metrics["RMSE"],
            "Test MAPE (%)": metrics["MAPE"] * 100,
            "Train Time (s)": elapsed,
        })

    # Display comparison table
    summary_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print("-" * 105)
    print(f"{'Model':<18} | {'CV R² (5-Fold)':<18} | {'Test R²':<9} | {'Test MAE':<14} | {'Test RMSE':<14} | {'Test MAPE':<10} | {'Time (s)':<8}")
    print("-" * 105)
    for r in results:
        mae_str = format_inr(r["Test MAE"])
        rmse_str = format_inr(r["Test RMSE"])
        mape_str = f"{r['Test MAPE (%)']:.2f}%"
        r2_str = f"{r['Test R²']:.4f}"
        time_str = f"{r['Train Time (s)']:.2f}s"
        print(f"{r['Model']:<18} | {r['CV R² (Mean±Std)']:<18} | {r2_str:<9} | {mae_str:<14} | {rmse_str:<14} | {mape_str:<10} | {time_str:<8}")
    print("-" * 105)

    return trained_pipelines, summary_df


def tune_best_model(X_train, y_train, X_test, y_test, preprocessor):
    """
    Tune the best model (RandomForest) with RandomizedSearchCV (n_iter=10, cv=3).
    """
    print("\n" + "=" * 70)
    print("STEP 3: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 70)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])

    param_distributions = {
        "model__regressor__n_estimators": [100, 150, 200],
        "model__regressor__max_depth": [14, 18, 22, None],
        "model__regressor__min_samples_split": [2, 4, 6],
        "model__regressor__min_samples_leaf": [1, 2, 4],
        "model__regressor__max_features": ["sqrt", 0.8, 1.0],
    }

    print("Running RandomizedSearchCV (n_iter=10, cv=3, scoring=neg_mean_absolute_error)...")
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"Hyperparameter tuning completed in {elapsed:.2f}s")
    print(f"Optimal parameters: {search.best_params_}")

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (Tuned RandomForest)")
    print("=" * 70)
    print(f"  Test R²:   {metrics['R2']:.4f}")
    print(f"  Test MAE:  {format_inr(metrics['MAE'])} ({metrics['MAE']:,.0f} INR)")
    print(f"  Test RMSE: {format_inr(metrics['RMSE'])} ({metrics['RMSE']:,.0f} INR)")
    print(f"  Test MAPE: {metrics['MAPE'] * 100:.2f}%")
    print("=" * 70)

    return best_pipeline, metrics, y_pred


def build_app_metadata(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_cols: list,
    num_cols: list,
    cat_cols: list,
    test_metrics: dict,
    best_params: dict,
):
    """
    Construct rich metadata artifact required for the Streamlit UI:
    - Input feature names and order
    - Valid options for each categorical field (brands, models per brand, fuel, etc.)
    - Numeric defaults (min, max, median, 25%, 75%)
    - Model specs lookup per (brand, model) for dynamic smart pre-filling
    - Residual percentiles (10th and 90th) for prediction confidence range
    """
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING APP METADATA")
    print("=" * 70)

    # 1. Categorical options
    valid_brands = sorted([b for b in X_train["brand"].unique() if b != "Other"]) + ["Other"]
    valid_fuel_types = sorted(X_train["fuel_type"].unique().tolist())
    valid_transmissions = sorted(X_train["transmission_type"].unique().tolist())
    valid_seller_types = sorted(X_train["seller_type"].unique().tolist())

    # Brand to models mapping
    brand_model_map = {}
    for b in valid_brands:
        if b == "Other":
            brand_model_map[b] = ["Other"]
        else:
            models_for_b = X_train[X_train["brand"] == b]["model"].unique().tolist()
            models_clean = sorted([m for m in models_for_b if m != "Other"])
            models_clean.append("Other")
            brand_model_map[b] = models_clean

    # 2. Numeric defaults (overall)
    numeric_defaults = {}
    for col in num_cols:
        s = df[col]
        numeric_defaults[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
        }

    # 3. Model specs lookup: typical specs per (brand, model) to auto-fill inputs
    specs_lookup = {}
    for (b, m), group in df.groupby(["brand", "model"]):
        specs_lookup[(b, m)] = {
            "engine": int(round(group["engine"].median())),
            "max_power": float(round(group["max_power"].median(), 1)),
            "mileage": float(round(group["mileage"].median(), 2)),
            "seats": int(group["seats"].mode()[0]),
            "fuel_type": group["fuel_type"].mode()[0],
            "transmission_type": group["transmission_type"].mode()[0],
        }

    # 4. Residual bounds for price range estimate (10th and 90th percentile of relative error)
    rel_errors = (y_test.values - y_pred) / y_pred
    p10_error = float(np.percentile(rel_errors, 10))
    p90_error = float(np.percentile(rel_errors, 90))

    metadata = {
        "feature_names": feature_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "categorical_options": {
            "brand": valid_brands,
            "seller_type": valid_seller_types,
            "fuel_type": valid_fuel_types,
            "transmission_type": valid_transmissions,
            "brand_model_map": brand_model_map,
        },
        "numeric_defaults": numeric_defaults,
        "specs_lookup": specs_lookup,
        "test_metrics": {
            "r2": float(test_metrics["R2"]),
            "mae": float(test_metrics["MAE"]),
            "rmse": float(test_metrics["RMSE"]),
            "mape": float(test_metrics["MAPE"]),
        },
        "residual_bounds": {
            "p10": p10_error,
            "p90": p90_error,
        },
        "best_model_name": "Tuned RandomForestRegressor",
        "best_hyperparameters": best_params,
        "train_samples": len(X_train),
        "test_samples": len(y_test),
        "target_col": "selling_price",
    }

    return metadata


def save_artifacts(pipeline, metadata):
    """Save fitted pipeline and metadata using joblib, and report file sizes."""
    print("=" * 70)
    print("STEP 5: SAVING ARTIFACTS")
    print("=" * 70)

    joblib.dump(pipeline, PIPELINE_PATH, compress=3)
    joblib.dump(metadata, METADATA_PATH, compress=3)

    pipeline_size = PIPELINE_PATH.stat().st_size / (1024 * 1024)
    metadata_size = METADATA_PATH.stat().st_size / 1024

    print(f"Saved Fitted Pipeline -> {PIPELINE_PATH} ({pipeline_size:.2f} MB)")
    print(f"Saved App Metadata    -> {METADATA_PATH} ({metadata_size:.2f} KB)")
    print("=" * 70)


def main():
    start_total = time.time()

    # 1. Load and clean
    df = load_and_clean_data(DATA_PATH)

    # 2. Features and split
    X_train, X_test, y_train, y_test, feature_cols, freq_brands, freq_models = prepare_features_and_split(df)

    # 3. Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # 4. Benchmark candidate models
    trained_pipelines, summary_df = benchmark_models(X_train, y_train, X_test, y_test, preprocessor)

    # 5. Tune best model
    best_pipeline, test_metrics, y_pred = tune_best_model(X_train, y_train, X_test, y_test, preprocessor)

    # 6. Build metadata
    best_params = {}
    if hasattr(best_pipeline.named_steps["model"].regressor, "get_params"):
        best_params = best_pipeline.named_steps["model"].regressor.get_params()

    metadata = build_app_metadata(
        df=df,
        X_train=X_train,
        y_test=y_test,
        y_pred=y_pred,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        test_metrics=test_metrics,
        best_params=best_params,
    )

    # 7. Save artifacts
    save_artifacts(best_pipeline, metadata)

    total_elapsed = time.time() - start_total
    print(f"\nAll steps completed successfully in {total_elapsed:.1f} seconds!")


if __name__ == "__main__":
    main()
