# 🚗 AutoValuate • Used Car Price Prediction in India

An end-to-end Machine Learning web application built with **scikit-learn** and **Streamlit** to estimate fair market valuation for used cars across India. The application features a leakage-free preprocessing and modeling pipeline, target log-transformation, automated hyperparameter tuning, residual-based confidence intervals, and a responsive dark-themed user interface.

---

## 📌 Project Overview

Valuing a pre-owned vehicle involves non-linear interactions across vehicle age, odometer mileage, brand equity, powertrain characteristics, fuel type, and transmission. This application delivers:
- **Instant Fair-Market Valuation**: Predicts selling prices in Indian Rupee format (e.g., `₹4,25,000`).
- **Statistical Confidence Ranges**: Computes an 80% confidence interval (`10th` to `90th` residual percentile bounds) to account for market variance.
- **Dynamic Auto-Specs Pre-fill**: Selecting a Make and Model dynamically pre-fills typical engine displacement, power, fuel efficiency, and seating specs while allowing user adjustments.
- **Leakage-Free Sklearn Pipeline**: Encapsulates preprocessing (`ColumnTransformer`) and model inference inside a single `joblib` artifact.

---

## 📂 Repository Structure

```
car-price-prediction/
├── data/
│   └── raw/
│       └── cardekho_dataset.csv   # Raw CarDekho dataset (~15,411 records)
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis with visualizations
├── models/
│   ├── car_price_pipeline.joblib  # Saved scikit-learn pipeline (model + preprocessor)
│   └── app_metadata.joblib        # Options, defaults, metrics, residual bounds
├── app.py                         # Streamlit interactive web application
├── train.py                       # Data cleaning, cross-validation, tuning & saving
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Documentation
```

---

## 📊 Dataset & Cleaning

The dataset is sourced from CarDekho and contains transaction listings across 32 vehicle manufacturers.

### Data Cleaning Operations (`train.py`)
1. **Deduplication**: Removed 167 exact duplicate records.
2. **Text Normalization**: Standardized brand names (e.g., `ISUZU` $\rightarrow$ `Isuzu`, `Mercedes-AMG` $\rightarrow$ `Mercedes-Benz`).
3. **Physical Validity**: Imputed 2 records with `seats == 0` to the median/mode of 5 seats.
4. **Data Corruption Removal**: Removed 13 records where `km_driven` was mistakenly entered identical to the selling price.
5. **Outlier Filtering**: Removed unrepresentative outliers:
   - `km_driven`: Excluded vehicles $> 350,000\text{ km}$ or $< 100\text{ km}$.
   - `vehicle_age`: Excluded vehicles $> 20\text{ years}$.
   - `selling_price`: Excluded vehicles $> ₹1.2\text{ Crore}$ (ultra-exotics with $n \le 2$) and $< ₹30,000$.
6. **Rare Category Handling**: Mapped brands and models with $< 10$ occurrences strictly in `X_train` to `'Other'` to avoid overfitting on unrepresentative samples.

---

## ⚙️ Model Architecture & Benchmark

### Pipeline Design
- **Numeric Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Pipeline**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Target Transformation**: The selling price target is heavily right-skewed. The pipeline uses `TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)` to train on $\log(1 + y)$ and invert back to Rupees seamlessly during prediction.

### Algorithm Comparison (5-Fold Cross Validation on 12,164 Train Samples)

| Model | 5-Fold CV $R^2$ (Mean ± Std) | Test $R^2$ | Test MAE (INR) | Test RMSE (INR) | Test MAPE (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | $0.8923 \pm 0.0174$ | $0.9007$ | ₹1,12,287 | ₹2,59,962 | 13.99% | 3.77s |
| **Ridge Regression** | $0.8887 \pm 0.0145$ | $0.8998$ | ₹1,14,473 | ₹2,61,139 | 14.06% | 2.32s |
| **Random Forest Regressor** | **0.9302 ± 0.0144** | **0.9195** | **₹98,901** | **₹2,34,026** | **12.94%** | 6.91s |
| **Gradient Boosting Regressor** | $0.9100 \pm 0.0137$ | $0.9090$ | ₹1,10,197 | ₹2,48,780 | 13.65% | 11.11s |

### Final Tuned Model
The best performing model, **RandomForestRegressor**, was tuned using `RandomizedSearchCV` ($n_{\text{iter}}=10$, $3$-fold CV, optimizing for negative MAE):
- **Best Hyperparameters**:
  - `n_estimators`: 150
  - `max_depth`: 22
  - `max_features`: 0.8
  - `min_samples_split`: 6
  - `min_samples_leaf`: 1
- **Held-Out Test Set Performance (3,041 Unseen Cars)**:
  - **$R^2$ Score**: **0.9204** (92.0% of price variance explained)
  - **MAE**: **₹96,921**
  - **RMSE**: **₹2,32,760**
  - **MAPE**: **12.51%**

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone the repository and set up a Python virtual environment (Python 3.10+ recommended):

```bash
# Clone the repository
git clone <repo-url>
cd car-price-prediction-sl

# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# macOS / Linux:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Train and Evaluate the Model
To reproduce data cleaning, benchmark algorithms, run hyperparameter tuning, and persist joblib artifacts:

```bash
python train.py
```

*Output:*
- Prints cleaning statistics and outlier counts.
- Displays 5-fold CV and test metric comparison table.
- Saves `models/car_price_pipeline.joblib` (~16 MB).
- Saves `models/app_metadata.joblib` (~4 KB).

### 3. Launch the Streamlit Web Application
Start the interactive Streamlit valuation app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Acceptance & Quality Verification

1. **Monotonic Depreciation**: Verified that prices decrease logically as vehicle age and odometer readings increase (e.g., a Maruti Swift drops from ~₹7.4 Lakh at 1 year / 10k km to ~₹4.3 Lakh at 6 years / 60k km, and ~₹2.1 Lakh at 14 years / 130k km).
2. **Offline Decoupling**: The web application loads directly from `models/` without querying or touching raw CSV files.
3. **No Unhandled Errors**: UI inputs are validated with fallback bounds; model inferences are wrapped in exception guards with user-friendly notices.

---

## ⚠️ Limitations & Real-World Considerations

- **Regional Tax Differences**: In India, state road taxes (RTO) vary between 8% and 20% across states (e.g., Delhi vs. Karnataka vs. Maharashtra). The model reflects nationwide average transaction values.
- **Vehicle Condition & Service Records**: Tabular listings do not reflect tire wear, cosmetic scratches, accidental damage, or full dealership service records, which can adjust fair value by ±10–15%.
- **Supercars & Ultra-Luxury Exotics**: Vehicles with new prices exceeding ₹1.5 Crore (e.g. Ferrari, Rolls-Royce) have minimal sample size in public datasets and are grouped into 'Other' or excluded as outliers.
