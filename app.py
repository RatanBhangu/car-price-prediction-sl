"""
Used Car Price Prediction - Streamlit Application
=================================================
A production-ready, beautiful web application for predicting used car prices
in India using scikit-learn and Streamlit.
"""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium dark aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background & container */
    .stApp {
        background: radial-gradient(circle at top right, #111827, #0B0F19 50%, #030712 100%);
        color: #F3F4F6;
    }

    /* Hero header styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 30%, #38BDF8 70%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin: 0;
    }

    /* Card styling */
    .custom-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }

    /* Result metric box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 12px 35px -8px rgba(16, 185, 129, 0.25);
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #34D399;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 12px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .range-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        display: inline-block;
        font-size: 0.95rem;
        color: #E2E8F0;
    }
    .range-highlight {
        color: #38BDF8;
        font-weight: 700;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 12px 28px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369A1 0%, #1D4ED8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Constants & Helper Functions
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "models" / "car_price_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "app_metadata.joblib"


def format_inr(amount: float) -> str:
    """Format an amount into Indian numbering system (e.g. ₹4,25,000)."""
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


def format_lakhs(amount: float) -> str:
    """Format in Lakhs/Crores for quick human readability."""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Crore"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} Lakh"
    else:
        return format_inr(amount)


# -----------------------------------------------------------------------------
# Step 3: Load Artifacts with Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning pipeline...")
def load_artifacts():
    """
    Load the trained scikit-learn pipeline and associated metadata once.
    Returns (pipeline, metadata) or raises FileNotFoundError.
    """
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        return None, None

    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


pipeline, metadata = load_artifacts()

if pipeline is None or metadata is None:
    st.error(
        "⚠️ **Model Artifacts Not Found!**\n\n"
        "The model pipeline (`car_price_pipeline.joblib`) or metadata (`app_metadata.joblib`) "
        "could not be located in `models/`.\n\n"
        "👉 **Action Required:** Please run `python train.py` from your terminal to train "
        "and persist the model artifacts, then reload this application.",
        icon="🚨",
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: Model Performance & Accuracy Explanations
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Performance")

    test_metrics = metadata.get("test_metrics", {})
    r2_val = test_metrics.get("r2", 0.9204)
    mae_val = test_metrics.get("mae", 96921.0)
    mape_val = test_metrics.get("mape", 0.1251)
    rmse_val = test_metrics.get("rmse", 232760.0)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="R² Score", value=f"{r2_val:.3f}")
    with col_m2:
        st.metric(label="Mean Error (MAPE)", value=f"{mape_val * 100:.1f}%")

    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; margin-top: 10px;">
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.4;">
                💡 <b>Plain-English Interpretation:</b><br>
                On unseen test transactions, this model predicts vehicle valuations with <b>{r2_val*100:.1f}% explained variance</b>.
                On average, the prediction is within <b>{format_inr(mae_val)}</b> (approx. <b>{mape_val*100:.1f}%</b>) of actual market prices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown(f"**Architecture:** {metadata.get('best_model_name', 'Random Forest Regressor')}")
    st.markdown(f"**Target Scaling:** `log1p(INR)` with automatic inverse expm1")
    st.markdown(f"**Training Split:** {metadata.get('train_samples', 12164):,} cars (80%)")
    st.markdown(f"**Validation:** 5-Fold Cross-Validation")

    st.markdown("---")
    st.caption("AutoValuate ML v1.0 • Built with scikit-learn & Streamlit")

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🚗 AutoValuate • Used Car Price Prediction</div>
        <p class="hero-subtitle">
            Get an instant, data-backed fair market price estimate based on real transactions across India.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Step 4: Interactive Input Form Built from Metadata
# -----------------------------------------------------------------------------
cat_opts = metadata["categorical_options"]
num_defs = metadata["numeric_defaults"]
specs_lookup = metadata.get("specs_lookup", {})
brands_list = cat_opts["brand"]

# Layout tabs / cards for interactive car selection
car_col1, car_col2 = st.columns([1, 1])

with car_col1:
    selected_brand = st.selectbox(
        "Car Make / Brand",
        options=brands_list,
        index=brands_list.index("Maruti") if "Maruti" in brands_list else 0,
        help="Select the vehicle manufacturer.",
    )

brand_models = cat_opts["brand_model_map"].get(selected_brand, ["Other"])

with car_col2:
    selected_model = st.selectbox(
        "Car Model",
        options=brand_models,
        index=0,
        help="Select vehicle model (mapped automatically to the chosen make).",
    )

# Retrieve smart defaults based on the selected (brand, model) pair
default_specs = specs_lookup.get((selected_brand, selected_model), {})
default_engine = default_specs.get("engine", int(num_defs["engine"]["median"]))
default_power = default_specs.get("max_power", float(num_defs["max_power"]["median"]))
default_mileage = default_specs.get("mileage", float(num_defs["mileage"]["median"]))
default_seats = default_specs.get("seats", int(num_defs["seats"]["median"]))
default_fuel = default_specs.get("fuel_type", "Petrol")
default_trans = default_specs.get("transmission_type", "Manual")

st.markdown("##### 📝 Vehicle Specifications & History")

with st.form("car_valuation_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        # Age / Year
        vehicle_age = st.slider(
            "Vehicle Age (Years)",
            min_value=int(num_defs["vehicle_age"]["min"]),
            max_value=int(min(20, num_defs["vehicle_age"]["max"])),
            value=int(min(5, num_defs["vehicle_age"]["median"])),
            help="Age of the car in completed years (e.g. 4 years old).",
        )

        # Odometer / Kilometers Driven
        km_driven = st.number_input(
            "Kilometers Driven (Odometer)",
            min_value=100,
            max_value=350000,
            value=int(round(num_defs["km_driven"]["median"], -3)),
            step=2500,
            help="Total kilometers accumulated on the odometer.",
        )

        # Fuel Type
        fuel_opts = cat_opts["fuel_type"]
        fuel_index = fuel_opts.index(default_fuel) if default_fuel in fuel_opts else 0
        fuel_type = st.selectbox(
            "Fuel Type",
            options=fuel_opts,
            index=fuel_index,
        )

        # Transmission Type
        trans_opts = cat_opts["transmission_type"]
        trans_index = trans_opts.index(default_trans) if default_trans in trans_opts else 0
        transmission_type = st.selectbox(
            "Transmission",
            options=trans_opts,
            index=trans_index,
        )

        # Seller Type
        seller_opts = cat_opts["seller_type"]
        seller_type = st.selectbox(
            "Seller Type",
            options=seller_opts,
            index=0,
            help="Identity of the party selling the vehicle.",
        )

    with col_right:
        # Engine Capacity
        engine = st.number_input(
            "Engine Displacement (CC)",
            min_value=int(num_defs["engine"]["min"]),
            max_value=int(num_defs["engine"]["max"]),
            value=int(default_engine),
            step=50,
            help="Engine capacity in cubic centimeters (e.g. 1197 CC).",
        )

        # Max Power
        max_power = st.number_input(
            "Max Power (BHP)",
            min_value=float(round(num_defs["max_power"]["min"], 1)),
            max_value=float(round(num_defs["max_power"]["max"], 1)),
            value=float(default_power),
            step=5.0,
            help="Maximum engine power output in Brake Horsepower (bhp).",
        )

        # Mileage / Fuel Efficiency
        mileage = st.number_input(
            "Mileage (kmpl or km/kg)",
            min_value=float(round(num_defs["mileage"]["min"], 1)),
            max_value=float(round(num_defs["mileage"]["max"], 1)),
            value=float(default_mileage),
            step=0.5,
            help="Fuel efficiency rating in kilometers per liter.",
        )

        # Seating Capacity
        seats = st.selectbox(
            "Seating Capacity (Seats)",
            options=[2, 4, 5, 6, 7, 8, 9],
            index=[2, 4, 5, 6, 7, 8, 9].index(default_seats) if default_seats in [2, 4, 5, 6, 7, 8, 9] else 2,
            help="Total passenger seating capacity.",
        )

        st.caption("ℹ️ Values above are intelligently pre-filled based on typical market specs for the selected car.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Get Predicted Price", use_container_width=True)

# -----------------------------------------------------------------------------
# Step 5 & 6: Preprocessing, Prediction & Display
# -----------------------------------------------------------------------------
if submitted:
    try:
        # Build single-row DataFrame matching the exact feature list & types from training
        input_dict = {
            "brand": selected_brand,
            "model": selected_model,
            "vehicle_age": int(vehicle_age),
            "km_driven": int(km_driven),
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": float(mileage),
            "engine": int(engine),
            "max_power": float(max_power),
            "seats": int(seats),
        }

        # Build single-row dataframe in the exact column order as training
        input_df = pd.DataFrame([input_dict])[metadata["feature_names"]]

        # Predict using the loaded pipeline (preprocessing + target inversion handled internally)
        raw_prediction = pipeline.predict(input_df)[0]
        predicted_price = max(10000, float(raw_prediction))

        # Calculate residual-based price range (10th to 90th percentile bounds)
        bounds = metadata.get("residual_bounds", {"p10": -0.17, "p90": 0.20})
        p10 = bounds.get("p10", -0.17)
        p90 = bounds.get("p90", 0.20)

        lower_bound = max(10000, predicted_price * (1 + p10))
        upper_bound = predicted_price * (1 + p90)

        # Render prediction results
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">ESTIMATED MARKET VALUE</div>
                <div class="result-value">{format_inr(predicted_price)}</div>
                <div style="font-size: 1.15rem; color: #34D399; font-weight: 600; margin-bottom: 14px;">
                    {format_lakhs(predicted_price)}
                </div>
                <div class="range-box">
                    Estimated Fair Range (80% Confidence): 
                    <span class="range-highlight">{format_inr(lower_bound)}</span> — 
                    <span class="range-highlight">{format_inr(upper_bound)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Vehicle summary snapshot
        st.markdown("##### 📋 Input Summary")
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Car", f"{selected_brand} {selected_model}")
        with summary_cols[1]:
            st.metric("Age & Mileage", f"{vehicle_age} yrs • {km_driven:,} km")
        with summary_cols[2]:
            st.metric("Powertrain", f"{fuel_type} • {transmission_type}")
        with summary_cols[3]:
            st.metric("Engine & Power", f"{engine} CC • {max_power} bhp")

    except Exception as e:
        # Graceful error handling: ensure stack trace never reaches the user
        st.error(
            "An unexpected error occurred while calculating the car valuation. "
            "Please check that all inputs are valid numbers and try again.",
            icon="⚠️",
        )
