import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model + training columns
# -----------------------------
model = joblib.load("rf_tuned.pkl")
train_cols = joblib.load("train_columns.pkl")

# -----------------------------
# Load dataset for dropdown options
# -----------------------------
df_options = pd.read_csv("used_cars_options_cleaned.csv")

# Clean text columns for stable dropdowns
for c in ["brand", "model", "fuel_type", "accident", "clean_title"]:
    if c in df_options.columns:
        df_options[c] = df_options[c].astype(str).str.strip()

# Dropdown lists
brand_options = sorted(df_options["brand"].dropna().unique().tolist())
default_brand = "Toyota" if "Toyota" in brand_options else brand_options[0]

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write("Select car details below and the app will estimate the used car price (USD).")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_options, index=brand_options.index(default_brand))

    models_for_brand = sorted(
        df_options.loc[df_options["brand"] == brand, "model"].dropna().unique().tolist()
    )
    if not models_for_brand:
        models_for_brand = ["Unknown"]

    model_name = st.selectbox("Model", models_for_brand)

    model_year = st.number_input("Model Year", min_value=1970, max_value=2025, value=2018)
    milage = st.number_input("Mileage (miles)", min_value=0, max_value=500000, value=60000)

with col2:
    if "fuel_type" in df_options.columns:
        fuel_options = sorted(df_options["fuel_type"].dropna().unique().tolist())
        if not fuel_options:
            fuel_options = ["Gasoline", "Hybrid", "Diesel", "Electric", "Other"]
    else:
        fuel_options = ["Gasoline", "Hybrid", "Diesel", "Electric", "Other"]

    fuel_type = st.selectbox("Fuel Type", fuel_options)

    engine_liters = st.number_input(
        "Engine Size (L)", min_value=0.5, max_value=10.0, value=2.5, step=0.1
    )
    engine_cylinders = st.number_input(
        "Engine Cylinders", min_value=2, max_value=16, value=4, step=1
    )

    transmission_group = st.selectbox(
        "Transmission Group", ["Automatic", "Manual", "CVT", "Other", "Unknown"]
    )

st.subheader("Condition / Listing Info")

col3, col4 = st.columns(2)

with col3:
    ext_color_group = st.selectbox(
        "Exterior Color Group",
        ["Black", "White", "Silver", "Gray", "Blue", "Red", "Green", "Brown", "Beige",
         "Yellow", "Orange", "Purple", "Other", "Unknown"]
    )
    int_color_group = st.selectbox(
        "Interior Color Group",
        ["Black", "White", "Silver", "Gray", "Blue", "Red", "Green", "Brown", "Beige",
         "Yellow", "Orange", "Purple", "Other", "Unknown"]
    )

with col4:
    if "accident" in df_options.columns:
        accident_options = sorted(df_options["accident"].dropna().unique().tolist())
        if not accident_options:
            accident_options = ["None reported", "At least 1 accident or damage reported", "Unknown"]
    else:
        accident_options = ["None reported", "At least 1 accident or damage reported", "Unknown"]

    accident = st.selectbox("Accident History", accident_options)

    if "clean_title" in df_options.columns:
        clean_title_options = sorted(df_options["clean_title"].dropna().unique().tolist())
        if not clean_title_options:
            clean_title_options = ["Yes", "No", "Unknown"]
    else:
        clean_title_options = ["Yes", "No", "Unknown"]

    clean_title = st.selectbox("Clean Title", clean_title_options)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Price"):
    new_car = {
        "brand": str(brand).strip(),
        "model": str(model_name).strip(),
        "model_year": int(model_year),
        "milage": float(milage),
        "fuel_type": str(fuel_type).strip(),
        "engine_liters": float(engine_liters),
        "engine_cylinders": float(engine_cylinders),
        "transmission_group": str(transmission_group).strip(),
        "ext_color_group": str(ext_color_group).strip(),
        "int_color_group": str(int_color_group).strip(),
        "accident": str(accident).strip(),
        "clean_title": str(clean_title).strip(),
    }

    new_df = pd.DataFrame([new_car])

    # One-hot encode new input (DO NOT drop_first for single-row input)
    new_encoded = pd.get_dummies(new_df, drop_first=False)

    # Align columns to match training
    new_aligned = new_encoded.reindex(columns=train_cols, fill_value=0)

    # Predict
    pred_price = model.predict(new_aligned)[0]

    st.success(f"Estimated Price: **${pred_price:,.2f}**")