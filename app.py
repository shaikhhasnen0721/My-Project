import streamlit as st
import numpy as np
import pickle
import json

# ----------------------------
# LOAD MODEL AND COLUMNS
# ----------------------------
@st.cache_resource
def load_model():
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
    return model, data_columns

model, data_columns = load_model()
locations = [col for col in data_columns if col not in ['sqft', 'bhk', 'price'] and not col.startswith("area_")]


# ----------------------------
# PRICE PREDICTION FUNCTION
# ----------------------------
def predict_price(location, sqft, bhk, area_type):

    # Create empty row
    x = np.zeros(len(data_columns))

    # numeric fields
    x[data_columns.index("total_sqft")] = sqft
    x[data_columns.index("bhk")] = bhk

    # handle location one-hot
    location = location.lower()
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # handle Urban / Rural
    area_col = f"area_{area_type.lower()}"
    if area_col in data_columns:
        idx = data_columns.index(area_col)
        x[idx] = 1

    return round(model.predict([x])[0], 2)


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🏠 House Price Predictor")
st.write("Enter details below to estimate property price (in Lakhs).")

# Inputs
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)
bhk = st.selectbox("BHK", [1,2,3,4,5,6,7,8,9,10])
location = st.selectbox("Location", sorted(locations))
area_type = st.radio("Area Type", ['urban', 'rural'])

if st.button("Predict Price"):
    price = predict_price(location, sqft, bhk, area_type)
    st.success(f"🏡 **Estimated Price: ₹ {price} Lakhs**")