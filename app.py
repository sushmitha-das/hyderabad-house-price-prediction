import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL AND DATASET
# ==============================

model = joblib.load("house_price_model.pkl")
df = pd.read_csv("hyderabad_house_price_dataset.csv")


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ==============================
# PINK THEME
# ==============================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff0f6, #ffe4ef);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #c2185b;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #7a284b;
    margin-bottom: 30px;
}

h2, h3 {
    color: #c2185b !important;
}

.stButton > button {
    background-color: #e91e63;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #c2185b;
    color: white;
}

.result {
    background-color: #ffd6e7;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #f48fb1;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================

st.markdown(
    '<div class="title">🏠 Hyderabad House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the estimated price of a house using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==============================
# PROJECT INFORMATION
# ==============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Dataset Size", "500 Houses")

with col2:
    st.metric("🤖 ML Algorithm", "Random Forest")

with col3:
    st.metric("🎯 R² Score", "94.94%")


st.divider()


# ==============================
# HOUSE DETAILS
# ==============================

st.subheader("🏡 Enter House Details")

col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "📐 Area (sqft)",
        min_value=300,
        max_value=5000,
        value=1200
    )

    bedrooms = st.number_input(
        "🛏️ Number of Bedrooms",
        min_value=1,
        max_value=6,
        value=2
    )

    bathrooms = st.number_input(
        "🛁 Number of Bathrooms",
        min_value=1,
        max_value=6,
        value=2
    )

    maintenance = st.number_input(
        "🧹 Maintenance Staff",
        min_value=0,
        max_value=3,
        value=1
    )


with col2:

    security = st.selectbox(
        "🔐 24×7 Security",
        ["Yes", "No"]
    )

    parking = st.number_input(
        "🚗 Parking Spaces",
        min_value=0,
        max_value=4,
        value=1
    )

    age = st.number_input(
        "🏠 House Age (years)",
        min_value=0,
        max_value=50,
        value=5
    )

    location = st.selectbox(
        "📍 Location",
        sorted(df["location"].unique())
    )


st.divider()


# ==============================
# PREDICTION
# ==============================

if st.button("🔮 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{

        "area_sqft": area,

        "bedrooms": bedrooms,

        "bathrooms": bathrooms,

        "maintenance_staff": maintenance,

        "security_24x7": 1 if security == "Yes" else 0,

        "parking": parking,

        "house_age_years": age,

        "location": location

    }])


    prediction = model.predict(input_data)[0]


    st.success("Prediction generated successfully!")


    st.markdown(
        f"""
        <div class="result">

        <h2>🏠 Estimated House Price</h2>

        <h1>₹{prediction:.2f} Lakhs</h1>

        <p>📍 Location: {location}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================
# DATA VISUALIZATION
# ==============================

st.divider()

st.subheader("📊 Hyderabad House Price Insights")


col1, col2 = st.columns(2)


with col1:

    st.write("### 📐 Area vs Price")

    st.scatter_chart(
        df,
        x="area_sqft",
        y="price_in_lakhs"
    )


with col2:

    st.write("### 📍 Average Price by Location")

    location_prices = (
        df.groupby("location")["price_in_lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(location_prices)


# ==============================
# HOW THE MODEL WORKS
# ==============================

st.divider()

st.subheader("🧠 How Does the Model Work?")


st.write("""
This project uses a Random Forest Regression algorithm to
predict house prices in Hyderabad.

The model considers the following factors:

• 📐 Area of the house
• 🛏️ Number of bedrooms
• 🛁 Number of bathrooms
• 🧹 Maintenance staff
• 🔐 24×7 security
• 🚗 Parking spaces
• 🏠 House age
• 📍 Location

These features are processed and given to the trained
Machine Learning model, which produces an estimated
house price.
""")


# ==============================
# MODEL PERFORMANCE
# ==============================

st.info(
    "🎯 Model Performance: R² Score = 94.94% | "
    "MAE = ₹16.73 Lakhs"
)


# ==============================
# NOTE
# ==============================

st.divider()

st.caption(
    "Note: This project uses a synthetic dataset created for academic purposes."
)