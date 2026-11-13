import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)


st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #2563EB;
        color: white;
        border: none;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


model = joblib.load('final_rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Online Shopping Purchase Prediction")
st.write(
    "Predict whether a shopping session will result in a purchase "
    "based on visitor behaviour and session information."
)


st.divider()

st.header("Enter Shopping Session Details")

st.caption(
    "Provide the details below and click Predict Purchase "
    "to generate a prediction."
)

col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input(
        "Administrative Pages Visited",
        min_value=0,
        step=1
    )

    informational = st.number_input(
        "Informational Pages Visited",
        min_value=0,
        step=1
    )

with col2:
    administrative_duration = st.number_input(
        "Time Spent on Administrative Pages (seconds)",
        min_value=0.0,
        step=1.0
    )

    informational_duration = st.number_input(
        "Time Spent on Informational Pages (seconds)",
        min_value=0.0,
        step=1.0
    )


col1, col2 = st.columns(2)

with col1:
    product_related = st.number_input(
        "Product Pages Visited",
        min_value=0,
        step=1
    )

    bounce_rates = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.4f"
    )

with col2:
    product_related_duration = st.number_input(
        "Time Spent on Product Pages (seconds)",
        min_value=0.0,
        step=1.0
    )

    exit_rates = st.number_input(
        "Exit Rate",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.4f"
    )


col1, col2 = st.columns(2)

with col1:
    page_values = st.number_input(
        "Page Value",
        min_value=0.0,
        step=1.0
    )

with col2:
    special_day = st.number_input(
        "Special Day",
        min_value=0.0,
        max_value=1.0,
        step=0.1
    )


col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(
        "Month",
        [
            "Jan", "Feb", "Mar", "Apr",
            "May", "June", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
    )

    operating_systems = st.selectbox(
        "Operating System",
        list(range(1, 9))
    )

    browser = st.selectbox(
        "Browser",
        list(range(1, 14))
    )

    region = st.selectbox(
        "Region",
        list(range(1, 10))
    )

with col2:
    traffic_type = st.selectbox(
        "Traffic Type",
        list(range(1, 21))
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        ["New_Visitor", "Other", "Returning_Visitor"]
    )

    weekend = st.selectbox(
        "Weekend",
        [False, True],
        format_func=lambda x: "Yes" if x else "No"
    )

input_data = pd.DataFrame({
    'Administrative': [administrative],
    'Administrative_Duration': [administrative_duration],
    'Informational': [informational],
    'Informational_Duration': [informational_duration],
    'ProductRelated': [product_related],
    'ProductRelated_Duration': [product_related_duration],
    'BounceRates': [bounce_rates],
    'ExitRates': [exit_rates],
    'PageValues': [page_values],
    'SpecialDay': [special_day],
    'Month': [month],
    'OperatingSystems': [operating_systems],
    'Browser': [browser],
    'Region': [region],
    'TrafficType': [traffic_type],
    'VisitorType': [visitor_type],
    'Weekend': [weekend]
})

if st.button("Predict Purchase",type="primary",use_container_width=True):

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)

    if prediction[0]:
        st.success("Purchase Prediction: Likely")
        st.write(
            "The model predicts that this shopping session "
            "is likely to result in a purchase."
        )

    else:
        st.warning("Purchase Prediction: Unlikely")
        st.write(
            "The model predicts that this shopping session "
            "is unlikely to result in a purchase."
        )