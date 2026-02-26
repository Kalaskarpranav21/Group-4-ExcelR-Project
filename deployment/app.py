import streamlit as st
import pandas as pd
import pickle

# ----------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# ----------------------------------------------------
# We use @st.cache_resource so the model loads only once when the app starts
@st.cache_resource
def load_model():
    with open('rf_churn_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# ----------------------------------------------------
# 2. BUILD THE USER INTERFACE (Header)
# ----------------------------------------------------
st.title("📡 Telecom Customer Churn Predictor")
st.write("Enter the customer's usage details below to predict if they are at risk of churning.")

# ----------------------------------------------------
# 3. COLLECT USER INPUTS
# ----------------------------------------------------
st.sidebar.header("Customer Details")

def user_input_features():
    # Categorical Inputs
    intl_plan = st.sidebar.selectbox("International Plan", ("No", "Yes"))
    vm_plan = st.sidebar.selectbox("Voice Mail Plan", ("No", "Yes"))
    
    # Numerical Inputs
    account_length = st.sidebar.number_input("Account Length (days)", min_value=1, max_value=250, value=100)
    customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
    
    # Usage Minutes 
    st.sidebar.subheader("Usage (Minutes)")
    day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, max_value=400.0, value=150.0)
    eve_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
    night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
    intl_mins = st.sidebar.number_input("International Minutes", min_value=0.0, max_value=25.0, value=10.0)
    
    # Usage Calls
    st.sidebar.subheader("Usage (Calls)")
    day_calls = st.sidebar.number_input("Day Calls", min_value=0, max_value=500, value=100)
    eve_calls = st.sidebar.number_input("Evening Calls", min_value=0, max_value=500, value=100)
    night_calls = st.sidebar.number_input("Night Calls", min_value=0, max_value=500, value=100)
    intl_calls = st.sidebar.number_input("International Calls", min_value=0, max_value=25, value=4)

    # Convert binary inputs back to 0 and 1 for the model
    intl_plan_encoded = 1 if intl_plan == "Yes" else 0
    vm_plan_encoded = 1 if vm_plan == "Yes" else 0

    # MUST EXACTLY MATCH THE COLUMNS THE MODEL WAS TRAINED ON!
    # Removed voice_mail_messages and total_charge since your EDA script dropped them
    data = {
        'account_length': account_length,
        'voice_mail_plan': vm_plan_encoded,
        'day_mins': day_mins,
        'evening_mins': eve_mins,
        'night_mins': night_mins,
        'international_mins': intl_mins,
        'customer_service_calls': customer_service_calls,
        'international_plan': intl_plan_encoded,
        'day_calls': day_calls,
        'evening_calls': eve_calls,
        'night_calls': night_calls,
        'international_calls': intl_calls
    }
    
    # Convert dictionary into a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Store the user's input
input_df = user_input_features()

# Display the user's input on the main screen
st.subheader("Customer Input Summary")
st.write(input_df)

# ----------------------------------------------------
# 4. MAKE PREDICTIONS
# ----------------------------------------------------
# Create a button to trigger the prediction
if st.button("Predict Churn Risk"):
    
    # Predict 0 (No Churn) or 1 (Churn)
    prediction = model.predict(input_df)
    
    # Grab the probability percentage
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ HIGH RISK: This customer is likely to CHURN.")
        st.write(f"Probability of churning: **{prediction_proba[0][1]:.2%}**")
    else:
        st.success(f"✅ LOW RISK: This customer is likely to STAY.")
        st.write(f"Probability of staying: **{prediction_proba[0][0]:.2%}**")