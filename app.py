import streamlit as st 
import requests 
from PIL import Image 
import os 

# Set Page Configuration 
st.set_page_config(page_title="Airline Forecaster", layout="wide") 

st.title("Airline Passenger Forecaster")
st.write("Enter the last 3 months of data to predict next month's passengers:") 

# 1. Inputs For The Three Lags 
st.sidebar.header("Input Features") 
lag1 = st.number_input("Passengers 3 months ago (Lag 1):", value=100.0) 
lag2 = st.number_input("Passengers 2 months ago (Lag 2):", value=110.0) 
lag3 = st.number_input("Passengers 1 month ago (Lag 3):", value=120.0) 

# 2. Main Prediction Logic 
if st.sidebar.button("Predict") : 
    payload = {"lags": [lag1, lag2, lag3]} 
    
    try: 
        # Requsting Prediction And Analysis From FastAPI Backend 
        response = requests.post("https://flight-forecast-api.onrender.com/predict", json=payload) 
        
        if response.status_code == 200 :
            data = response.json() 
            
            # Displaying Prediction 
            st.subheader("Results") 
            st.metric("Predicted Passengers", f"{data['prediction']:.2f}") 
            
            # Displaying Gemini Insight 
            st.info(f"**AI Business Insight:** {data['analysis']}") 
        else : 
            st.error("Error connecting to the Prediction API.") 
            
    except Exception as e : 
        st.error(f"Could not connect to API: {e}") 

# 3. SHAP Explanation Section 
st.divider() 
if st.checkbox("Show Model Explanation (SHAP)") : 
    try : 
        image_path = 'models/shap_summary.png' 
        if os.path.exists(image_path) : 
            image = Image.open(image_path) 
            st.image(image, caption='Feature Importance: Which lags drive this forecast?') 
        else : 
            st.warning("SHAP plot not found. Run tune_model.py first.") 
    except Exception as e : 
        st.error(f"Could not load SHAP plot: {e}") 