import os 
from fastapi import FastAPI 
from pydantic import BaseModel 
import joblib  
import numpy as np 
import google.generativeai as genai 
from dotenv import load_dotenv 
import logging 

# Configuring Logging 
logging.basicConfig( 
    filename='api_usage.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s' 
) 

# 1. Setup Enviroment & Gemini 
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY") 
if not api_key : 
    print("DEBUG ERROR : GEMINI_API_KEY was not found in .env or enviroment variables !!!") 
else : 
    print(f"DEBUG : API Key loaded successfully ...") 

genai.configure(api_key=api_key) 

gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite') 
 
# 2. Initialize FastAPI 
app = FastAPI() 

# 3. Loading Model 
model = joblib.load('models/random_forest.pkl') 

class PredictionRequest(BaseModel) : 
    lags: list[float] 

@app.post("/predict") 
def get_prediction(request: PredictionRequest) :  
    logging.info(f"Prediction requested with lags: {request.lags}") 
    data = np.array(request.lags).reshape(1, -1) 
    raw_pred = float(model.predict(data)[0]) 

    # Generate Bussiness Insight With Gemini 
    prompt = (
        f"The predictive model for airline passengers forecasts {raw_pred:.2f} passengers "
        "for the upcoming period. Based on this number, provide a professional, "
        "one-sentence strategic recommendation for airline resource management."
    ) 
    
    try : 
        response = gemini_model.generate_content(prompt) 
        analysis = response.text 
    except Exception as e : 
        analysis = "Insight generation currently unavailable." 

    logging.info(f"Prediction successful: {raw_pred}")     

    return {
        "prediction": raw_pred, 
        "analysis": analysis 
    } 

@app.get("/") 
def read_root() : 
    return {"message": "Forecasting API is live and connected to Gemini."} 