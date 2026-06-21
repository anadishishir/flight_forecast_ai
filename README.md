# Flight Forecast AI   
 
A professional-grade, end-to-end Machine Learning pipeline for time-series forecasting. This project utilizes a **Random Forest regressor** for predictive modeling, integrated with **Generative AI (Gemini)** for strategic business insights and **SHAP** for model explainability. 

##  Live Demo 
https://flight-forecast-ui.onrender.com 

---

##  System Architecture 
This project is built on a **decoupled, service-oriented architecture**, separating the frontend, backend, and AI integration for better scalability and maintainability: 
 
* **Data Pipeline:** Automated ingestion and feature engineering (lag creation). 
* **Modeling:** Scikit-learn Random Forest with hyperparameter tuning. 
* **Explainability:** SHAP integration for transparent feature importance. 
* **API Layer (Backend):** FastAPI backend that serves predictions and generates strategic business insights using the Gemini API. 
* **Frontend (UI):** An interactive Streamlit dashboard for real-time forecasting. 
 
 
 
--- 
 
##  Tech Stack 
* **Language:** Python 3.12 
* **ML Frameworks:** Scikit-learn, SHAP 
* **API Framework:** FastAPI, Uvicorn 
* **GenAI:** Google Generative AI (Gemini 1.5-flash) 
* **UI:** Streamlit 
* **Deployment:** Render (Automated CI/CD) 
 
--- 
 
##  Deployment Overview 
This application is deployed as a two-service architecture on **Render**, ensuring modularity and independent scaling. 
 
| Service | Responsibility | 
| :--- | :--- | 
| **Backend (FastAPI)** | Serves model predictions & interacts with the Gemini API. | 
| **Frontend (Streamlit)** | Provides an interactive dashboard for user forecasting. | 
 
### Deployment Steps 
1.  **Backend:** Deploy as a "Web Service" on Render. Set the Start Command to `uvicorn main:app --host 0.0.0.0 --port $PORT`. Inject `GEMINI_API_KEY` via Environment Variables. 
2.  **Frontend:** Deploy as a separate "Web Service" on Render. Set the Start Command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. Set the `API_BASE_URL` Environment Variable to your live backend URL. 
 
--- 
 
##  Quick Start 
**Clone the repo:** 
```bash 
git clone https://github.com/anadishisir/flight_forecast_ai.git 
cd flight_forecast_ai
Install dependencies : 
 
Bash 
pip install -r requirements.txt 
Run Locally : 
 
Set your GEMINI_API_KEY in your environment. 
 
Start the Backend: uvicorn main:app --reload 
 
Start the Frontend: streamlit run app.py 
 
Project Structure 
Plaintext 
flightfirecastai/ 
├── api/             # FastAPI backend + Gemini integration 
├── models/          # Saved model (.pkl) and SHAP artifacts 
├── src/             # Modular ingestion, preprocessing, and tuning logic 
├── app.py           # Streamlit dashboard 
├── tune_model.py    # Main training and evaluation pipeline 
└── requirements.txt # Project dependencies 