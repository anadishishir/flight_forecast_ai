Airline Passenger Forecasting System 
A professional-grade, end-to-end Machine Learning pipeline for time-series forecasting. This project utilizes a Random Forest regressor, integrated with Generative AI for business insights and SHAP for model explainability and deployed on AWS. 
    
Project Architecture 
The system is built on a modular, production-ready microservices architecture : 
 
Data Pipeline : Automated ingestion and feature engineering (lag creation). 
 
Modeling : Scikit-learn Random Forest with hyperparameter tuning. 
 
Explainability : SHAP integration for transparent feature importance. 
 
API Layer : FastAPI backend that serves predictions and generates strategic business insights using the Gemini API. 
 
Frontend : An interactive Streamlit dashboard for real-time forecasting.
 
Deployment : Fully containerized with Docker. 
 
Key Features 
Predictive Analytics: Accurate passenger forecasting using historical lag data.
 
Explainable AI (XAI) : Visual insights into model decision-making via SHAP plots.
 
Generative Insights : Automated business recommendations derived from predictions via Google Gemini. 
 
Security : Secure environment management using .env variables and .gitignore. 
 
Tech Stack 
Language: Python 3.12 
 
ML Frameworks : Scikit-learn, SHAP 

API : FastAPI, Uvicorn 

GenAI: Google Generative AI (Gemini 1.5-flash) 
 
UI : Streamlit 
 
Deployment & Production : 
This project is built for professional scale.

1. Local Development
For quick local testing and development:

Run Backend: uvicorn api.main:app --reload

Run Frontend: streamlit run app.py

2. Cloud Deployment (AWS)
This project is fully containerized, making it production-ready for AWS App Runner, ECS, or EC2.

Containerization: The Dockerfile packages the application, dependencies, and model artifacts into a single portable image.

Secret Management: For security, the GEMINI_API_KEY is never included in the image. It must be injected as an Environment Variable within your AWS Service configuration (e.g., via AWS App Runner "Environment Variables" or ECS Task Definitions).

Workflow: 1. Build image: docker build -t airline-forecaster .
2. Push to Amazon ECR.
3. Deploy to your chosen AWS compute service. 

Quick Start 
Clone the repo : 

Bash 
git clone https://github.com/anadishishir/airplane_forecasting_system   
cd time_series_project 
Install dependencies : 
 
Bash 
pip install -r requirements.txt 
Configure API : 
Create a .env file and add your key: GEMINI_API_KEY=your_key_here 
 
Run the pipeline : 
 
Bash 
python tune_model.py 
Start the services : 

Terminal 1: uvicorn api.main:app --reload 
 
Terminal 2: streamlit run app.py 
 
Project Structure 
Plaintext 
time_series_project/ 
├── api/             # FastAPI backend + Gemini integration 
├── models/          # Saved model (.pkl) and SHAP plots (.png) 
├── src/             # Modular ingestion, preprocessing, and tuning logic 
├── app.py           # Streamlit dashboard 
├── tune_model.py    # Main training and evaluation pipeline 
└── Dockerfile       # Containerization configuration 