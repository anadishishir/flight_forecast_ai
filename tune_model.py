import numpy as np 
import joblib 
import os 
from sklearn.metrics import mean_squared_error 
import shap 
import matplotlib.pyplot as plt 

# Imports From Project Structur 
from src.ingestion.data_loader import load_data 
from src.preprocessing.preprocessor import preprocess_time_series, create_lags 
from src.modelling.tuner import tune_random_forest 
from analyze_errors import perform_error_analysis  
import numpy as np 

# 1. Pipeline Load And Prepare 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 
df = preprocess_time_series(load_data(url), date_col='Month', target_col='Passengers') 
df_lagged = create_lags(df, target_col='Passengers', lags=3) 

# 2. Split And Tune 
# Split 
split = int(len(df_lagged) * 0.8)  
train, test = df_lagged.iloc[:split], df_lagged.iloc[split:]  
x_train, y_train = train.drop('Passengers', axis=1), train['Passengers']  
x_test, y_test = test.drop('Passengers', axis=1), test['Passengers']  

# Tune  
print("Running Grid Search for optimal parameters...") 
best_model = tune_random_forest(x_train, y_train) 

# 3. Final Evaluation 
predictions = best_model.predict(x_test) 
perform_error_analysis(y_test, predictions) 

# 4. Save Model 
if not os.path.exists('models') : 
    os.makedirs('models') 
joblib.dump(best_model, 'models/random_forest.pkl') 
print("Pipeline complete. Model saved.")  

# 5. Creating The Explainer 
explainer = shap.TreeExplainer(best_model) 
shap_values = explainer.shap_values(x_test)  

# 6. Generate And Save The Plot 
plt.figure() 
shap.summary_plot(shap_values, x_test, show=False) 
plt.savefig('models/shap_summary.png') 
print("SHAP summary saved to models/shap_summary.png") 