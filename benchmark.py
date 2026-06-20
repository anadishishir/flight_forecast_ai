from src.ingestion.data_loader import load_data 
from src.preprocessing.preprocessor import preprocess_time_series, create_lags 
from src.modelling.trainer import train_arima, train_random_forest 
from sklearn.metrics import mean_squared_error 
import numpy as np 

# 1. Load And Preprocess 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 
df = load_data(url) 
df = preprocess_time_series(df, date_col='Month', target_col='Passengers') 

# 2. Prepare Data For Machine Learning (Creating Lags) 
df_lagged = create_lags(df, target_col='Passengers', lags=3) 

# Split For Random Forest (Using The Lagged Version) 
split = int(len(df_lagged) * 0.8) 
train_ml = df_lagged.iloc[:split] 
test_ml = df_lagged.iloc[split:] 

x_train, y_train = train_ml.drop('Passengers', axis=1), train_ml['Passengers'] 
x_test, y_test = test_ml.drop('Passengers', axis=1), test_ml['Passengers'] 

# 3. Train And Predict 
# ARIMA (Uses The Original Series) 
arima_model = train_arima(df['Passengers'].iloc[:int(len(df)*0.8)]) 
arima_pred = arima_model.forecast(steps=len(df) - split)  

# Random Forest 
rf_model = train_random_forest(x_train, y_train) 
rf_pred = rf_model.predict(x_test) 

# 4. Compare
arima_rmse = np.sqrt(mean_squared_error(df['Passengers'].iloc[split:], arima_pred)) 
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred)) 

print(f"ARIMA RMSE: {arima_rmse:.2f}")  
print(f"Random Forest RMSE: {rf_rmse:.2f}")  