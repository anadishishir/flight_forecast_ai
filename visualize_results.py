import matplotlib.pyplot as plt 
import numpy as np 
from src.ingestion.data_loader import load_data 
from src.preprocessing.preprocessor import preprocess_time_series, create_lags 
from src.modelling.tuner import tune_random_forest 

# 1. Load And Prepare Data 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 
df = preprocess_time_series(load_data(url), date_col='Month', target_col='Passengers') 
df_lagged = create_lags(df, target_col='Passengers', lags=3) 

# 2. Split 
split = int(len(df_lagged) * 0.8)  
train, test = df_lagged.iloc[:split], df_lagged.iloc[split:]  
x_train, y_train = train.drop('Passengers', axis=1), train['Passengers']  
x_test, y_test = test.drop('Passengers', axis=1), test['Passengers']  

# 3. Get Best Model 
best_model = tune_random_forest(x_train, y_train) 
predictions = best_model.predict(x_test) 

# 4. Plotting 
plt.figure(figsize=(12, 6)) 
plt.plot(test.index, y_test, label='Actual Passengers', marker='o') 
plt.plot(test.index, predictions, label='Random Forest Predictions', color='red', linestyle='--') 
plt.title('Actual vs Predicted Airline Passengers (Tuned Random Forest)') 
plt.xlabel('Date') 
plt.ylabel('Passengers') 
plt.legend() 
plt.grid(True) 
plt.savefig('models/results_visualization.png') 
print("Visualization saved to models/results_visualization.png") 