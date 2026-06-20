from src.ingestion.data_loader import load_data 
from src.preprocessing.preprocessor import preprocess_time_series 
from src.modelling.splitter import time_series_train_test_split 

# 1. Pipeline Execution 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 
df = load_data(url) 
df_clean = preprocess_time_series(df, date_col='Month', target_col='Passengers') 

# 2. Perform The Split 
train, test = time_series_train_test_split(df_clean, test_size=0.2) 

print(f"Training set shape: {train.shape}") 
print(f"Test set shape: {test.shape}") 
print("--- Training Head ---")  
print(train.head()) 