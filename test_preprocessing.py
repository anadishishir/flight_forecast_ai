from src.ingestion.data_loader import load_data 
from src.preprocessing.preprocessor import preprocess_time_series, add_time_features 

# Load Data 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 
df = load_data(url) 

# Run Preprocessing 
df_clean = preprocess_time_series(df, date_col='Month', target_col='Passengers') 
df_final = add_time_features(df_clean) 

print(df_final.head()) 