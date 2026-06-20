from src.ingestion.data_loader import load_data 

# Example URL (A Public Dataset For Airline Passenger) 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" 

df = load_data(url) 

if df is not None : 
    print("Data loaded successfully!") 
    print(df.head())   