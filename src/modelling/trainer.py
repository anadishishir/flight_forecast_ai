from statsmodels.tsa.arima.model import ARIMA 
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd 

def train_arima(train_data: pd.Series, order=(5, 1, 0)) -> any : 
    model = ARIMA(train_data, order=order) 
    return model.fit() 

def train_random_forest(x_train: pd.DataFrame, y_train : pd.Series) -> any : 
    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(x_train, y_train) 
    return model 

