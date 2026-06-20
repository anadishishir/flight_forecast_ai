import pandas as pd 
import numpy as np 
import logging 

# Setup Logging 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

def preprocess_time_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame: 
    '''
    Cleans The DataFrame, Sets The Date Index, And Handels Missing Values 
    '''
    logger.info("Starting preprocessing...") 
    
    # 1. Convert To DateTime 
    df[date_col] = pd.to_datetime(df[date_col]) 
    
    # 2. Set Index For Time Series 
    df.set_index(date_col, inplace=True) 
    df.sort_index(inplace=True) 

    # 3. Handle Missing Values 
    if df[target_col].isnull().sum() > 0 : 
        logger.warning(f"Found {df[target_col].isnull().sum()} missing values. Interpolating...") 
        df[target_col] = df[target_col].interpolate(method='linear') 

    return df 

def add_time_features(df: pd.DataFrame) -> pd.DataFrame: 
    '''
    Extracts Useful Features From The DateTime Index 
    '''
    df = df.copy() 
    df['month'] = df.index.month 
    df['day_of_week'] = df.index.dayofweek 
    df['quarter'] = df.index.quarter 
    
    logger.info("Time features added successfully.") 
    return df 

def create_lags(df: pd.DataFrame, target_col: str, lags: int = 3) : 
    df_lagged = df.copy() 
    for i in range(1, lags + 1) : 
        df_lagged[f'lag_{i}'] = df_lagged[target_col].shift(i) 
    return df_lagged.dropna() 