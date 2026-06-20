import pandas as pd 
import logging 

logger = logging.getLogger(__name__) 

def time_series_train_test_split(df: pd.DataFrame, test_size: float = 0.2) :  
    ''' 
    Splits The Data Chronologically Based On A Percentage 
    ''' 
    # Calculate Split Point 
    split_idx = int(len(df) * (1 - test_size)) 
    
    train = df.iloc[:split_idx] 
    test = df.iloc[split_idx:] 
    
    logger.info(f"Split data into {len(train)} training samples and {len(test)} test samples.") 
    return train, test 