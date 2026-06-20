import pandas as pd 
import logging 

# Setting Up Logging So We Can Track What The App Is Doing 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

def load_data(url: str) -> pd.DataFrame : 
    ''' 
    The Function Fetches Data From URL And Returns A DataFrame 
    ''' 
    try : 
        logger.info(f"Attempting to load data from : {url}") 
        df = pd.read_csv(url) 
        
        if df.empty : 
            logger.warning("The loaded dataframe is empty!")  
            return None 
            
        logger.info(f"Successfully loaded data with shape: {df.shape}") 
        return df 
    
    except Exception as e : 
        logger.error(f"Error loading data: {e}") 
        return None 