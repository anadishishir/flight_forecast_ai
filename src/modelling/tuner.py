from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_random_forest(x_train, y_train) : 
    param_grid = { 
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20], 
        'min_samples_split': [2, 5] 
    } 
    
    rf = RandomForestRegressor(random_state=42) 
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error') 
    grid_search.fit(x_train, y_train) 
    
    print(f"Best Parameters: {grid_search.best_params_}") 
    return grid_search.best_estimator_ 