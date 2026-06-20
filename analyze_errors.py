import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import mean_squared_error 

def perform_error_analysis(y_test, predictions) : 
    '''
    Analyzes Residuals To Identify Model Bias And Prediction Accuracy 
    '''
    # 1. Calculate Residuals 
    residuals = y_test - predictions 

    # 2. Setup The Plot 
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) 

    # Plot A : Residual Historograms 
    axes[0].hist(residuals, bins=20, color='teal', edgecolor='black') 
    axes[0].set_title('Distribution of Residuals') 
    axes[0].set_xlabel('Prediction Error') 
    axes[0].set_ylabel('Frequency') 

    # Plot B : Actual Versus Predicted 
    axes[1].scatter(y_test, predictions, color='blue', alpha=0.5) 
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
    axes[1].set_title('Actual vs Predicted') 
    axes[1].set_xlabel('Actual Values') 
    axes[1].set_ylabel('Predicted Values') 
    
    plt.tight_layout() 
    plt.show() 

    # 3. Print Summary 
    print(f"--- Error Analysis Summary ---") 
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}") 
    print(f"Mean Residual (Bias): {np.mean(residuals):.2f}") 
    print(f"Std Dev of Residuals: {np.std(residuals):.2f}") 

if __name__ == "__main__" : 
    print("This file is meant to be imported not run independently") 
    print("To run this analysis, execute tune_model.py") 