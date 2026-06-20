import shap 
import matplotlib.pyplot as plt 

def get_shap_explanation(model, x_test) : 
    # Initialize The Explainer 
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer.shap_values(x_test) 

    # Generate The Summary Plot 
    shap.summary_plot(shap_values, x_test, show=False) 
    plt.savefig('models/shap_summary.png') 
    return "SHAP summary saved to models/shap_summary.png" 
