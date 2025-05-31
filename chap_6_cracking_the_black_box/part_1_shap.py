import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aquarel
from aquarel import load_theme

# SHAP (SHapley Additive exPlanations)
# https://medium.com/biased-algorithms/shap-values-explained-08764ab16466
# https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretabi
# https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
# or see AI fund. course
import shap
import shap.plots._bar as Bar
import shap.plots._waterfall as Waterfall

from main_config import (
    best_nn_model,
    COLORS_TERMINAL,
    COLORS_BOLD,
    full_dataset_cleaned_no_outliers_path,
    RANDOM_SEED,
    shap_results_waterfall_path,
    shap_results_bar_path,
    shap_results_beeswarm_path,
)

from sklearn.model_selection import train_test_split

def main():
    # loading the best model
    # https://github.com/keras-team/keras/issues/1933
    model = tf.keras.models.load_model(
        best_nn_model,
        custom_objects={
            "mse":'mse', 
            "mae":'mae'
        }
    )
    
    print(f"{COLORS_BOLD['BOLD_GREEN']}Best model loaded successfully!{COLORS_TERMINAL['RESET_COLOR']}")

    # loading the full dataset
    df_full_dataset = pd.read_csv(full_dataset_cleaned_no_outliers_path)
    print(f"{COLORS_BOLD['BOLD_GREEN']}Full dataset loaded successfully!{COLORS_TERMINAL['RESET_COLOR']}")
    
    # preparing the data
    X_TRAIN_FULL , X_TEST , Y_TRAIN_FULL , Y_TEST = train_test_split(
        df_full_dataset.drop(columns=["Exam_Score"]).copy(), 
        df_full_dataset["Exam_Score"], 
        random_state=RANDOM_SEED
    )

    X_TEST = pd.DataFrame(X_TEST, columns=df_full_dataset.drop(columns=["Exam_Score"]).columns)

    print(f"{COLORS_BOLD['BOLD_GREEN']}Data prepared successfully!{COLORS_TERMINAL['RESET_COLOR']}")
    
    sample_size = 500 
    X_sample = X_TEST.sample(min(sample_size, len(X_TEST)), random_state=RANDOM_SEED)
    # initializing the SHAP explainer
    explainer = shap.Explainer(model, X_sample)
    print(f"{COLORS_BOLD['BOLD_GREEN']}SHAP explainer initialized successfully!{COLORS_TERMINAL['RESET_COLOR']}")
    
    # calculating SHAP values
    shap_values = explainer.shap_values(X_sample)
    print(f"{COLORS_BOLD['BOLD_GREEN']}SHAP values calculated successfully!{COLORS_TERMINAL['RESET_COLOR']}")
        
    theme = load_theme("umbra_dark")
    theme.apply()

    plt.title("SHAP Summary Plot")

    shap.summary_plot(shap_values, X_sample,show=False,axis_color="#ffffff")

    plt.savefig(
        shap_results_beeswarm_path,
        format="pdf"
    )
    print(f"{COLORS_BOLD['BOLD_GREEN']}SHAP beeswarm plot saved successfully!{COLORS_TERMINAL['RESET_COLOR']}")

    
    print(f"{COLORS_BOLD['BOLD_GREEN']}SHAP analysis completed successfully!{COLORS_TERMINAL['RESET_COLOR']}")

