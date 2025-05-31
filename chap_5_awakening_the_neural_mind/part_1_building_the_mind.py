# based upon the previous results we will fine tune the best models
from chap_5_awakening_the_neural_mind.config import X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
from main_config import (
    report_assets_path ,
    COLORS_TERMINAL,    
    COLORS_BOLD,
    models_dir,
    filename_results_best_models_csv,
    filename_results_best_models_pdf , 
    RANDOM_SEED,
    print_cursor,
    type_out,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV , cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
import pandas as pd
from math import ceil
import os
import time
from pprint import pprint
import pickle

def enter_model_name(default_prefix: str = "ver1_"):
    while True:
        type_out(f"Enter model prefix (or press enter to stick to default ({default_prefix}))",delay=0.02)
        model_prefix = print_cursor()
        if len(model_prefix) == 0:
            model_prefix = default_prefix

        if not model_prefix.endswith("_"):
            model_prefix += "_"

        model_prefix = model_prefix.replace(" ", "_")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(os.path.join(models_dir,model_prefix)):
            os.makedirs(os.path.join(models_dir,model_prefix))
            return model_prefix   

        else:
            print(f"{COLORS_TERMINAL['RED']}Directory already exists: {models_dir}{COLORS_TERMINAL['RESET_COLOR']}")
            print(f"{COLORS_TERMINAL['RED']}Models will be overwritten if they already exist.{COLORS_TERMINAL['RESET_COLOR']}")
            type_out(f"Do you want to overwrite the models in {os.path.join(models_dir,model_prefix)}? (y/n) ", delay=0.02)
            overwrite = print_cursor().strip().lower()
            
            if overwrite in ["y", "yes"]:
                print(f"{COLORS_BOLD['BOLD_GREEN']}Overwriting models in {models_dir}{model_prefix}.{COLORS_TERMINAL['RESET_COLOR']}")
                return model_prefix
            

def main():
    models = {}
    results = {}

    model_prefix = enter_model_name(default_prefix="nn1_")

    start_training = time.time()
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        activation='relu',          
        random_state=RANDOM_SEED,
        solver="lbfgs"
    ).fit(X_TRAIN, Y_TRAIN)            
    end_training = time.time()

    test_score = mlp_regressor.score(X_TEST, Y_TEST)
    y_pred = mlp_regressor.predict(X_TEST)
    results["basic_MLP"] = {
        "cv_mean_r2"    : np.mean(cross_val_score(mlp_regressor, X_TRAIN, Y_TRAIN, cv=10, scoring="r2", n_jobs=-1)),
        "cv_std_r2"         : np.std(cross_val_score(mlp_regressor, X_TRAIN, Y_TRAIN, cv=10, scoring="r2", n_jobs=-1)),
        "accuracy"      : test_score,
        "R2"            : r2_score(Y_TEST, y_pred), 
        "MSE"           : mean_squared_error(Y_TEST, y_pred),
        "RMSE"          : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
        "MAE"           : np.mean(np.abs(Y_TEST - y_pred)),
        "training_time" : end_training - start_training,
        "best_params"   : None ,
        "epochs"        : mlp_regressor.n_iter_,
    }        
    models["basic_MLP"] = mlp_regressor

    start_training = time.time()

    searchCV = GridSearchCV(
        MLPRegressor(random_state=RANDOM_SEED), 
        cv=10,
        param_grid={
            'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 32), (128, 64)],
            'activation': ['relu', 'tanh'],
            'solver': ['lbfgs'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [500, 550, 600]
        }
    ).fit(X_TRAIN,Y_TRAIN)
    
    end_training = time.time()

    test_score = searchCV.score(X_TEST, Y_TEST)
    y_pred = np.array(searchCV.predict(X_TEST))
    
    searchCV.best_params_.pop("regressor", None) 
    results["gridsearch_MLP"] = {
        "cv_mean_r2"    : np.mean(searchCV.cv_results_['mean_test_score']),
        "cv_std_r2"     : np.std(searchCV.cv_results_['mean_test_score']),
        "accuracy"      : test_score,
        "R2"            : r2_score(Y_TEST, y_pred), 
        "MSE"           : mean_squared_error(Y_TEST, y_pred),
        "RMSE"          : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
        "MAE"           : np.mean(np.abs(Y_TEST - y_pred)),
        "training_time" : end_training - start_training,
        "best_params"   : searchCV.best_params_,
        "epochs"        : searchCV.best_estimator_.n_iter_,
    }
    models["gridsearch_MLP"] = searchCV.best_estimator_

    return results, models , model_prefix


# ----------------------------- Plotting Results ----------------------------- #
def plot_results(results: dict, new_best_models: dict, model_prefix: str,save_models: bool = True,y_pred: np.ndarray = None,y_test: np.ndarray = None):
    theme = load_theme("umbra_dark")
    theme.apply()
    max_cols = 4
    max_rows = ceil(len(results.keys()) / max_cols)
    plot_idx = 0


    plt.figure(figsize=(20, 5 * max_rows),layout='constrained')
    for i , model_res in enumerate(results.items()):
        name = model_res[0]
        model = new_best_models[name]
        res = model_res[1]

        print(f"======================== Start model: {name} ========================")
        print(f"""
            {name.ljust(30)}: 
            R2 = {res['R2']:.3f}, 
            MSE = {res['MSE']:.3f}, 
            RMSE = {res['RMSE']:.3f}, 
            MAE = {res['MAE']:.3f}, 
            Training Time = {res['training_time']:.2f} seconds
        """)
        pprint(f"""
            Best parameters: {res['best_params']}
        """)

        if y_pred is None:
            y_pred = np.array(model.predict(X_TEST))

        if y_test is None:
            y_test = np.array(Y_TEST)

        if name in new_best_models.keys():
            plt.subplot(max_rows,max_cols,i + 1)

            row = plot_idx // max_cols
            col = plot_idx % max_cols
            print(plot_idx, row, col)
            # ----------------------------- True vs Predicted -----------------------------
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot(y_test, y_test, 'r--', lw=2)
            plt.xlabel("True Values",labelpad=10)
            plt.ylabel("Predicted Values",labelpad=10)
            plt.title(
                f"{name}: True vs Predicted",
                fontdict={
                    "fontsize": 9,
                    "fontweight": "bold",
                    "fontfamily": "monospace"
                }, 
                pad=10
            )
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.xlim([y_test.min(), y_test.max()])
            plt.ylim([y_test.min(), y_test.max()])
            plt.gca().spines[:].set_visible(True) 
            plt.annotate(
                text=f"epochs: {res['epochs']:.3f}\nR2: {res['R2']:.3f}\nMSE: {res['MSE']:.3f}\nRMSE: {res['RMSE']:.3f}\nMAE: {res['MAE']:.3f}\nTraining Time: {res['training_time']:.2f} seconds",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontfamily='monospace',
                fontsize=8,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            )
            
            plot_idx += 1
            # ----------------------------- True vs Predicted -----------------------------
            
        print(f"======================== End model: {name} ========================")


    plt.tight_layout(pad=2.0,h_pad=3.5)

    # ----------------------------- Plotting Results ----------------------------- #


    # ----------------------------- Save Results ----------------------------- #
    filename_results_pdf = model_prefix + filename_results_best_models_pdf
    plt.savefig(
        os.path.join(report_assets_path,filename_results_pdf), 
        format="pdf"
    )

    results_df =  pd.DataFrame().from_dict(
        results, 
        orient='index', 
        columns=[
            "cv_mean_r2", 
            "cv_std_r2", 
            "accuracy", 
            "R2", 
            "MSE", 
            "RMSE", 
            "MAE", 
            "training_time",
            "best_params",
        ]
    )

    filename_results_csv = model_prefix + filename_results_best_models_csv
    results_df.to_csv(os.path.join(report_assets_path,filename_results_csv))

    if save_models:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(os.path.join(models_dir,model_prefix)):
            os.makedirs(os.path.join(models_dir,model_prefix))

        for name, model in new_best_models.items():
            model_filename = model_prefix + name.replace(' ', '_').lower() + "_model.pkl"
            model_path = os.path.join(models_dir,model_prefix, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"{COLORS_TERMINAL['CYAN']}Model {name} saved to {model_path}{COLORS_TERMINAL['RESET_COLOR']}")

    # ----------------------------- Save Results ----------------------------- #


    print(f"{COLORS_TERMINAL['GREEN']}All models have been fine-tuned and results saved in {report_assets_path}{COLORS_TERMINAL['RESET_COLOR']}")

    return model_prefix