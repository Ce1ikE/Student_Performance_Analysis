# based upon the previous results we will fine tune the best models
from chap_4_sculpting_the_Ideal.config import X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
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
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor , BaggingRegressor , RandomForestRegressor , VotingRegressor , AdaBoostRegressor , StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
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

            
def enter_hyperparameter_sweep(HYPERPARAMETER_SWEEP):
    

    while True:
        if HYPERPARAMETER_SWEEP == 1:
            best_models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(random_state=RANDOM_SEED),
                "BayesianRidge" : BayesianRidge(),
                "HuberRegressor" : HuberRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(random_state=RANDOM_SEED),
                "GradientBoostingRegressor" : GradientBoostingRegressor(random_state=RANDOM_SEED),
                "BaggingRegressor" : BaggingRegressor(estimator=LinearRegression(),random_state=RANDOM_SEED),
                "AdaBoostRegressor" : AdaBoostRegressor(random_state=RANDOM_SEED),
            }
            print("Using the first sweep of hyperparameters")
            # ----- hyperparameter init ----- 1st SWEEP #
            param_LR = {
                # i know... linear regression does not have hyperparameters to tune, but we can still include it for consistency
                "regressor" : [best_models["LinearRegression"]]
            }
            param_Ridge = {
                "regressor__alpha": [0.8, 1.0, 1.2],
                "regressor" : [best_models["Ridge"]]
            }
            param_BayesianRidge = {
                "regressor__alpha_1": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor__alpha_2": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor__lambda_1": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor__lambda_2": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor__max_iter": [390,400,450,500,550],
                "regressor__tol": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor" : [best_models["BayesianRidge"]]
            }
            param_HuberRegressor = {
                "regressor__epsilon": [1.4, 1.5, 1.6],
                "regressor__alpha": [0.4, 0.5, 0.6],
                "regressor__max_iter": [390,400,450,500,550],
                "regressor__tol": [1e-4,1e-5, 1e-6, 1e-7, 1e-8],
                "regressor" : [best_models["HuberRegressor"]]
            }
            param_RandomForestRegressor = {
                "regressor__n_estimators": [90, 100, 120,150,200],
                "regressor__max_depth": [15, 20, 25,30,35],
                "regressor__min_samples_split": [2, 5, 10, 15],
                "regressor__min_samples_leaf": [1, 2, 4, 6],
                "regressor" : [best_models["RandomForestRegressor"]]
            }
            param_GradientBoostingRegressor = {
                "regressor__n_estimators": [50, 100, 200,250,300,350],
                "regressor__learning_rate": [0.05, 0.1, 0.15,0.2,0.25,0.3],
                "regressor__max_depth": [3, 5, 7],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__subsample": [0.6, 0.8, 1.0],
                "regressor" : [best_models["GradientBoostingRegressor"]]
            }
            param_BaggingRegressor = {
                "regressor__n_estimators": [10, 50, 100],
                "regressor__max_samples": [0.5, 0.75, 1.0],
                "regressor__max_features": [0.5, 0.75, 1.0],
                "regressor" : [best_models["BaggingRegressor"]]
            }

            param_AdaBoostRegressor = {
                "regressor__n_estimators": [50, 100, 200,250,300,350],
                "regressor__learning_rate": [0.05, 0.1, 0.15,0.2,0.25,0.3],
                "regressor" : [best_models["AdaBoostRegressor"]]
            }

            pipeline = Pipeline([('regressor',best_models["LinearRegression"])])
            params = [
                param_LR,
                param_Ridge,
                param_BayesianRidge,
                param_HuberRegressor,
                param_RandomForestRegressor,
                param_GradientBoostingRegressor,
                param_BaggingRegressor,
                param_AdaBoostRegressor,
            ]

            search_method = "random" 
            model_prefix = "ver1_"
            model_prefix = enter_model_name(default_prefix="ver1_")
            return pipeline, params, search_method, model_prefix, best_models

        # this is the second sweep of hyperparameters, where we search for more refined hyperparameters based on the previous models results.
        elif HYPERPARAMETER_SWEEP == 2:
            best_models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(random_state=RANDOM_SEED),
                "BayesianRidge" : BayesianRidge(),
                "HuberRegressor" : HuberRegressor(),
                "BaggingRegressor" : BaggingRegressor(estimator=LinearRegression(),random_state=RANDOM_SEED),
            }
            print("Using the second sweep of hyperparameters")
            # ----- hyperparameter init ----- 2nd SWEEP #
            param_LR = {
                "regressor" : [best_models["LinearRegression"]]
            }
            param_Ridge = {
                "regressor" : [best_models["Ridge"]],
                "regressor__alpha": [3.5], 
            }
            param_BayesianRidge = {
                "regressor" : [best_models["BayesianRidge"]],
                "regressor__tol": [0.00012],
                "regressor__max_iter": [1],      
                "regressor__lambda_2": [1e-10],
                "regressor__lambda_1": [1.0], 
                "regressor__alpha_2": [1.5e-05], 
                "regressor__alpha_1": [1.5e-05], 
            }
            param_HuberRegressor = {
                "regressor" : [best_models["HuberRegressor"]],
                "regressor__tol": [1e-08],   
                "regressor__max_iter": [550],      
                "regressor__epsilon": [1.7],       
                "regressor__alpha": [0.7],         
            }
            param_BaggingRegressor = {
                "regressor" : [best_models["BaggingRegressor"]],
                "regressor__n_estimators": [80],    
                "regressor__max_samples": [0.5],  
                "regressor__max_features": [1.0],              
            }

            pipeline = Pipeline([('regressor', best_models["LinearRegression"])])
            params = [
                param_LR,
                param_Ridge,
                param_BayesianRidge,
                param_HuberRegressor,
                param_BaggingRegressor,
            ]

            search_method = "grid" 
            model_prefix = "ver2_"
            model_prefix = enter_model_name(default_prefix="ver2_")
            return pipeline, params, search_method, model_prefix, best_models

        # this is the third sweep of hyperparameters, where we use ensemble methods to combine the best models from the previous sweeps.
        elif HYPERPARAMETER_SWEEP == 3:
            print("Using the third sweep of hyperparameters")
            # ----- hyperparameter init ----- 3rd SWEEP #
            estimators_found = []
            estimators_for_ensemble = []
            searching_for_models = True
            # load the best models from the previous sweeps
            while searching_for_models:
                for root, dirs, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith(".pkl"):
                            model_name = file.replace(".pkl", "")
                            estimators_found.append((model_name, os.path.join(root, file)))
                
                print(f"{COLORS_BOLD['BOLD_GREEN']}Found {len(estimators_found)} models in the directory: {models_dir}{COLORS_TERMINAL['RESET_COLOR']}")

                for i, (model_name, model_path) in enumerate(estimators_found):

                    print(f"Current ensemble models:")
                    for n , (name, model) in enumerate(estimators_for_ensemble):
                        print(f"{n+1}. \"{name}\"")
                    
                    if (model_name, model_path) not in estimators_for_ensemble:
                        print(f"{COLORS_TERMINAL['YELLOW']}Found model: \"{model_name}\"\nAt: {model_path}{COLORS_TERMINAL['RESET_COLOR']}")
                        type_out(f"\nDo you want to use this model? (y/n) \nUse current models (u)", delay=0.02)
                        
                        user_input = print_cursor().strip().lower()
                        if user_input in ["y", "yes"]:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                                estimators_for_ensemble.append((model_name, model))
                            print(f"{COLORS_TERMINAL['GREEN']}Model \"{model_name}\" added to the ensemble.{COLORS_TERMINAL['RESET_COLOR']}")
                        
                        elif user_input in ["u", "use"]:
                            if len(estimators_for_ensemble) == 0:
                                print(f"{COLORS_TERMINAL['RED']}No models selected for the ensemble.{COLORS_TERMINAL['RESET_COLOR']}")
                                continue
                            else:
                                searching_for_models = False
                                break
                        
                        else:
                            print(f"{COLORS_TERMINAL['RED']}Model \"{model_name}\" skipped.{COLORS_TERMINAL['RESET_COLOR']}")



            best_models = {
                "VotingRegressor" : VotingRegressor(
                    weights=[1.0] * len(estimators_for_ensemble),
                    estimators=[
                        (name, model) for name, model in estimators_for_ensemble
                    ],
                    n_jobs=-1
                ),
                "StackingRegressor" : StackingRegressor(
                    estimators=[
                        (name, model) for name, model in estimators_for_ensemble
                    ],
                    final_estimator=LinearRegression(),
                    n_jobs=-1
                ),
            }
            param_StackingRegressor = {
                "regressor__cv": [5, 10],
                "regressor" : [best_models["StackingRegressor"]]
            }
            param_VotingRegressor = {
                "regressor" : [best_models["VotingRegressor"]]
            }

            pipeline = Pipeline([('regressor', best_models["StackingRegressor"])])
            params = [
                param_StackingRegressor,
                param_VotingRegressor,
            ]

            search_method = "grid" 
            model_prefix = "ver3_"
            model_prefix = enter_model_name(default_prefix="ver3_")

            return pipeline, params, search_method, model_prefix, best_models



# ----------------------------- Cross Validation and Model Fitting ----------------------------- #
def main(pipeline: list, params: list, search_method: str, model_prefix: str,best_models: dict):
    results: dict[str,dict] = dict()
    new_best_models = {}
    for i ,  model_name_estimator in enumerate(best_models.items()):
        name, model = model_name_estimator
        start_training = time.time()
        if search_method == "grid":
            searchCV = GridSearchCV(
                estimator=pipeline,
                param_grid=params[i],
                cv=10,
                n_jobs=-1,
                scoring='r2',
            ).fit(X_TRAIN, Y_TRAIN)
        else:
            searchCV = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=params[i],
                cv=10,
                n_iter=15,
                n_jobs=-1,
                scoring='r2',
                random_state=RANDOM_SEED
            ).fit(X_TRAIN, Y_TRAIN)        
        end_training = time.time()
        print(f"fitting model with cross validation : {name} DONE\nTime taken: {end_training - start_training:.2f} seconds")
        
        test_score = searchCV.score(X_TEST, Y_TEST)
        y_pred = np.array(searchCV.predict(X_TEST))
        print(f"prediction model: {name} DONE")
        
        searchCV.best_params_.pop("regressor", None) 
        results[name] = {
            "cv_mean_r2"    : np.mean(searchCV.cv_results_['mean_test_score']),
            "cv_std_r2"     : np.std(searchCV.cv_results_['mean_test_score']),
            "accuracy"      : test_score,
            "R2"            : r2_score(Y_TEST, y_pred), 
            "MSE"           : mean_squared_error(Y_TEST, y_pred),
            "RMSE"          : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
            "MAE"           : np.mean(np.abs(Y_TEST - y_pred)),
            "training_time" : end_training - start_training,
            "best_params"   : searchCV.best_params_ 
        }

        new_best_models[name] = searchCV.best_estimator_
        print(f"model {name} best params:\n")
        pprint(searchCV.best_params_)

    return results, new_best_models
# ----------------------------- Cross Validation and Model Fitting ----------------------------- #



# ----------------------------- Plotting Results ----------------------------- #
def plot_results(results: dict, new_best_models: dict, model_prefix: str,save_models: bool = True):
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
            CV R2 = {res['cv_mean_r2']:.3f} ± {res['cv_std_r2']:.3f}, 
            R2 = {res['R2']:.3f}, 
            MSE = {res['MSE']:.3f}, 
            RMSE = {res['RMSE']:.3f}, 
            MAE = {res['MAE']:.3f}, 
            Accuracy = {res['accuracy']:.3f},
            Training Time = {res['training_time']:.2f} seconds
        """)
        pprint(f"""
            Best parameters: {res['best_params']}
        """)

        y_pred = np.array(model.predict(X_TEST))

        if name in new_best_models.keys():
            plt.subplot(max_rows,max_cols,i + 1)

            row = plot_idx // max_cols
            col = plot_idx % max_cols
            print(plot_idx, row, col)
            # ----------------------------- True vs Predicted -----------------------------
            plt.scatter(Y_TEST, y_pred, alpha=0.7)
            plt.plot(Y_TEST, Y_TEST, 'r--', lw=2)
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
            plt.xlim([Y_TEST.min(), Y_TEST.max()])
            plt.ylim([Y_TEST.min(), Y_TEST.max()])
            plt.gca().spines[:].set_visible(True) 
            plt.annotate(
                text=f"accuracy: {res['accuracy']:.3f}\nR2: {res['R2']:.3f}\nMSE: {res['MSE']:.3f}\nRMSE: {res['RMSE']:.3f}\nMAE: {res['MAE']:.3f}\nTraining Time: {res['training_time']:.2f} seconds",
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
