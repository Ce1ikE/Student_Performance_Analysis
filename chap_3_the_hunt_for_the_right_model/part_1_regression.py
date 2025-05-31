# ----------------------------------------------------------------------------------------------------------
# evaluating basic regression models not using ensemble methods and not using neural networks
from chap_3_the_hunt_for_the_right_model.config import X_TEST , X_TRAIN , Y_TEST , Y_TRAIN
from main_config import (
    RANDOM_SEED,
    filename_results_base_models_pdf,
    filename_results_base_models_csv,
) 

# ------------------------------------------------- models -------------------------------------------------
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
# ----------------------------------------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
import pandas as pd
from math import ceil
import os
import time

# https://www.geeksforgeeks.org/regression-in-machine-learning/
# https://joseph-gatto.medium.com/intuition-why-l1-regularization-pushes-coefficients-to-0-dece7673896b
models = {
    "DummyRegressor" : DummyRegressor(),
    "LinearRegression" : LinearRegression(),
    "Ridge" : Ridge(random_state=RANDOM_SEED),
    "Lasso" : Lasso(random_state=RANDOM_SEED),
    "ElasticNet" : ElasticNet(random_state=RANDOM_SEED),
    "BayesianRidge" : BayesianRidge(),
    "HuberRegressor" : HuberRegressor(),
    "PassiveAggressiveRegressor" : PassiveAggressiveRegressor(random_state=RANDOM_SEED),
    "SVR" : SVR(),
    "DecisionTreeRegressor" : DecisionTreeRegressor(random_state=RANDOM_SEED),
    "KNeighborsRegressor" : KNeighborsRegressor(),
}


results: dict[str,dict] = dict()

# ----------------------------- Cross Validation and Model Fitting ----------------------------- #
for name, model in models.items():
    start_training = time.time()
    cv_scores = cross_val_score(model, X_TRAIN, Y_TRAIN, cv=10, scoring="r2", n_jobs=-1)
    end_training = time.time()
    print(f"fitting model with cross validation : {name} DONE")
    
    test_score = model.score(X_TEST, Y_TEST)
    y_pred = np.array(model.predict(X_TEST))
    print(f"prediction model: {name} DONE")

    # https://medium.com/@brandon93.w/regression-model-evaluation-metrics-r-squared-adjusted-r-squared-mse-rmse-and-mae-24dcc0e4cbd3
    results[name] = {
        "cv_mean_r2" : np.mean(cv_scores),
        "cv_std_r2"  : np.std(cv_scores),
        "accuracy"  : test_score,

        # basically saying how well the independent variables (X) explain the variance in the dependent variable (Y)
        # for ensemble methods, use adjusted R2 =>  "Adj_R2"     : 1 - (1 - r2_score(Y_TEST, y_pred)) * (len(Y_TEST) - 1) / (len(Y_TEST) - X_TRAIN.shape[1] - 1)
        "R2"         : r2_score(Y_TEST, y_pred), 
        
        # very easy to understand, it is just the distance between the predicted values and the actual values, we just square it to represent that distance
        # for all samples ("mean" squared error) =>  (sum((y[n] - y_pred[n])**2) for all n in y_pred) * 1 / len(y_pred) 
        "MSE"        : mean_squared_error(Y_TEST, y_pred),
        
        # root mean squared error, it is the square root of the MSE
        "RMSE"       : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
        
        # mean absolute error, it is the average of the absolute differences between predicted and actual values
        # basically , gives us an idea of how far off our predictions are from the actual values, on average
        # but RMSE and MSE both square the differences, which means they penalize larger errors more heavily
        # by taking the absolute value, we avoid that squaring effect
        "MAE"        : np.mean(np.abs(Y_TEST - y_pred)),

        "training_time" : end_training - start_training,
    }
# ----------------------------- Cross Validation and Model Fitting ----------------------------- #



# ----------------------------- Plotting Results ----------------------------- #
theme = load_theme("umbra_dark")
theme.apply()
y_true = np.array(Y_TEST)
max_cols = 4
max_rows = ceil(len(results.keys()) / max_cols)
plot_idx = 0


plt.figure(figsize=(20, 5 * max_rows),layout='constrained')
for i , model_res in enumerate(results.items()):
    name = model_res[0]
    model = models[name]
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

    y_pred = np.array(model.predict(X_TEST))

    if name in models.keys():
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
            text=f"accuracy: {res["accuracy"]:.3f}\nR2: {res['R2']:.3f}\nMSE: {res['MSE']:.3f}\nRMSE: {res['RMSE']:.3f}\nMAE: {res['MAE']:.3f}\nTraining Time: {res['training_time']:.2f} seconds",
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
plt.savefig(
    filename_results_base_models_pdf, 
    format="pdf"
)
results_df = pd.DataFrame().from_dict(
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
        "training_time"
    ]
)
results_df.to_csv(filename_results_base_models_csv)
# ----------------------------- Save Results ----------------------------- #
