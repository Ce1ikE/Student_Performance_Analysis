from chap_5_awakening_the_neural_mind.config import *
from chap_5_awakening_the_neural_mind.part_1_building_the_mind import enter_model_name , plot_results
from main_config import (
    report_assets_path ,
    COLORS_TERMINAL,
    COLORS_BOLD,
    models_dir,
    filename_results_best_models_csv,
    filename_results_best_models_pdf , 
    filename_results_NN_pdf,
    full_dataset_cleaned_no_outliers_path,
    RANDOM_SEED,
    print_cursor,
    type_out,
)

import tensorflow as tf


from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aquarel import load_theme
from math import ceil
import os
import time
from pprint import pprint
import pickle


def main():
    tf.random.set_seed(RANDOM_SEED)

    models = {}
    results = {}
    model_prefix = enter_model_name(default_prefix="nn1_")

    type_out(f"Use full dataset for training? (y/n) ", delay=0.02)
    use_full_dataset = print_cursor().strip().lower()
    if use_full_dataset in ["y", "yes"]:
        print(f"{COLORS_BOLD['BOLD_GREEN']}Using full dataset for training...{COLORS_TERMINAL['RESET_COLOR']}")
        
        df_full_dataset = pd.read_csv(full_dataset_cleaned_no_outliers_path)
        X_TRAIN_FULL , X_TEST , Y_TRAIN_FULL , Y_TEST = train_test_split(
            df_full_dataset.drop(columns=["Exam_Score"]).copy(), 
            df_full_dataset["Exam_Score"], 
            random_state=RANDOM_SEED
        )

        X_TRAIN, X_EVAL, Y_TRAIN, Y_EVAL = train_test_split(
            X_TRAIN_FULL, 
            Y_TRAIN_FULL, 
            random_state=RANDOM_SEED
        )
        


    # build the Neural Network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=X_TRAIN.shape[1:]),
        tf.keras.layers.Dense(1)
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10, 
        restore_best_weights=True, 
        verbose=1
    )
    
    # compile the model
    # https://keras.io/api/metrics/
    model.compile(
        optimizer='adam', 
        loss='mse', 
        metrics=[
            'mae', 
            'mse',
            tf.keras.metrics.R2Score(
                class_aggregation="uniform_average", num_regressors=0, name="r2", dtype=None
            )
        ]
    )

    nbr_of_epochs = 700
    type_out(f"Enter number of epochs (default: {nbr_of_epochs})", delay=0.02)
    user_input = print_cursor().strip()
    if user_input.isdigit():
        nbr_of_epochs = int(user_input)
        if nbr_of_epochs <= 0:
            print(f"{COLORS_TERMINAL['RED']}Number of epochs must be a positive integer.{COLORS_TERMINAL['RESET_COLOR']}")
            nbr_of_epochs = 700
    else:
        print(f"{COLORS_TERMINAL['YELLOW']}Using default number of epochs: {nbr_of_epochs}{COLORS_TERMINAL['RESET_COLOR']}")

    print(f"{COLORS_BOLD['BOLD_GREEN']}Training custom MLP model with {nbr_of_epochs} epochs...{COLORS_TERMINAL['RESET_COLOR']}")
    # ------------------------------------------------------- #
    start_training = time.time()
    history: tf.keras.callbacks.History = model.fit(
        X_TRAIN, 
        Y_TRAIN, 
        epochs=nbr_of_epochs, 
        batch_size=ceil(X_TRAIN.shape[0] / 10),
        validation_data=(X_EVAL,Y_EVAL),
        callbacks=[
            early_stopping
        ],
    )
    end_training = time.time()
    # ------------------------------------------------------- #
    print(f"{COLORS_BOLD['BOLD_GREEN']}Training custom MLP model DONE!{COLORS_TERMINAL['RESET_COLOR']}")
    
    y_pred = model.predict(X_TEST)
    results["custom_MLP"] = {
        "cv_mean_r2"    : None,
        "cv_std_r2"     : None,
        "accuracy"      : None,
        "R2"            : r2_score(Y_TEST, y_pred), 
        "MSE"           : mean_squared_error(Y_TEST, y_pred),
        "RMSE"          : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
        "MAE"           : history.history['mae'][-1],
        "training_time" : end_training - start_training,
        "best_params"   : None ,
        "epochs"        : history.epoch[-1] + 1,
    }        
    models["custom_MLP"] = model


    theme = load_theme("umbra_dark")
    theme.apply()
    plt.figure(figsize=(18, 15),layout='constrained')
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['r2'])
    plt.plot(history.history['val_r2'])
    plt.title(
        'model r2 over epochs',
        fontdict={
            "fontsize": 9,
            "fontweight": "bold",
            "fontfamily": "monospace"
        }, 
        pad=10
    )
    plt.ylabel('R2',labelpad=10)
    plt.ylim(0.8, 1.0)
    plt.xlabel('epoch',labelpad=10)
    plt.legend(['train', 'test'], loc='upper left')
    plt.gca().spines[:].set_visible(True) 


    plt.subplot(1, 2, 2)
    plt.semilogy(history.history['loss'], label='train')
    plt.semilogy(history.history['val_loss'], label='val')
    plt.title(
        'model loss over epochs',
        fontdict={
            "fontsize": 9,
            "fontweight": "bold",
            "fontfamily": "monospace"
        }, 
        pad=10
    )
    plt.ylabel('loss (log scale)')
    plt.xlabel('epoch')
    plt.gca().spines[:].set_visible(True) 
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.tight_layout(pad=2.0,h_pad=3.5)

    filename_results_pdf = model_prefix + filename_results_NN_pdf
    plt.savefig(
        os.path.join(report_assets_path,filename_results_pdf), 
        format="pdf"
    )

    plot_results(
        results, 
        models, 
        model_prefix, 
        save_models=False, 
        y_pred=y_pred, 
        y_test=Y_TEST
    )
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(os.path.join(models_dir,model_prefix)):
        os.makedirs(os.path.join(models_dir,model_prefix))

    model.save(
        os.path.join(
            models_dir,
            model_prefix,
            model_prefix + "custom_MLP.h5"
        )
    )


