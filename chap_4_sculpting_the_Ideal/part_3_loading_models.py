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
import sys
import pickle
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

def load_models_from_dir(path: str):
    models = {}
    print(f"{COLORS_BOLD['BOLD_GREEN']}Loading models from directory: {path}{COLORS_TERMINAL['RESET_COLOR']}")
    print(f"{COLORS_BOLD['BOLD_GREEN']}Files found: {os.listdir(path)}{COLORS_TERMINAL['RESET_COLOR']}")

    for filename in os.listdir(path):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]
            file_path = os.path.join(path, filename)
            with open(file_path, "rb") as f:
                models[model_name] = pickle.load(f)
                
        if filename.endswith(".h5"):
            model_name = os.path.splitext(filename)[0]
            file_path = os.path.join(path, filename)
            models[model_name] = tf.keras.models.load_model(file_path)

    return models


def enter_model_prefix():
    type_out("Enter the model prefix (e.g., 'ver1_', 'ver2_', etc.): ")
    model_prefix = print_cursor().strip()
    
    if not model_prefix:
        print(f"{COLORS_TERMINAL['RED']}Model prefix cannot be empty.{COLORS_TERMINAL['RESET_COLOR']}")
        sys.exit(1)
    
    return model_prefix

def main():
    print(f"{COLORS_BOLD['BOLD_GREEN']}Loading models and results from previous chapters...{COLORS_TERMINAL['RESET_COLOR']}")
    
    try:
        model_prefix = enter_model_prefix()
        
        if not os.path.exists(models_dir):
            print(f"{COLORS_TERMINAL['RED']}Models directory does not exist: {models_dir}{COLORS_TERMINAL['RESET_COLOR']}")
            sys.exit(1)

        if not os.path.exists(os.path.join(models_dir, model_prefix)):
            print(f"{COLORS_TERMINAL['RED']}Model prefix directory does not exist: {os.path.join(models_dir, model_prefix)}{COLORS_TERMINAL['RESET_COLOR']}")
            sys.exit(1)

        models = load_models_from_dir(os.path.join(models_dir, model_prefix))
        print(f"{COLORS_BOLD['BOLD_GREEN']}Models loaded successfully!{COLORS_TERMINAL['RESET_COLOR']}")

        return models , model_prefix
    
    except Exception as e:
        print(f"{COLORS_TERMINAL['RED']}An error occurred while loading models: {e}{COLORS_TERMINAL['RESET_COLOR']}")
        sys.exit(1)


def test_loaded_models(models):
    results = {}
    if not models:
        print(f"{COLORS_TERMINAL['RED']}No models loaded.{COLORS_TERMINAL['RESET_COLOR']}")
        return False
    
    print(f"{COLORS_BOLD['BOLD_GREEN']}Testing loaded models...{COLORS_TERMINAL['RESET_COLOR']}")
    
    for i ,  model_name_estimator in enumerate(models.items()):
        model_name, model = model_name_estimator

        print(f"Model: {model_name}, Type: {type(model)}")
        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X_TEST)

                test_score = model.score(X_TEST, Y_TEST)

                print(f"prediction model: {model_name} DONE")

                important_params = {}
                params = model.get_params() if hasattr(model, 'get_params') else None
                print(f"Model {model_name} parameters: {params}")
                if params is None:
                    pass
                else:
                    for param in params:
                        if 'regressor__' in param:
                            print(f"{COLORS_BOLD['BOLD_GREEN']}Parameter: {param} = {params[param]}{COLORS_TERMINAL['RESET_COLOR']}")
                            important_params[param] = params[param]

                results[model_name] = {
                    "cv_mean_r2"    : np.nan if not hasattr(model, 'cv_results_') else np.mean(model.cv_results_['mean_test_score']),
                    "cv_std_r2"     : np.nan if not hasattr(model, 'cv_results_') else np.std(model.cv_results_['mean_test_score']),
                    "accuracy"      : test_score,
                    "R2"            : r2_score(Y_TEST, y_pred), 
                    "MSE"           : mean_squared_error(Y_TEST, y_pred),
                    "RMSE"          : np.sqrt(mean_squared_error(Y_TEST, y_pred)),
                    "MAE"           : np.mean(np.abs(Y_TEST - y_pred)),
                    "training_time" : 0,
                    "best_params"   : important_params
                }


            except Exception as e:
                print(f"{COLORS_TERMINAL['RED']}Error predicting with model {model_name}: {e}{COLORS_TERMINAL['RESET_COLOR']}")
                return False
        else:
            print(f"{COLORS_TERMINAL['RED']}Model {model_name} does not have a predict method.{COLORS_TERMINAL['RESET_COLOR']}")
            return False
        
    
    print(f"{COLORS_BOLD['BOLD_GREEN']}All models loaded successfully!{COLORS_TERMINAL['RESET_COLOR']}")
    return results