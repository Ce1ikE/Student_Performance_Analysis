import pandas as pd
from main_config import (
    train_data_cleaned_no_outliers_path,
    test_data_cleaned_no_outliers_path,
    RANDOM_SEED,
    full_dataset_cleaned_no_outliers_path
)
from sklearn.model_selection import train_test_split

df_test = pd.read_csv(test_data_cleaned_no_outliers_path)
df_train = pd.read_csv(train_data_cleaned_no_outliers_path)

Y_TRAIN_FULL = df_train["Exam_Score"]
X_TRAIN_FULL = df_train.drop(columns=["Exam_Score"]).copy()

X_TRAIN , X_EVAL , Y_TRAIN , Y_EVAL = train_test_split(X_TRAIN_FULL, Y_TRAIN_FULL, random_state=RANDOM_SEED)

Y_TEST = df_test["Exam_Score"]
X_TEST = df_test.drop(columns=["Exam_Score"]).copy()



