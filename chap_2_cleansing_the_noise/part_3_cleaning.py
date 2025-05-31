import pandas as pd
import os
from main_config import (
    train_data_cleaned_path,
    test_data_cleaned_path,
    data_cleaned_path,
    train_data_cleaned_no_outliers_path,
    test_data_cleaned_no_outliers_path,
    data_cleaned_no_outliers_path, 
    full_dataset_cleaned_path,
    full_dataset_cleaned_no_outliers_path,
)

df_test = pd.read_csv(test_data_cleaned_path)
df_train = pd.read_csv(train_data_cleaned_path)
df_test_train = pd.read_csv(data_cleaned_path)
df_full_dataset = pd.read_csv(full_dataset_cleaned_path)

q1 =  df_test_train["Exam_Score"].quantile(0.25)
q3 =  df_test_train["Exam_Score"].quantile(0.75)
iqr = q3 - q1

df_test = df_test[
    (df_test["Exam_Score"] >= q1 - 1.5 * iqr) &
    (df_test["Exam_Score"] <= q3 + 1.5 * iqr)
]
df_train = df_train[
    (df_train["Exam_Score"] >= q1 - 1.5 * iqr) &
    (df_train["Exam_Score"] <= q3 + 1.5 * iqr)
]
df_test_train = df_test_train[
    (df_test_train["Exam_Score"] >= q1 - 1.5 * iqr) &
    (df_test_train["Exam_Score"] <= q3 + 1.5 * iqr)
]
df_full_dataset = df_full_dataset[
    (df_full_dataset["Exam_Score"] >= q1 - 1.5 * iqr) &
    (df_full_dataset["Exam_Score"] <= q3 + 1.5 * iqr)
]

df_test.to_csv(train_data_cleaned_no_outliers_path, index=False)
df_train.to_csv(test_data_cleaned_no_outliers_path, index=False)
df_test_train.to_csv(data_cleaned_no_outliers_path, index=False)
df_full_dataset.to_csv(full_dataset_cleaned_no_outliers_path, index=False)
