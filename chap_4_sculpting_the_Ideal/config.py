import pandas as pd
from main_config import (
    train_data_cleaned_no_outliers_path,
    test_data_cleaned_no_outliers_path,
    data_cleaned_path
)

df_test = pd.read_csv(test_data_cleaned_no_outliers_path)
df_train = pd.read_csv(train_data_cleaned_no_outliers_path)

Y_TRAIN = df_train["Exam_Score"]
X_TRAIN = df_train.drop(columns=["Exam_Score"]).copy()

Y_TEST = df_test["Exam_Score"]
X_TEST = df_test.drop(columns=["Exam_Score"]).copy()

# Hours_Studied                     0.445104 <-
# Attendance                        0.580259 <-
# Previous_Scores                   0.174283 <-
# Tutoring_Sessions                 0.156829 <-
# Parental_Involvement              0.156014 <-
# Access_to_Resources               0.167856 <-
# Parental_Education_Level          0.105253 <-
# Resource_Utilization_Score        0.121853 <-
# Motivation_Effort                 0.338626 <-
# Parental_Support_Score            0.145135 <-

# removing :
# - Resource_Utilization_Score
# - Motivation_Effort
# - Parental_Support_Score
# from the training and test sets
# seems to help the models but only slightly

X_TRAIN.drop(columns=[
    "Resource_Utilization_Score",
    "Motivation_Effort",
    "Parental_Support_Score"
], inplace=True)
X_TEST.drop(columns=[
    "Resource_Utilization_Score",
    "Motivation_Effort",
    "Parental_Support_Score"
], inplace=True)


