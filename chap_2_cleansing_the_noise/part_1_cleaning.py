from main_config import (
    raw_data_path , 
    RANDOM_SEED,
    train_data_cleaned_path,
    test_data_cleaned_path,
    data_cleaned_path,
    full_dataset_cleaned_path,
)
from chap_1_peering_into_the_data_abyss.config import (
    categorical_cols , 
    numerical_cols , 
    ordinal_cols ,
    ordinal_categories ,
    nominal_cols ,
    nominal_categories ,
)

from sklearn.preprocessing import StandardScaler , OrdinalEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os 
import pandas as pd

target_column = [
    "Exam_Score",
]

selected_columns = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Parental_Involvement",
    "Access_to_Resources",
    "Parental_Education_Level",
]

df = pd.read_csv(raw_data_path)

df.dropna(inplace=True)
df['Exam_Score'] = df['Exam_Score'].replace(101, 100)

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer_ordinal = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories)),
    ]
)
categorical_transformer_nominal = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

ct = ColumnTransformer(
    transformers=[
        ("numerical_cols",numerical_transformer,numerical_cols),
        ("ordinal_categories",categorical_transformer_ordinal,ordinal_cols),        
        ("nominal_categories",categorical_transformer_nominal,nominal_cols),        
    ]
)



    # --------------- from this ---------------

    # Hours_Studied                     0.445104 <-
    # Attendance                        0.580259 <-
    # Sleep_Hours                      -0.017171
    # Previous_Scores                   0.174283 <-
    # Tutoring_Sessions                 0.156829 <-
    # Physical_Activity                 0.025148
    # Exam_Score                        1.000000
    # Parental_Involvement              0.156014 <-
    # Access_to_Resources               0.167856 <-
    # Motivation_Level                  0.088502
    # Family_Income                     0.094555
    # Teacher_Quality                   0.075107
    # Parental_Education_Level          0.105253 <-
    # Peer_Influence                    0.099133
    # Extracurricular_Activities_No    -0.063063
    # Extracurricular_Activities_Yes    0.063063
    # Internet_Access_No               -0.051124
    # Internet_Access_Yes               0.051124
    # Learning_Disabilities_No          0.083911
    # Learning_Disabilities_Yes        -0.083911
    # School_Type_Private               0.010868
    # School_Type_Public               -0.010868
    # Distance_from_Home_Far           -0.064088
    # Distance_from_Home_Moderate      -0.044868
    # Distance_from_Home_Near           0.081204
    # Resource_Utilization_Score        0.121853 <-
    # Motivation_Effort                 0.338626 <-
    # Parental_Support_Score            0.145135 <-

    # --------------- to this ---------------

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


exam_score = pd.DataFrame(df["Exam_Score"].copy(),columns=target_column)
df.drop(columns=target_column,inplace=True)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    df,
    exam_score,
    test_size=0.2,
    random_state=RANDOM_SEED
)

TRAIN = pd.concat([X_TRAIN,Y_TRAIN],axis=1)
TEST = pd.concat([X_TEST,Y_TEST],axis=1)
DATA = pd.DataFrame()
FULL_DATASET = pd.DataFrame()

# fit the transformer on the training set
# and transform the training set
df_transformed_train = ct.fit_transform(TRAIN)
df_transformed_test = ct.transform(TEST)

all_columns = (
    numerical_cols +
    ordinal_cols +
    list(ct.named_transformers_["nominal_categories"].get_feature_names_out(nominal_cols))
)
df_transformed_train = pd.DataFrame(df_transformed_train, columns=all_columns)
df_transformed_test = pd.DataFrame(df_transformed_test, columns=all_columns)

FULL_DATASET = pd.concat([df_transformed_train, df_transformed_test], axis=0, ignore_index=True)
FULL_DATASET.to_csv(full_dataset_cleaned_path, index=False)
# after we do some more feature engineering
# and create new features
df_transformed_train["Resource_Utilization_Score"] = df_transformed_train["Access_to_Resources"] * df_transformed_train["Internet_Access_Yes"] * df_transformed_train["Tutoring_Sessions"]
df_transformed_train['Motivation_Effort'] = df_transformed_train['Motivation_Level'] * df_transformed_train['Hours_Studied']
df_transformed_train["Parental_Support_Score"] = df_transformed_train["Parental_Involvement"] * df_transformed_train["Parental_Education_Level"]   
# then we create a new dataframe with the transformed data
# and the new features
selected_features = selected_columns + [
    "Resource_Utilization_Score",
    "Motivation_Effort",
    "Parental_Support_Score"
]
df_preprocessed_train = df_transformed_train[target_column + selected_features]
df_preprocessed_train.info()
# and finally we save the dataframe to a csv file
df_preprocessed_train.to_csv(train_data_cleaned_path)

DATA = df_preprocessed_train.copy()

# same for the test set, we repeat everything...

df_transformed_test["Resource_Utilization_Score"] = df_transformed_test["Access_to_Resources"] * df_transformed_test["Internet_Access_Yes"] * df_transformed_test["Tutoring_Sessions"]
df_transformed_test['Motivation_Effort'] = df_transformed_test['Motivation_Level'] * df_transformed_test['Hours_Studied']
df_transformed_test["Parental_Support_Score"] = df_transformed_test["Parental_Involvement"] * df_transformed_test["Parental_Education_Level"]   

df_preprocessed_test = df_transformed_test[target_column + selected_features]
df_preprocessed_test.info()
df_preprocessed_test.to_csv(test_data_cleaned_path)

DATA = pd.concat([DATA, df_preprocessed_test], axis=0,ignore_index=True)

# save full dataframe with all features
DATA.to_csv(data_cleaned_path)