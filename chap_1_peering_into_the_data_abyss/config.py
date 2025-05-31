import os
import pandas as pd
from main_config import raw_data_path 

df_raw = pd.read_csv(raw_data_path)
categorical_cols = [column for column in df_raw.columns if df_raw[column].dtype == "object"]
numerical_cols = [column for column in df_raw.columns if df_raw[column].dtype != "object"]

ordinal_cols = [
    'Parental_Involvement',
    'Access_to_Resources',
    'Motivation_Level',
    'Family_Income',
    'Teacher_Quality',
    'Parental_Education_Level',
    'Peer_Influence'
]

ordinal_categories = [
    ['Low', 'Medium', 'High'],                      # Parental_Involvement
    ['Low', 'Medium', 'High'],                      # Access_to_Resources
    ['Low', 'Medium', 'High'],                      # Motivation_Level
    ['Low', 'Medium', 'High'],                      # Family_Income
    ['Low', 'Medium', 'High'],                      # Teacher_Quality
    ['High School', 'College', 'Postgraduate'],     # Parental_Education_Level
    ['Negative', 'Neutral', 'Positive']             # Peer_Influence
]

nominal_cols = [
    'Extracurricular_Activities',
    'Internet_Access',
    'Learning_Disabilities',
    'School_Type',
    'Distance_from_Home'
]

nominal_categories = [
    ['No', 'Yes'],          # Extracurricular_Activities
    ['No', 'Yes'],          # Internet_Access
    ['No', 'Yes'],          # Learning_Disabilities
    ['Private', 'Public'],  # School_Type
    ['Far', 'Moderate']     # Distance_from_Home
]