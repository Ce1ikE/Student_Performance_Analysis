from chap_1_peering_into_the_data_abyss.part_3_preprocesssing import df_preprocessed
from chap_1_peering_into_the_data_abyss.config import df_raw
from main_config import correlation_matrix_new_features_path  

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from aquarel import load_theme

theme = load_theme("umbra_dark")
theme.apply()

df_raw.dropna(inplace=True)
extra_columns = [
    "Study_Efficiency",
    "Preparation",
    "Resource_Utilization_Score",
    "Health_Score",
    "Motivation_Effort",
    "Parental_Support_Score",
]

# https://medium.com/@bijit211987/10-advanced-feature-engineering-methods-46b63a1ee92e
# https://medium.com/@silva.f.francis/advanced-feature-engineering-for-machine-learning-9e2e34c39a82

df_preprocessed["Study_Efficiency"] = df_preprocessed["Hours_Studied"] * df_preprocessed["Attendance"]

df_preprocessed["Preparation"] = df_preprocessed["Tutoring_Sessions"] * df_preprocessed['Previous_Scores']

df_preprocessed["Resource_Utilization_Score"] = df_preprocessed["Access_to_Resources"] * df_preprocessed["Internet_Access_Yes"] * df_preprocessed["Tutoring_Sessions"]

df_preprocessed["Health_Score"] = df_preprocessed["Sleep_Hours"] * df_preprocessed["Physical_Activity"]

df_preprocessed['Motivation_Effort'] = df_preprocessed['Motivation_Level'] * df_preprocessed['Hours_Studied']

df_preprocessed["Parental_Support_Score"] = df_preprocessed["Parental_Involvement"] * df_preprocessed["Parental_Education_Level"]   

correlation_matrix = df_preprocessed.corr(method="pearson")
correlation_matrix["Exam_Score"].sort_values(ascending=True)


# okay it seems that "Resource_Utilization_Score", "Parental_Support_Score" and "Motivation_Effort"
# are quite nice
print(correlation_matrix["Exam_Score"])

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
# Study_Efficiency                  0.007305
# Preparation                       0.000650
# Resource_Utilization_Score        0.121853 <-
# Health_Score                     -0.012703
# Motivation_Effort                 0.338626 <-
# Parental_Support_Score            0.145135 <-



annotations = np.where(
    np.abs(correlation_matrix) > 0.1, 
    correlation_matrix.round(2).astype(dtype=str).copy(deep=True), 
    ""
)
theme.apply_transforms()
plt.figure(figsize=(12,9))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title("Correlation Matrix with new features")
sns.heatmap(
    correlation_matrix,
    cmap=plt.get_cmap(name="RdBu"),
    vmin=-1,
    vmax=1,
    annot=annotations,
    annot_kws={"fontsize":8},
    fmt="s",
    linewidths=.5,
    linecolor='black',
)

plt.savefig(
    correlation_matrix_new_features_path, 
    format="pdf", 
    bbox_inches="tight"
)