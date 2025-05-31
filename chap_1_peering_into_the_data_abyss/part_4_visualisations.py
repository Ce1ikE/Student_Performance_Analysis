from main_config import correlation_matrix_path
from chap_1_peering_into_the_data_abyss.part_3_preprocesssing import df_preprocessed

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme

theme = load_theme("umbra_dark")
theme.apply()



# see "math for AI" for the Pearson correlation coefficient
# the Pearson corr coefficient is between [-1,1] where 1 or -1 means highly correlated (negatively or positively)
# however it only measures linear correlation 
# !! I don't know what "spearman" and "kendall" does
correlation_matrix = df_preprocessed.corr(method="pearson")
correlation_matrix["Exam_Score"].sort_values(ascending=True)

# the Exam_Score has somehow a correlation with: 
# "Hours_Studied"
# "Attendance" 
# and a little bit with: 
# "Previous_Scores" 
# "Tutoring_Sessions"
# "Parental_Involvement" 
# "Access_to_Resources"
# "Parental_Education_Level"
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
# Name: Exam_Score, dtype: float64

# as correlation is pretty low i might try to combine some features later on to get
# a new feature for better results  

# https://medium.com/data-science/feature-selection-with-pandas-e3690ad8504b
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# https://stackoverflow.com/questions/60611055/add-annotation-to-specific-cells-in-heatmap
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
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
plt.title("Correlation Matrix of all features")
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
    correlation_matrix_path, 
    format="pdf", 
    bbox_inches="tight"
)
