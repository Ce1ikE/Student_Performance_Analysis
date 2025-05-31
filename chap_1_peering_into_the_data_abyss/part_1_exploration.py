from chap_1_peering_into_the_data_abyss.config import df_raw

df_raw.info()
print(df_raw.describe())

# # output :
# ----------------------------------------------------- #
# RangeIndex: 6607 entries, 0 to 6606
# Data columns (total 20 columns):
#  #   Column                      Non-Null Count  Dtype
# ---  ------                      --------------  -----
#  0   Hours_Studied               6607 non-null   int64
#  1   Attendance                  6607 non-null   int64
#  2   Parental_Involvement        6607 non-null   object
#  3   Access_to_Resources         6607 non-null   object
#  4   Extracurricular_Activities  6607 non-null   object
#  5   Sleep_Hours                 6607 non-null   int64
#  6   Previous_Scores             6607 non-null   int64
#  7   Motivation_Level            6607 non-null   object
#  8   Internet_Access             6607 non-null   object
#  9   Tutoring_Sessions           6607 non-null   int64
#  10  Family_Income               6607 non-null   object
#  11  Teacher_Quality             6529 non-null   object
#  12  School_Type                 6607 non-null   object
#  13  Peer_Influence              6607 non-null   object
#  14  Physical_Activity           6607 non-null   int64
#  15  Learning_Disabilities       6607 non-null   object
#  16  Parental_Education_Level    6517 non-null   object
#  17  Distance_from_Home          6540 non-null   object
#  18  Gender                      6607 non-null   object
#  19  Exam_Score                  6607 non-null   int64
# ----------------------------------------------------- #
# dtypes: int64(7), object(13)
# memory usage: 1.0+ MB

#        Hours_Studied   Attendance  Sleep_Hours  Previous_Scores  Tutoring_Sessions   Exam_Score
# count    6607.000000  6607.000000   6607.00000      6607.000000        6607.000000  6607.000000
# mean       19.975329    79.977448      7.02906        75.070531           1.493719    67.235659
# std         5.990594    11.547475      1.46812        14.399784           1.230570     3.890456
# min         1.000000    60.000000      4.00000        50.000000           0.000000    55.000000
# 25%        16.000000    70.000000      6.00000        63.000000           1.000000    65.000000
# 50%        20.000000    80.000000      7.00000        75.000000           1.000000    67.000000
# 75%        24.000000    90.000000      8.00000        88.000000           2.000000    69.000000
# max        44.000000   100.000000     10.00000       100.000000           8.000000   101.000000



# ----------------------------------------------------- #
# our target column is: 
# Exam_score

# we have 20 columns
# 7  which are numerical   (int64)
# 13 which are categorical (object)

# 3 columns which are missing values :
# Parental_Education_Level
# Distance_from_Home
# Teacher_Quality

# we also see that exam score is ranging from 55 to 101
# so if we consider a exam score of "exam_score < 50" as a fail then that means in this dataset we have no fails
# also we should remove or at least replace the 101 by a 100 once we're cleaning the data


