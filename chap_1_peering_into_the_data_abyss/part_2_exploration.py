from chap_1_peering_into_the_data_abyss.config import df_raw , categorical_cols

from sklearn.preprocessing import OrdinalEncoder

df_raw.dropna(inplace=True)


categorical_transformer = OrdinalEncoder()
categorical_transformer.fit(df_raw[categorical_cols])


print("========================================================================================================================")
for i , category in enumerate(categorical_cols):
    print(f"{category.ljust(30)}         : {categorical_transformer.categories_[i]}")

# Parental_Involvement                   : ['High' 'Low' 'Medium']
# Access_to_Resources                    : ['High' 'Low' 'Medium']
# Extracurricular_Activities             : ['No' 'Yes']
# Motivation_Level                       : ['High' 'Low' 'Medium']
# Internet_Access                        : ['No' 'Yes']
# Family_Income                          : ['High' 'Low' 'Medium']
# Teacher_Quality                        : ['High' 'Low' 'Medium']
# School_Type                            : ['Private' 'Public']
# Peer_Influence                         : ['Negative' 'Neutral' 'Positive']
# Learning_Disabilities                  : ['No' 'Yes']
# Parental_Education_Level               : ['College' 'High School' 'Postgraduate']
# Distance_from_Home                     : ['Far' 'Moderate']

print("========================================================================================================================")
print(df_raw.head(n=25))




