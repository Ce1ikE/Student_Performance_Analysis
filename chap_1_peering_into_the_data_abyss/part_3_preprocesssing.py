from chap_1_peering_into_the_data_abyss.config import (
    numerical_cols , 
    ordinal_cols ,
    ordinal_categories ,
    nominal_cols ,
    df_raw
)

from sklearn.preprocessing import StandardScaler , OrdinalEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd

df_raw.dropna(inplace=True)
numerical_transformer = StandardScaler()
categorical_transformer_ordinal = OrdinalEncoder(categories=ordinal_categories)
categorical_transformer_nominal = OneHotEncoder()

ct = ColumnTransformer(
    transformers=[
        ("numerical_cols",numerical_transformer,numerical_cols),
        ("ordinal_categories",categorical_transformer_ordinal,ordinal_cols),        
        ("nominal_categories",categorical_transformer_nominal,nominal_cols),        
    ]
)
df_transformed = ct.fit_transform(df_raw)
nominal_cols = list(ct.named_transformers_["nominal_categories"].get_feature_names_out(nominal_cols))
df_preprocessed = pd.DataFrame(df_transformed,columns=numerical_cols + ordinal_cols + nominal_cols)