from main_config import KDE_plots_of_numerical_features_path  
from chap_1_peering_into_the_data_abyss.config import (
    numerical_cols,
    df_raw,
)

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from aquarel import load_theme

theme = load_theme("umbra_dark")
theme.apply()


colors = [
    "#0EA4C29F",
    "#1665789E",
    "#3867B293",
    "#A1841BB6",
    "#19A21997",
    "#B3641A8C",
    "#6F1987C7"
]

rows = len(numerical_cols)// 2 + 1

plt.figure(figsize=(16, 20))
for i , column in enumerate(numerical_cols):
    plt.subplot(rows,2,i + 1)
    plt.subplot(rows,2,i + 1).set_title(column)

    count_df = df_raw[column].value_counts().reset_index()  
    count_df.columns = [column,'count']
    count_df = count_df.sort_values(by=column)

    # https://www.geeksforgeeks.org/how-to-plot-a-smooth-curve-in-matplotlib/
    # https://www.kaggle.com/code/aishwaryanagchandi/student-performance-factors-eda/notebook
    # https://medium.com/data-science/histograms-and-density-plots-in-python-f6bda88f5ac0
    X_Y_Spline = make_interp_spline(
        count_df[column],
        count_df["count"],
    )
    X_=np.linspace(count_df[column].min(), count_df[column].max(), 500)
    Y_=X_Y_Spline(X_)

    plt.plot(
        X_,
        Y_,
        "-o",
        linewidth=0.7,
        markersize=0.5,
        color=colors[i]
    )
    plt.fill_between(
        X_,
        Y_,
        alpha=0.4,
        color=colors[i]
    )
    plt.xlabel(column)
    
plt.tight_layout(pad=10)

theme.apply_transforms()
plt.savefig(
    KDE_plots_of_numerical_features_path, 
    format="pdf", 
    bbox_inches="tight"
)
        


