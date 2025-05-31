from main_config import (
    train_data_cleaned_path,
    test_data_cleaned_path,
    data_cleaned_path,
    KDE_Exam_Score_distribution_test_vs_train_with_outliers_path,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.interpolate import make_interp_spline
from drawarrow import ax_arrow
from aquarel import load_theme

# --------------------- Exam_Score Distribution - Test Set vs Train Set ----------------------------- #
df_test = pd.read_csv(test_data_cleaned_path)
df_train = pd.read_csv(train_data_cleaned_path)
df_test_train = pd.read_csv(data_cleaned_path)
# ----------------------------------------------------------------------------------------------------------

colors = [
    "#56F65EAC",
    "#382EF7B8",
    "#56F7A0B8",
    "#00FFEAA4",
]
datasets = [
    "train",
    "test",
]
data = {}
theme = load_theme("umbra_dark")
theme.apply()

fig = plt.figure(figsize=(10, 2.7), layout="constrained")
plt.title(
    label='Exam_Score Distribution - Test Set vs Train Set',
    pad=10,
    fontdict={
        "fontsize": 16,
        "fontweight": "bold",
        "fontfamily": "monospace"
    },
)
plt.xlim(
    df_test_train["Exam_Score"].min() - 1,
    df_test_train["Exam_Score"].max() + 1
)
plt.ylim(0, df_test_train["Exam_Score"].value_counts().max() + 1)

# removing outliers from datasets based upon outliers  => ] -inf , Q1 - 1.5 * IQR [U] Q3 + 1.5 * IQR , +inf [
q1 = df_test_train["Exam_Score"].quantile(0.25)
q3 = df_test_train["Exam_Score"].quantile(0.75)
less_than_q1_IQR = q1 - 1.5 * (q3 - q1)
more_than_q3_IQR = q3 + 1.5 * (q3 - q1)
df_test = df_test[
    (df_test["Exam_Score"] >= less_than_q1_IQR) &
    (df_test["Exam_Score"] <= more_than_q3_IQR)
]
df_train = df_train[
    (df_train["Exam_Score"] >= less_than_q1_IQR) &
    (df_train["Exam_Score"] <= more_than_q3_IQR)
]
df_test_train = df_test_train[
    (df_test_train["Exam_Score"] >= less_than_q1_IQR) &
    (df_test_train["Exam_Score"] <= more_than_q3_IQR)
]
# ----------------------------------------------------------------------------------------------------------

def add_connected_scatter(
    data:pd.DataFrame,
    axes: Axes=None
):

    # https://www.geeksforgeeks.org/how-to-plot-a-smooth-curve-in-matplotlib/
    # https://www.kaggle.com/code/aishwaryanagchandi/student-performance-factors-eda/notebook
    # https://medium.com/data-science/histograms-and-density-plots-in-python-f6bda88f5ac0
    X_Y_Spline = make_interp_spline(
        data["Exam_Score"],
        data["count"],
    )
    X_=np.linspace(data["Exam_Score"].min(), data["Exam_Score"].max(), 500)
    Y_=X_Y_Spline(X_)

    if axes is None:
        axes = plt.gca()
        
    axes.plot(
        X_,
        Y_,
        "-o",
        linewidth=0.7,
        markersize=0.5,
        color=colors[i]
    )
    axes.fill_between(
        X_,
        Y_,
        alpha=0.4,
        color=colors[i]
    )

def add_outliers(
    data: pd.DataFrame,
    datasets_train_test: dict,
    axes: Axes = None, 
    color_q1_q3: str = "#00FFE5A2",
    color_outliers: str = "#FE5959D9",
):
    if axes is None:
        axes = plt.gca()

    q3 = data["Exam_Score"].quantile(0.75)
    q1 = data["Exam_Score"].quantile(0.25)

    # see "math for AI" course
    # IQR == Interquartile Range which is the range between the 25% and 75% percentiles    
    # Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
    count_df = data["Exam_Score"].value_counts().reset_index()  
    count_df.columns = ["Exam_Score",'count']
    count_df = count_df.sort_values(by="Exam_Score")
    plt.axvline(
        q1 - 1.5 * (q3 - q1),
        color=color_q1_q3,
        linestyle="--",
        linewidth=3,
        alpha=0.7,
        label=fr"$\mathrm{{Q1 - 1.5 \cdot IQR}}$"
    )
    plt.axvline(
        q3 + 1.5 * (q3 - q1),
        color=color_q1_q3,
        linestyle="--",
        linewidth=3,
        alpha=0.7,
        label=fr"$\mathrm{{Q3 + 1.5 \cdot IQR}}$"
    )
    ax_arrow(
        tail_position=(q1 - 1.5 * (q3 - q1), count_df["count"].max()//2),
        head_position=(count_df["Exam_Score"].min(), count_df["count"].max()//2),   
        color=color_outliers,
        linewidth=1.5,
    )
    plt.gca().add_patch(
        Ellipse(
            xy=(q1 - 1.5 * (q3 - q1), count_df["count"].max()//2),
            width=0.1,
            height=10,
            color=color_outliers,
            alpha=0.4,
        )
    )
    ax_arrow(
        tail_position=(q3 + 1.5 * (q3 - q1), count_df["count"].max()//2),
        head_position=(count_df["Exam_Score"].max(), count_df["count"].max()//2),   
        color=color_outliers,
        linewidth=1.5,
    )    
    plt.gca().add_patch(
        Ellipse(
            xy=(q3 + 1.5 * (q3 - q1), count_df["count"].max()//2),
            width=0.1,
            height=10,
            color=color_outliers,
            alpha=0.4,
        )
    )

    outliers = pd.DataFrame(columns=["Exam_Score", "count"])
    for dataset_name , count_df in datasets_train_test.items():

        # outliers  => ] -inf , Q1 - 1.5 * IQR [U] Q3 + 1.5 * IQR , +inf [
        outliers_left = count_df[count_df["Exam_Score"] < (q1 - 1.5 * (q3 - q1))]
        outliers_right = count_df[count_df["Exam_Score"] > (q3 + 1.5 * (q3 - q1))]
        outliers = pd.concat([outliers, outliers_right], axis=0)
        outliers = pd.concat([outliers, outliers_left], axis=0)

    if not outliers.empty:
        axes.scatter(
            outliers["Exam_Score"],
            outliers["count"],
            color=color_outliers,
            marker="x",
            s=125,
            label="Outliers"
        )
# ----------------------------------------------------------------------------------------------------------


for i, dataset in enumerate(datasets):
    if dataset == "test":
        df = df_test
    elif dataset == "train":
        df = df_train

    count_df = df["Exam_Score"].value_counts().reset_index()  
    count_df.columns = ["Exam_Score",'count']
    count_df = count_df.sort_values(by="Exam_Score")
    data[dataset] = count_df

    add_connected_scatter(count_df)

plt.axvline(
    df_test_train["Exam_Score"].max(),
    color=colors[2], 
    linestyle=":", 
    linewidth=3, 
    alpha=0.7,
    label=fr"$\mathrm{{Max Exam Score}}$",
)
plt.axvline(
    df_test_train["Exam_Score"].min(),
    color=colors[2], 
    linestyle=":", 
    linewidth=3, 
    alpha=0.7,
    label=fr"$\mathrm{{Min Exam Score}}$"
)

add_outliers(df_test_train,data)

plt.xticks([])
plt.xlabel("Exam_Score",labelpad=10,fontdict={"fontsize": 10, "fontweight": "bold", "fontfamily": "monospace"})
plt.ylabel("Count",labelpad=10,fontdict={"fontsize": 10, "fontweight": "bold", "fontfamily": "monospace"})
fig.legend(
    fontsize=10, 
    frameon=True,
    loc='outside upper right',
    ncol=3, fancybox=True, shadow=True,
    title="Legend",
)
plt.tight_layout()
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

theme.apply_transforms()

# https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
manager = plt.get_current_fig_manager()
plt.gcf().set_size_inches(15, 6)



plt.savefig(
    KDE_Exam_Score_distribution_test_vs_train_with_outliers_path, 
    format="pdf", 
)    




