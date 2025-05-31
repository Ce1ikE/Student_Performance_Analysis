from chap_1_peering_into_the_data_abyss.config import df_raw
from main_config import scatter_plots_exploration_path  

import matplotlib.pyplot as plt
import seaborn as sns
from aquarel import load_theme

theme = load_theme("umbra_dark")
theme.apply()

titles = [
    "Exam score vs Hours studied",
    "Exam score vs Attendance",
    "Exam score vs Previous scores",
]

x_y_plot_scatter = [
    ["Hours_Studied"],
    ["Attendance"],
    ["Previous_Scores"],
]

colors = [
    "#1105F64B",
    "#DEE51668",
    "#51E5163F",
]
color_line = "#F51616B8"


rows = len(x_y_plot_scatter) // 2 + 1
plt.figure(figsize=(9,12))
for i in range(1,len(x_y_plot_scatter) + 1):
    axes = plt.subplot(rows,2,i)
    plt.title(
        titles[i - 1],
        fontdict={
            "fontsize": 14,
            "fontweight": "bold",
            "fontfamily": "monospace",
        },
        pad=10,
        loc="left",
    )
    sns.regplot(
        x=df_raw[x_y_plot_scatter[i - 1]],
        y=df_raw["Exam_Score"],
        marker='o',
        label= "Data Points",
        scatter_kws={
            "s": 9, 
            "alpha": 0.5, 
            "linewidths": 0.2, 
            "color": colors[i - 1],
        },
        line_kws={
            "color": color_line, 
            "alpha": 0.7,
            "label": "Regression Line",
        },
    )
    plt.xlabel(
        x_y_plot_scatter[i - 1][0].replace("_"," "),
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",
        loc="left",
        labelpad=10,
    )
    plt.ylabel(
        "Exam Score",
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",
        loc="bottom",
        labelpad=10,
    )
    plt.legend(
        fontsize=10, 
        frameon=True,
        loc='upper right',
        fancybox=True, shadow=True,
        title="Legend",
    )
    

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


theme.apply_transforms()
plt.tight_layout(pad=2.0,h_pad=3.5)
plt.gcf().set_size_inches(15, 10)
plt.savefig(
    scatter_plots_exploration_path, 
    format="pdf", 
    bbox_inches="tight"
)
    
