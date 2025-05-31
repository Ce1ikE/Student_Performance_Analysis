from main_config import (
    filename_results_base_models_table_pdf,
    filename_results_base_models_csv,
)


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plottable import ColumnDefinition, Table

cmap_R2 = LinearSegmentedColormap.from_list(
    name="R2", colors=[
        "#f7f7f7", 
        "#e9a3c9", 
        "#a1d76a", 
        "#4d9221",
    ], 
    N=256
)
cmap_metrics = LinearSegmentedColormap.from_list(
    name="errors", colors=[
        "#7add3c",
        "#a1d76a", 
        "#cfd6c8", 
        "#e7d2d2", 
        "#e7a3a5", 
        "#e55555", 
    ]
    , N=256
)

columns = [
    "cv_mean_r2",
    "cv_std_r2",
    "accuracy",
    "R2",
    "MSE",
    "RMSE",
    "MAE",
    "training_time"
]

base_models_results = pd.read_csv(filename_results_base_models_csv)
base_models_results.columns = ["Model"] + columns

print(
    base_models_results.head()
)

column_def = (
    ColumnDefinition(
        name="Model",
        width=1.5,
        textprops={
            "ha": "center",
        },
        group="Model",
    ),
    ColumnDefinition(
        name="R2",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_R2,
        group="Metrics",
    ),
    ColumnDefinition(
        name="RMSE",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_metrics,
        group="Metrics",
    ),
    ColumnDefinition(
        name="MAE",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_metrics,
        group="Metrics",
    ),
    ColumnDefinition(
        name="MSE",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_metrics,
        group="Metrics",
    ),
    ColumnDefinition(
        name="accuracy",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_metrics.reversed(),
        group="Metrics",
    ),
    ColumnDefinition(
        name="cv_mean_r2",
        width=0.75,
        textprops={
            "ha": "center",
        },
        group="Cross-Validation",
    ),
    ColumnDefinition(
        name="cv_std_r2",
        width=0.75,
        textprops={
            "ha": "center",
        },
        group="Cross-Validation",
    ),
    ColumnDefinition(
        name="training_time",
        width=0.75,
        textprops={
            "ha": "center",
            "bbox": {"boxstyle": "circle", "pad": 0.35},
        },
        cmap=cmap_metrics,
        group="Time",
    ),
)

fig, ax = plt.subplots(figsize=(18, 10))
tab = Table(
    base_models_results.round(3),
    column_definitions=column_def,
    odd_row_color="#76cfde85",
    even_row_color="#416b717e",
    row_dividers=True,
    footer_divider=True,
    ax=ax,
    textprops={
        "fontsize": 10, 
        "fontfamily": "monospace"
    },
    row_divider_kw={
        "linewidth": 1, 
        "linestyle": (0, (1, 5))
    },
    col_label_divider_kw={
        "linewidth": 1, 
        "linestyle": "-"
    },
    column_border_kw={
        "linewidth": 1, 
        "linestyle": "-"
    },
)


plt.savefig(
    filename_results_base_models_table_pdf,
    format="pdf",
)


