from main_config import report_assets_path , filename_results_best_models_csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from plottable import ColumnDefinition, Table
import textwrap

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
    "training_time",
    "best_params",
]

def main(model_prefix: str):
    filename_results_csv = model_prefix + filename_results_best_models_csv
    base_models_results = pd.read_csv(os.path.join(report_assets_path,filename_results_csv))
    base_models_results.columns = ["Model"] + columns

    print(
        base_models_results.head()
    )

    base_models_results.rename(
        columns={
            "cv_mean_r2": "CV mean_r2",
            "cv_std_r2": "CV std_r2",
            "accuracy": "Accuracy",
            "R2": "R2",
            "MSE": "MSE",
            "RMSE": "RMSE",
            "MAE": "MAE",
            "training_time": "Training time (s)",
            "best_params": "Best parameters",
        },
        inplace=True,
    )

    def wrap_best_params(params: dict, width=40):
        dict_as_text = "\n".join(textwrap.wrap(str(params), width=width))
        dict_as_text = dict_as_text.replace("'", "")
        dict_as_text = dict_as_text.replace("{", "")
        dict_as_text = dict_as_text.replace("}", "")
        dict_as_text = dict_as_text.replace(":", ": ")
        dict_as_text = dict_as_text.replace(",", "")
        return dict_as_text

    def wrap_text(text: str, width=40):
        return "\n".join(textwrap.wrap(text, width=width))

    column_def = (
        ColumnDefinition(
            name="Model",
            width=2.5,
            formatter=lambda x: wrap_text(x, width=40),
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
            name="Accuracy",
            width=0.75,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            cmap=cmap_metrics.reversed(),
            group="Metrics",
        ),
        ColumnDefinition(
            name="CV mean_r2",
            width=0.75,
            textprops={
                "ha": "center",
            },
            group="Cross-Validation",
        ),
        ColumnDefinition(
            name="CV std_r2",
            width=0.75,
            textprops={
                "ha": "center",
            },
            group="Cross-Validation",
        ),
        ColumnDefinition(
            name="Training time (s)",
            width=0.75,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            cmap=cmap_metrics,
            group="Time",
        ),
        ColumnDefinition(
            name="Best parameters",
            width=2.5,
            formatter=lambda x: wrap_best_params(x, width=40),
            textprops={
                "ha": "left",
                "fontsize": 10,
                "fontfamily": "monospace",
            },
            group="Parameters",
        ),
    )

    fig, ax = plt.subplots(figsize=(18, 14))
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
        os.path.join(report_assets_path, model_prefix + "best_models_result_table.pdf"), 
        format="pdf",
    )


