import os
import time
import datetime

script_path = os.path.dirname(__file__)

CURSOR = ">>> "
RANDOM_SEED = 5
HYPERPARAMETER_SWEEP = 1  

GITHUB = "https://github.com/Ce1ikE"
PROJECT_REPO = "Student_Performance_Analysis"
GITHUB_URL = f"{GITHUB}/{PROJECT_REPO}"
TM_COURSE_URL = "https://thomasmore.be/nl/opleidingen/professionele-bachelor/elektronica-ict/applied-artificial-intelligence/sint-katelijne-waver/basistraject"

COLORS_TERMINAL = {
    "RED" : "\033[38;5;196m",
    "GREEN" : "\033[38;5;119m",
    "BLUE" : "\033[38;5;21m",
    "VIOLET" : "\033[38;5;129m",
    "PURPLE" : "\033[38;5;90m",
    "PINK" : "\033[38;5;198m",
    "CYAN" : "\033[38;5;87m",
    "ORANGE" : "\033[38;5;202m",
    "YELLOW" : "\033[38;5;226m",
    "GOLD" : "\033[38;5;172m",
    "TURQUOISE" : "\033[38;5;37m",
    "WHITE" : "\033[38;5;15m",
    "BLACK" : "\033[38;5;0m",
    "GRAY" : "\033[38;5;245m",
    "DARK_GRAY" : "\033[38;5;236m",
    "LIGHT_RED" : "\033[38;5;196m",
    "LIGHT_GREEN" : "\033[38;5;119m",
    "LIGHT_BLUE" : "\033[38;5;21m",
    "LIGHT_VIOLET" : "\033[38;5;129m",
    "LIGHT_PURPLE" : "\033[38;5;90m",
    "LIGHT_PINK" : "\033[38;5;198m",
    "LIGHT_CYAN" : "\033[38;5;87m",
    "LIGHT_ORANGE" : "\033[38;5;202m",
    "LIGHT_YELLOW" : "\033[38;5;226m",
    "LIGHT_GOLD" : "\033[38;5;172m",
    "LIGHT_TURQUOISE" : "\033[38;5;37m",
    "RESET_COLOR" : "\033[0m",
}

COLORS_BG = {
    "BLACK_BG" : "\033[40m",
    "RED_BG" : "\033[41m",
    "GREEN_BG" : "\033[42m",
    "YELLOW_BG" : "\033[43m",
    "BLUE_BG" : "\033[44m",
    "MAGENTA_BG" : "\033[45m",
    "CYAN_BG" : "\033[46m",
    "WHITE_BG" : "\033[47m",
}

COLORS_BOLD = {
    "BOLD" : "\033[1m",
    "BOLD_RED" : "\033[1;38;5;196m",
    "BOLD_GREEN" : "\033[1;38;5;119m",
    "BOLD_BLUE" : "\033[1;38;5;21m",
    "BOLD_VIOLET" : "\033[1;38;5;129m",
    "BOLD_PURPLE" : "\033[1;38;5;90m",
    "BOLD_PINK" : "\033[1;38;5;198m",
    "BOLD_CYAN" : "\033[1;38;5;87m",
    "BOLD_ORANGE" : "\033[1;38;5;202m",
    "BOLD_YELLOW" : "\033[1;38;5;226m",
    "BOLD_GOLD" : "\033[1;38;5;172m",
    "BOLD_TURQUOISE" : "\033[1;38;5;37m",
    "BOLD_WHITE" : "\033[1;38;5;15m",
    "BOLD_BLACK" : "\033[1;38;5;0m",
    "BOLD_GRAY" : "\033[1;38;5;245m",
    "BOLD_DARK_GRAY" : "\033[1;38;5;236m",
    
    "UNDERLINE" : "\033[4m",
}

COLORS_ITALIC = {
    "ITALIC" : "\033[3m",
    "ITALIC_RED" : "\033[3;38;5;196m",
    "ITALIC_GREEN" : "\033[3;38;5;119m",
    "ITALIC_BLUE" : "\033[3;38;5;21m",
    "ITALIC_VIOLET" : "\033[3;38;5;129m",
    "ITALIC_PURPLE" : "\033[3;38;5;90m",
    "ITALIC_PINK" : "\033[3;38;5;198m",
    "ITALIC_CYAN" : "\033[3;38;5;87m",
    "ITALIC_ORANGE" : "\033[3;38;5;202m",
    "ITALIC_YELLOW" : "\033[3;38;5;226m",
    "ITALIC_GOLD" : "\033[3;38;5;172m",
    "ITALIC_TURQUOISE" : "\033[3;38;5;37m",
    "ITALIC_WHITE" : "\033[3;38;5;15m",
    "ITALIC_BLACK" : "\033[3;38;5;0m",
    "ITALIC_GRAY" : "\033[3;38;5;245m",
    "ITALIC_DARK_GRAY" : "\033[3;38;5;236m",
}


INTRO_LOGO = f"""
{COLORS_ITALIC['ITALIC_PURPLE']}
From the depths of data, we rise,
            to sculpt the ideal model,
                with precision and wisdom, we strive.
{COLORS_TERMINAL['RESET_COLOR']}
{COLORS_BOLD['BOLD_GRAY']}
            
               .+i+;I:
              :=..    t:
             =;+;:..   ii.
            +I::;=:;;+t=i;=.
            +;;;+;i.;;:It++i;             
          ;X  t+=+=;;i;=iItt+V
          :t  =ii+.=.;:=+ii++iIY
          :R   i=ti+=+;i+i=t=++:+Ii+==
          :R  .+iii==;;itt=++i=i::=YRBBBMRRVY+;
           ;+    +i+;+;+itiiitii+i+i .=iYVIi+iitI=;=
   +. ::.X:.;   .:=;+=i=ii++YYIIiiiIt.  ;YI::+:;=iit===;
  I;:. .  :+:YI;R..=;;=i+titIVItIYVYYYXVX=+:.....;;+t=+::=
  +i;.::......:;:=;;;;;=+iii=+=++ii++tttVI===;;;;::;;+;tti=
   tI+.::::.;::;:=+++i=+;i++ititttItIItt=;=t+==;:;::;:;=+IY=:
    :=i;::.::::;=:;++=i===;iiittitttttItt=;=;:;;...::;::;.;+ii:;
      :=+::.;+i+t++itiIIY=RRRXXV+VYi===:::;;:.:.........::;;;:;;;;:;;;;
          :tYti=;=+;+;=+++=;iIVRRRRVVRXRYYYV=;=::::..........:.:==+i==;;==;;:
            ;Xti;=;+i;+ti++=+tRBBBYBVRYXIVtYY++=..:........:.;;::==;::;.;;;
              YVi==;++:I;;ii+IRXIYIY=:;i;i;=;;;;;.........;:::;:;=;..:;::
              :=XI=+iItIiit=:IXRRIItiXiIYiIt;I==:.......:..:....;:........
              :BWRRV;YRIXY...+YRRVYVR+XIRItitI++=:.....;:.........:....:.::..
             ==+RWBXtIRRV+.+IiYRBYBRRYYIRI;VitI;=;..........:::.::;::::...;;;:.
{COLORS_TERMINAL['RESET_COLOR']}
"""

CREDITS = f"""
{COLORS_TERMINAL['RESET_COLOR']}
This project was created and brought to you by the creator of
{COLORS_BOLD['BOLD_TURQUOISE']}
 ________                     __      __                             ___             ______                                                              
/        |                   /  |    /  |                           /   \           /      \                                                             
$$$$$$$$/______    _______  _$$ |_   $$/   _______   _______       /$$$  |         /$$$$$$  |  ______   _______    ______   __    __   ______    ______  
   $$ | /      \  /       |/ $$   |  /  | /       | /       |      $$ $$ \__       $$ |  $$/  /      \ /       \  /      \ /  |  /  | /      \  /      \ 
   $$ | $$$$$$  |/$$$$$$$/ $$$$$$/   $$ |/$$$$$$$/ /$$$$$$$/       /$$$     |      $$ |      /$$$$$$  |$$$$$$$  |/$$$$$$  |$$ |  $$ |/$$$$$$  |/$$$$$$  |
   $$ | /    $$ |$$ |        $$ | __ $$ |$$ |      $$      \       $$ $$ $$/       $$ |   __ $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$    $$ |$$ |  $$/ 
   $$ |/$$$$$$$ |$$ \_____   $$ |/  |$$ |$$ \_____  $$$$$$  |      $$ \$$  \       $$ \__/  |$$ \__$$ |$$ |  $$ |$$ \__$$ |$$ \__$$ |$$$$$$$$/ $$ |      
   $$ |$$    $$ |$$       |  $$  $$/ $$ |$$       |/     $$/       $$   $$  |      $$    $$/ $$    $$/ $$ |  $$ |$$    $$ |$$    $$/ $$       |$$ |      
   $$/  $$$$$$$/  $$$$$$$/    $$$$/  $$/  $$$$$$$/ $$$$$$$/         $$$$/$$/        $$$$$$/   $$$$$$/  $$/   $$/  $$$$$$$ | $$$$$$/   $$$$$$$/ $$/       
                                                                                                                       $$ |                              
                                                                                                                       $$ |                              
                                                                                                                       $$/                             

{COLORS_TERMINAL['RESET_COLOR']}
Originally created by E.C. as part of a data science course : Python for A.I
It is inspired by the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.
Feel free to reach out with any questions or feedback.
You can find the code and more information on my GitHub repository.
Link : {GITHUB_URL}

Copyright © Thomas More Mechelen-Antwerpen vzw - Campus De Nayer - Professionele bachelor elektronica-ict - {datetime.datetime.now().year}
Link : {TM_COURSE_URL}
"""

raw_data_path = os.path.join(script_path,"assets","StudentPerformanceFactors.csv")

train_data_cleaned_path     = os.path.join(script_path, "assets","StudentPerformanceFactors_cleaned_train.csv")
test_data_cleaned_path      = os.path.join(script_path, "assets","StudentPerformanceFactors_cleaned_test.csv")
data_cleaned_path           = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_full.csv")
full_dataset_cleaned_path   = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_full_dataset.csv")

train_data_cleaned_no_outliers_path     = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_test_no_outliers.csv")
test_data_cleaned_no_outliers_path      = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_train_no_outliers.csv")
data_cleaned_no_outliers_path           = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_full_no_outliers.csv")
full_dataset_cleaned_no_outliers_path   = os.path.join(script_path,"assets","StudentPerformanceFactors_cleaned_full_dataset_no_outliers.csv")

report_assets_path = os.path.join(script_path,"report","assets")

KDE_Exam_Score_distribution_test_vs_train_with_outliers_path     = os.path.join(report_assets_path,"Exam_Score_Distribution_test_vs_train_without_outliers.pdf")
KDE_Exam_Score_distribution_test_vs_train_without_outliers_path  = os.path.join(report_assets_path,"Exam_Score_Distribution_test_vs_train_with_outliers.pdf")

correlation_matrix_path                         = os.path.join(report_assets_path,"correlation_matrix.pdf")
correlation_matrix_new_features_path            = os.path.join(report_assets_path,"correlation_matrix_new_features.pdf")
scatter_plots_exploration_path                  = os.path.join(report_assets_path,"scatter_plots_exploration.pdf")
KDE_plots_of_numerical_features_path            = os.path.join(report_assets_path,"KDE_plots_of_numerical_features.pdf")

filename_results_base_models_pdf                = os.path.join(report_assets_path,"base_models_results.pdf")
filename_results_base_models_table_pdf          = os.path.join(report_assets_path,"base_models_results_table.pdf")
filename_results_base_models_csv                = os.path.join(report_assets_path,"base_models_results.csv")

filename_results_ensemble_methods_pdf           = os.path.join(report_assets_path,"ensemble_methods_results.pdf")
filename_results_ensemble_methods_table_pdf     = os.path.join(report_assets_path,"ensemble_methods_results_table.pdf")
filename_results_ensemble_methods_csv           = os.path.join(report_assets_path,"ensemble_methods_results.csv")

models_dir                              = os.path.join(script_path,"models")
filename_results_best_models_csv        = "best_models_result.csv"
filename_results_best_models_pdf        = "best_models_result.pdf"
filename_results_best_models_table_pdf  = "best_models_result_table.pdf"

filename_results_NN_pdf                 = "nn_results.pdf"

best_nn_model = os.path.join(models_dir, "nn10_", "nn10_custom_MLP.h5")

shap_results_waterfall_path = os.path.join(report_assets_path,"SHAP_values_waterfall.pdf")
shap_results_bar_path       = os.path.join(report_assets_path,"SHAP_values_bar.pdf")
shap_results_beeswarm_path  = os.path.join(report_assets_path,"SHAP_values_beeswarm.pdf")


target_column = [
    "Exam_Score",
]

selected_columns = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Parental_Involvement",
    "Access_to_Resources",
    "Parental_Education_Level",
]
    

def type_out(text, delay=0.001):
    print(COLORS_TERMINAL['RESET_COLOR'])
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print("\n")

def print_cursor():
    print(f"{COLORS_BOLD['BOLD_PURPLE']}{CURSOR}{COLORS_TERMINAL['RESET_COLOR']}", end='')
    input_text = input()
    return input_text.strip()

def show_intro():
    print(COLORS_TERMINAL['RESET_COLOR'])
    print(INTRO_LOGO)
    type_out("Welcome to the Student Performance Analysis Project!")
    type_out("This project explores factors affecting student performance and builds predictive models.")
    type_out("Let's get started!\n")
    print(COLORS_TERMINAL['RESET_COLOR'])

def show_outro():
    print(COLORS_TERMINAL['RESET_COLOR'])
    type_out("Thank you for exploring the Student Performance Analysis Project!")
    type_out("We hope you found the insights valuable.")
    type_out("Feel free to reach out with any questions or feedback.")
    type_out("Goodbye!\n")
    print(CREDITS)
    print(COLORS_TERMINAL['RESET_COLOR'])
