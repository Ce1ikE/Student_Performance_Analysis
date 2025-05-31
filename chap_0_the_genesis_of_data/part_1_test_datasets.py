import pandas as pd
import os

colorDictionary = {
    0  : "\033[38;5;196m",#"RED"
    1  : "\033[38;5;119m",#"GREEN"
    2  : "\033[38;5;21m",#"BLUE"
    3  : "\033[38;5;129m",#"VIOLET"
    4  : "\033[38;5;90m",#"PURPLE"
    5  : "\033[38;5;198m",#"PINK"
    6  : "\033[38;5;87m",#"CYAN"
    7  : "\033[38;5;202m",#"ORANGE"
    8  : "\033[38;5;226m",#"YELLOW"
    9  : "\033[38;5;172m",#"GOLD"
    10 : "\033[38;5;37m",#"TURQUOISE"
    11 : "\033[0m", #"RESET_COLOR"
}

file_and_location = [
    ["student-por.csv"                      , "https://www.kaggle.com/datasets/larsen0966/student-performance-data-set"],
    ["StudentPerformanceFactors.csv"        , "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"],
    ["student_performance_prediction.csv"   , "https://www.kaggle.com/datasets/souradippal/student-performance-prediction"],
    ["ResearchInformation3.csv"             , "https://data.mendeley.com/datasets/5b82ytz489/1"],
    ["DATA (1).csv"                         , "https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation"],
    ["Students_Grading_Dataset.csv"         , "https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset"],
]


location = os.path.dirname(__file__)
max_entries = len(colorDictionary.values())

for index in range(len(file_and_location)):
    
    file , source = file_and_location[index]

    print(colorDictionary[index % max_entries] + source)
    
    pd.read_csv(os.path.join(location ,file)).info()

# DATASET 2 was allowed StudentPerformanceFactors.csv