# Load the Data
import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Boston.csv")
pd.set_option('display.max_columns', None)

# Auto EDA
# ---------
# Sweetviz
# Autoviz
# Dtale
# Pandas Profiling
# Dataprep


# Sweetviz
###########
#pip install sweetviz
import sweetviz as sv

s = sv.analyze(df)
s.show_html()

df.value_counts()
df.nunique()

print(df.describe())
df.head(5)


# Autoviz
###########
# pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class



av = AutoViz_Class()
a = av.AutoViz(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Cars.csv", chart_format = 'html')

a.HP = a.HP.astype('float64')
a.VOL = a.VOL.astype('float64')

import os
os.getcwd()

# If the dependent variable is known:
a = av.AutoViz(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Cars.csv", depVar = 'MPG', chart_format = 'html')  # depVar - target variable in your dataset


# D-Tale
########

# pip install dtale   # In case of any error then please install werkzeug appropriate version (pip install werkzeug==2.0.3)
import dtale
import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/InClass_DataPreprocessing_datasets/education.csv")

d = dtale.show(df)
d.open_browser()


# Pandas Profiling
###################

# pip install pandas_profiling / pip install ydata-profiling
# from pandas_profiling import ProfileReport 
from ydata_profiling import ProfileReport 

p = ProfileReport(df)
p
p.to_file("output.html")

import os
os.getcwd()

# Dataprep
##########

# pip install dataprep
from dataprep.eda import create_report

report = create_report(df, title = 'My Report')

report.show_browser()


