# Data Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data into Python
education = pd.read_csv(r"C:\Users\Lenovo\Downloads\Study material\EDA\InClass_DataPreprocessing_datasets\education.csv")

# Read data into Python
education.shape

print(education.describe())

print(education.columns)

print(education.dtypes)

plt.bar(height = education.workex, x = np.arange(1, 774, 1))
plt.hist(education.gmat, density=True, alpha=0.5, color='b')

# barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1)) # initializing the parameter

# Histogram
plt.hist(education.gmat) # histogram
plt.hist(education.gmat, bins = [600, 680, 710, 740, 780], color = 'green', edgecolor="red") 
plt.hist(education.workex)
plt.hist(education.workex, color='red', edgecolor = "black", bins = 6)

help(plt.hist)

# Histogram using Seaborn
import seaborn as sns
sns.distplot(education.gmat) # Deprecated histogram function from seaborn

# Histogram from seaborn


# Boxplot
plt.figure()
plt.boxplot(education.gmat) # boxplot

help(plt.boxplot)


# Density Plot
sns.kdeplot(education.gmat) # Density plot
sns.kdeplot(education.gmat, bw = 0.5 , fill = True)
sns.kdeplot(education.workex, bw= 0.5, fill = True) 

# Descriptive Statistics
# describe function will return descriptive statistics including the 
# central tendency, dispersion and shape of a dataset's distribution.

education.describe()


# Bivariate visualization
# Scatter plot
import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv("C:/Data/Cars.csv")

cars.info()

plt.scatter(x = cars['HP'], y = cars['MPG']) 

plt.scatter(x = cars['HP'], y = cars['SP'], color = 'green') 

