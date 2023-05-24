import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib as mplib
from matplotlib import pyplot as plt
import seaborn as sns
import pyro
import seaborn as sns
sns.set_style('whitegrid')
data = pd.read_csv("Data/Data_processed.csv", sep=';')
data2 = pd.read_csv("Data/Data_processed2.csv", sep=';')    

# plot for each country the life expectancy over time together with the GDP, Polio and OverweightOfAdults%
print(data.info())
for i, country in enumerate(data['Country'].unique()):
    d = data[data['Country'] == country]
    d2 = data2[data2['Country'] == country]
    # only plot if d2 contains a nan value
    plt.plot(d['Year'], d2['Hepatitis B'], label='Life expectancy ')
    plt.plot(d['Year'], d['Hepatitis B'], label='Life expectancy ')

    plt.title(country)
    plt.legend()
    plt.show()
