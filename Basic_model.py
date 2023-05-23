import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib as mplib
from matplotlib import pyplot as plt
import seaborn as sns
import pyro
import seaborn as sns
sns.set_style('whitegrid')

data = pd.read_csv("Data/addtional_infor_life.csv", sep=';')
data = data.drop(['Unnamed: 0','Adult Mortality','under-five deaths ','percentage expenditure','Income composition of resources',' thinness 5-9 years'],axis=1)
data.dropna(inplace=True)
print(data.info())

# print all unique years

# make subplot with all the numerical columns
# num_cols = data.select_dtypes(include=['int64', 'float64']).drop('Year', axis=1)
# num_cols.hist(figsize=(20,20), bins=30, grid=False, layout=(7,3), color='b')
# plt.show()

# loop over all countries and plot life expectancy over time
for i, country in enumerate(data['Country'].unique()):
    d = data[data['Country'] == country]
    plt.plot(d['Year'], d['Life expectancy '], label=country)
    if i % 10 == 0:
        plt.legend()
        plt.show()


# GDP 
year = 2014
d = data[data['Year'] == year]
# sort data by counytrn name
d = d.sort_values(by='Country')
print(d)

for col in data.columns:
    # convert to numpy and print sorted alphabetically by country
    for i, v in enumerate(d[col].values):
        print(d['Country'].values[i], v)


    plt.plot(d[col], d['Life expectancy '], 'o', color='blue')
    plt.title(year)
    plt.xlabel(col)
    plt.ylabel('Life expectancy ')
    # annotate each point with country name
    for i, txt in enumerate(d['Country']):
        plt.annotate(txt, (d[col].iloc[i], d['Life expectancy '].iloc[i]))
    plt.show()

# infant death per 1000 can be too high - very weird values
# Alcohol seems right for those that isn't 0.01, but it's not clear what 0.01 means
# Hepatitis B all low values are wrong
# Measles very weird
# BMI can be too high, doesn't allign with wikipedia
# Polio doesn't allign with WHO and low values all wrong
# Diphteria doesn't allign with WHO and low values all wrong
