import numpy as np # linear algebra
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd # data processing
import matplotlib as mplib
from matplotlib import pyplot as plt
import seaborn as sns
import pyro
import seaborn as sns
sns.set_style('whitegrid')

data = pd.read_csv("Data/DataSetReconstruct.csv", sep=';')
print(data.info())
# log transform GDP, Population and MeaslesPrMillion

data['GDP'] = np.log(data['GDP'] + 1)
data['Population'] = np.log(data['Population'] + 1)
data['MeaslesPrMillion'] = np.log(data['MeaslesPrMillion'] + 1)
import time
# print total amount of nan values in entire dataset
print(data.isnull().sum().sum())
time.sleep(5)
for col in data.columns:
    if col == 'Country' or col == 'Year':
        continue
    # check if column is numerical
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        # replace missing values with closest year of missing value of that country
        for country in data['Country'].unique():
            d = data[data['Country'] == country]
            # check if there are missing values
            if d[col].isnull().sum() > 0:
                # get index of missing values
                idx = d[d[col].isnull()].index.values
                # loop over all missing values
                # print(idx, col)
                for i in idx:
                    # print(i)
                    # get year of missing value by indexing on the row index
                    year = data['Year'].loc[i] 
                    # get index of closest year
                    list_of_closest_years = (d['Year'] - year).abs().argsort()[::-1]

                    for new_year in list_of_closest_years.index:
                        new_data = d[col].loc[new_year]
                        # overwrite missing value with closest year
                        if np.isnan(new_data):
                            continue
                        data[col][i] = new_data
                        print('Replaced missing value of {} in {} with {}'.format(col, country, new_data))
                        print(data[col][i])
                        break

# check if there are still missing values
# replace remaining nan values by mean of column
for col in data.columns:
    if col == 'Country' or col == 'Year':
        continue
    # check if column is numerical
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        # replace missing values with mean of column
        data[col] = data[col].fillna(data[col].mean())
# only keep relevant columns
# relevant columns: Country; continent; Life expectancy ;Year;Status;infant deaths;AlcoholHepatitis B;MeaslesPrMillion;OverweightOfAdults% ;Polio;Total expenditure;Diphtheria ; HIV/AIDS;GDP;Population; thinness  1-19 years;Schooling;WaterFacility;WomenInParlament

data = data[['Country', 'continent', 'Life expectancy ', 'Year', 'Status', 'infant deaths', 'Alcohol', 'Hepatitis B', 'MeaslesPrMillion', 'OverweightOfAdults%', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 'Schooling', 'WaterFacility', 'WomenInParlament']]

# save data
data.to_csv('Data/Data_processed.csv', sep=';', index=False)

print(data.isnull().sum().sum())
