# =============================================================================
# My Housing Price Prediction -- 16/8/2020
# 
# Ehsan Mirhashemi
#
# =============================================================================

# Import libraries
import numpy as np
import pandas as pd

# import dataset
columns = ['CRIM', 
           'ZN', 
           'INDUS', 
           'CHAS', 
           'NOX', 
           'RM', 
           'AGE', 
           'DIS', 
           'RAD', 
           'TAX', 
           'PTRATIO', 
           'B', 
           'LSTAT', 
           'MEDV']

dataset = pd.read_csv('housing_dataset.csv', names=columns, delimiter=r"\s+")

# Some view of dataset
print(dataset.head(6))
print(np.shape(dataset))
print(dataset.describe())

# Visualizing the dataset
import matplotlib.pyplot as plt
# Box Plot
import seaborn as sns 

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()

for k, v in dataset.items():
    sns.boxplot(y = k, data = dataset, ax = axs[index])
    index += 1

plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 0.5)

# Remove MEDV Outliers
dataset = dataset[~(dataset["MEDV"] >= 50.0)]
print(np.shape(dataset))

# Scatter plot
from sklearn.preprocessing import MinMaxScaler
# Let's scale the columns before plotting them against MEDV
min_max_scaler = MinMaxScaler()

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = dataset.loc[:,column_sels]
y = dataset['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)

fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Remove the skewness of the data trough log transformation
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])

# Test for branch bug