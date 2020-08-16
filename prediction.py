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
