# =============================================================================
# My Housing Price Prediction -- 16/8/2020
# 
# Ehsan Mirhashemi
#
# =============================================================================

import pandas as pd

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
