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


# Import cross validation libraries
from sklearn.model_selection import cross_val_score, KFold

# Regressors
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores_map = {}
kf = KFold(n_splits=10)


# Linear Model
linear_model = LinearRegression()
scores = cross_val_score(linear_model, 
                          x_scaled,
                          y,
                          cv = kf, 
                          scoring = 'neg_mean_squared_error')
 
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
scores_map['LinearRegression'] = scores
 
# Ridge Model
ridge_model = Ridge()
scores = cross_val_score(ridge_model, 
                          x_scaled, 
                          y, 
                          cv = kf, 
                          scoring= 'neg_mean_squared_error')
 
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
scores_map['Ridge'] = scores

# Polynomial Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

poly_ridge_model = make_pipeline(PolynomialFeatures(degree=3), ridge_model)

scores = cross_val_score(poly_ridge_model,
                         x_scaled,
                         y,
                         cv = kf,
                         scoring= 'neg_mean_squared_error')

print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
scores_map['PolyRidge'] = scores

# SVM Model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

svr_model = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

grid_svm = GridSearchCV(svr_model,
                        cv = kf,
                        param_grid={
                            "C": [1e0, 1e2, 1e3],
                            "gamma": np.logspace(-2, 2, 5)},
                        scoring='neg_mean_squared_error')
grid_svm.fit(x_scaled, y)
print("Best Classifier:", grid_svm.best_estimator_)



scores = cross_val_score(svr_model,
                         x_scaled,
                         y,
                         cv = kf,
                         scoring='neg_mean_squared_error')

scores_map['SVR'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))



# Decison Tree model
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(max_depth=4)

grid_dt = GridSearchCV(dt_model,
                       cv = kf,
                       param_grid={
                           "max_depth": [1,2,3,4,5,6,7,8,9,10]},
                       scoring='neg_mean_squared_error')
grid_dt.fit(x_scaled, y)

print("Best Decision Tree Model:", grid_dt.best_estimator_)



scores = cross_val_score(dt_model,
                         x_scaled,
                         y,
                         cv = kf,
                         scoring= 'neg_mean_squared_error')

scores_map['DecisionTree'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# K-Neighbor Regressor
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)
scores = cross_val_score(knn, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['KNeighborsRegressor'] = scores
grid_knr = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
grid_knr.fit(x_scaled, y)
print("Best K-Neighbor Regressor :", grid_knr.best_estimator_)
print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
