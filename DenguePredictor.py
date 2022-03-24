# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
xt = pd.read_csv('dengue_features_train.csv')
yt = pd.read_csv('dengue_labels_train.csv')
wt = pd.read_csv('dengue_features_test.csv')

X = xt.iloc[:,4:].values
y = yt.iloc[:, 3].values
w = wt.iloc[:,4:].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

imputer = imputer.fit(w)
w = imputer.transform(w)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(w)

filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))