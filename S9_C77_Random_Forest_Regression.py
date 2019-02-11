# Random Forest Regression

#Import the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting the Training set and Test Set

'''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)'''

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  sc_X.fit_transform(X_train)
X_test =  sc_X.transform(X_test)
sc_y = StandartScaler()
y_train = sc_y.fit_transform(y_train)'''

#Fitting the Randon Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)


#Predicting a new result Geral
y_pred = regressor.predict(np.array(6.5).reshape(1, -1))

#Visualising the Randon Forest Regression results (For hugh resolution and smoother curve)
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Randon Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

