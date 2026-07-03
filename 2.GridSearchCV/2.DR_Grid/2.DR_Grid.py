import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('insurance_pre.csv')
print(dataset)

dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset)

independent = dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dependent = dataset['charges']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
param_grid = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter': ['best', 'random'], 'max_features': [None, 'sqrt', 'log2'], 'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid = GridSearchCV(DecisionTreeRegressor(), param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(x_train, y_train)

result = grid.cv_results_
table = pd.DataFrame.from_dict(result)
print(table)

grid_predictions = grid.predict(x_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test, grid_predictions)
print("R_score value for best parameters{}:".format(grid.best_params_), r_score)

age_input = float(input("Enter age: "))
bmi_input = float(input("Enter bmi: "))
children_input = int(input("Enter number of children: "))
sex_input = int(input("Enter sex (0 for female, 1 for male): "))
smoker_input = int(input("Enter smoker status (0 for non-smoker, 1 for smoker): "))

Future_Prediction = grid.predict([[age_input, bmi_input, children_input, sex_input, smoker_input]])
print("Future Prediction= {}".format(Future_Prediction))

import pickle
file_name = 'finalized_model_DR_Grid.sav'
pickle.dump(grid, open(file_name, 'wb'))


