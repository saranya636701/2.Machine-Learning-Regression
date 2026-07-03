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
from sklearn.svm import SVR
param_grid = {'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'C': [1, 10, 100, 1000, 2000, 3000], 'gamma': ['auto', 'scale']}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(x_train, y_train)
result = grid.cv_results_ #grid.cv_results_ is a dictionary that contains the results of the grid search, including the mean test scores and the parameters for each combination of hyperparameters.
print("R_score value for best parameters{}:".format(grid.best_params_))
table = pd.DataFrame.from_dict(result) #table is a DataFrame that contains the results of the grid search, which can be used to analyze the performance of different hyperparameter combinations.
print(table)
age_input = float(input("Enter age: "))
bmi_input = float(input("Enter bmi: "))
children_input = int(input("Enter number of children: "))
sex_input = int(input("Enter sex (0 for female, 1 for male): "))
smoker_input = int(input("Enter smoker status (0 for non-smoker, 1 for smoker): "))
Future_Prediction = grid.predict([[age_input, bmi_input, children_input, sex_input, smoker_input]])
print("Future Prediction= {}".format(Future_Prediction))
print("grid.best_estimator_", grid.best_estimator_) #grid.best_estimator_ is the best model found by the grid search, which is the model with the highest mean test score.
print("grid.best_params_", grid.best_params_) #grid.best_params_ is the combination of hyperparameters that resulted in the best model.
print("grid.best_score_", grid.best_score_) #grid.best_score_ is the mean test score of the best model.
y_pred = grid.predict(x_test)
print("Predicted values:", y_pred)
from sklearn.metrics import r2_score 
print("R2 Score:", r2_score(y_test, y_pred))  
import pickle
file_name = 'finalized_model_SVR_Grid.sav'
pickle.dump(grid, open(file_name, 'wb'))
loaded_model = pickle.load(open(file_name, 'rb'))
result = loaded_model.predict([[34, 28.5, 2, 1, 0]])
print("Result of loaded model:", result)

