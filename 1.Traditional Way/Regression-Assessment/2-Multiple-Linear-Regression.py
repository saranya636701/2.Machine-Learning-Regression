import pandas as pd
dataset = pd.read_csv("insurance_pre.csv")
print(dataset)
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset)
dataset.columns
independent = dataset[["age", "sex_male", "bmi", "children", "smoker_yes"]]
dependent = dataset[["charges"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
weight = regressor.coef_
print("Weight: ", weight)
bias = regressor.intercept_
print("Bias: ",bias)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test, y_pred)
print("r2_score: ",r_score)
import pickle
filename = 'finalized_model_mul_linear_regression.sav'
pickle.dump(regressor, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([[19, 1, 27.9, 0, 0]])
print(result)
