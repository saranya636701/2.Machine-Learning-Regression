import pandas as pd
dataset = pd.read_csv("insurance_pre.csv")
print(dataset)
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset)
dataset.columns
independent = dataset[["age", "sex_male", "bmi", "children", "smoker_yes"]]
dependent = dataset[["charges"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'friedman_mse', splitter = 'best', max_features = 'log2', random_state = 42)
regessor = regressor.fit(x_train, y_train)
import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(regressor)
plt.show()
y_pred = regressor.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
