import pandas as pd
dataset = pd.read_csv("50_Startups.csv")
print(dataset)
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset)
dataset.columns
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
dependent = dataset['Profit']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)
print(r2_score)
import pickle
filename = 'finalized model_random_forest_regressor_model.sav'
pickle.dump(regressor, open(filename,'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([[165349.20, 136897.80, 471784.10, 0, 0]])
print(result)


