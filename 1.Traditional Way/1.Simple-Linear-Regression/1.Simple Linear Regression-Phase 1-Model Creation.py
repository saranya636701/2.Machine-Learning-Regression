import pandas as pd 
#pandas is a library used for working with data (like Excel tables). pd is just a short name (alias) for pandas.

dataset = pd.read_csv('Salary_Data.csv') 
#reads csv file and store it in variable called dataset. csv file is a type of file that stores data in a table format, where each line is a row and values are separated by commas.

independent=dataset[["YearsExperience"]] 
#split input data (YearsExperience) and store it in variable called independent. This is the feature we will use to predict the salary.

dependent=dataset[["Salary"]] 
#split output data (Salary) and store it in variable called dependent. This is the target we want to predict.

print(dataset) 
#prints dataset. both input and output data are shown in the form of a table.

print(independent) 
#prints independent variable (YearsExperience).

print(dependent) 
#prints dependent variable (Salary).

from sklearn.model_selection import train_test_split 
#train_test_split is a function from the sklearn library that helps us split our data into two parts: one for training the model and one for testing it.

X_train,X_test,y_train,y_test=train_test_split(independent, dependent, test_size=0.30,random_state=0) 
#this line splits the independent and dependent variables into training and testing sets. test_size=0.30 means 30% of the data will be used for testing, and random_state=0 ensures that python uses same random pattern everytime to split the data. So,the split and the results become reproducible.

print(y_test)
#prints the test set of the dependent variable (Salary) that we will use to evaluate our model's performance after training.

from sklearn.linear_model import LinearRegression
#Imports Linear Regression (a simple ML algorithm).LinearRegression is a class from the sklearn library that allows us to create a linear regression model, which is a type of machine learning model used to predict a continuous target variable (like Salary) based on one or more input features (like YearsExperience).

regressor=LinearRegression()
#creates a regression model - empty model with default settings and stores it in variable regressor. This object will be used to fit the model to our training data and make predictions.

regressor.fit(X_train,y_train)
#model starts learning from the training data. It finds the best line that fits the relationship between YearsExperience and Salary by minimizing the difference between the predicted and actual values in the training set.

weight = regressor.coef_
#once the model is trained, we can access the weight (coefficient) of the independent variable (YearsExperience) using regressor.coef_. This weight indicates how much the Salary is expected to change for each additional year of experience.

bias = regressor.intercept_ 
#the bias (intercept) is the value of Salary when YearsExperience is zero. It represents the starting point of the regression line on the y-axis.

print("Weight: ", weight)
#prints the weight (coefficient) of the independent variable (YearsExperience). This value indicates how much the Salary is expected to change for each additional year of experience. A positive weight means that as YearsExperience increases, Salary also increases, while a negative weight would indicate that Salary decreases as YearsExperience increases.

print("Bias: ", bias)
#prints the bias (intercept) of the model. This is the value of Salary when YearsExperience is zero. It represents the starting point of the regression line on the y-axis. In other words, it indicates the expected Salary for someone with no years of experience.

y_pred=regressor.predict(X_test)
#after training the model, we can use it to make predictions on the test set (X_test). The predict method takes the input features from the test set and returns the predicted Salary values based on the learned relationship between YearsExperience and Salary. These predicted values are stored in the variable y_pred.

print("Predicted values: ", y_pred)
#print predicted salary values for the test set. These are the Salary predictions made by the model based on the YearsExperience values in the test set (X_test). We can compare these predicted values (y_pred) with the actual Salary values in y_test to evaluate how well our model is performing.

from sklearn.metrics import r2_score
#R2 score is a metric that tells us how well our model's predictions match the actual values. It ranges from 0 to 1, where 1 means perfect predictions and 0 means the model does not explain any of the variability in the target variable (Salary). A higher R2 score indicates a better fit of the model to the data.

print("R2 Score: ", r2_score(y_test, y_pred))
#compares the actual Salary values in y_test with the predicted Salary values in y_pred to calculate the R2 score. This score helps us understand how well our linear regression model is performing in terms of explaining the variability in Salary based on YearsExperience. A higher R2 score indicates that the model is doing a good job of predicting Salary based on YearsExperience, while a lower R2 score suggests that the model may not be capturing the relationship effectively.

import pickle
#pickle is a Python library used for saving and loading machine learning models. It allows us to serialize (save) our trained model to a file and later deserialize (load) it back into memory for making predictions without needing to retrain the model.

filename = 'finalized model.sav'
#saves model into a file named 'finalized model.sav'. This file will contain the serialized version of our trained linear regression model, allowing us to reuse it later without having to retrain it from scratch.

pickle.dump(regressor, open(filename, 'wb'))
#this line saves the trained model (regressor) to a file using pickle. The open function is used to create a new file (or overwrite an existing one) in write-binary mode ('wb'), and pickle.dump writes the model object to that file. This allows us to preserve the state of our trained model so that we can load it later for making predictions without needing to retrain it.

loaded_model = pickle.load(open(filename, 'rb'))
#load saved model from the file. The open function is used to read the file in binary mode ('rb'), and pickle.load reads the model object from the file and stores it in the variable loaded_model. This allows us to use the trained model for making predictions without needing to retrain it.

result = loaded_model.predict([[5.5]])
#uses the loaded model to make a prediction for a new input value of 5.5 years of experience. The predict method takes a 2D array as input, so we pass [[5.5]] to represent one sample with one feature (YearsExperience). The predicted Salary for this input is stored in the variable result.

print("Predicted salary for 5.5 years of experience: ", result)
#prints the predicted salary for 5.5 years of experience. This output shows the Salary value that the loaded model predicts based on the input of 5.5 years of experience, demonstrating how we can use a saved model to make predictions on new data without needing to retrain it.
