import pandas as pd
dataset = pd.read_csv("50_Startups.csv")
print(dataset)
dataset = pd.get_dummies(dataset, drop_first = True)
print(dataset)
dataset.columns
independent = dataset[["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"]]
dependent = dataset[["Profit"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size = 0.30, random_state = 0)
#adding the code for standardization to see if it improves the r2_score
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() #creating an instance of the StandardScaler class, which will be used to standardize the features in the dataset. Standardization is a common preprocessing step in machine learning that helps to improve the performance of certain algorithms by scaling the features to have a mean of 0 and a standard deviation of 1.
X_train = sc.fit_transform(X_train) #calculating the mean and standard deviation of the training data and then transforming it
X_test = sc.transform(X_test) #using the mean and standard deviation calculated from the training data to transform the test data
print("X_train: ",X_train)
from sklearn.svm import SVR #importing the SVR class from the sklearn.svm module, which is used for Support Vector Regression, a type of regression analysis that uses support vector machines to predict continuous outcomes.
regressor = SVR(kernel = 'rbf', C=10) #creating an instance of the SVR class with the kernel parameter set to 'rbf', which stands for Radial Basis Function. This kernel is commonly used in support vector machines and can capture non-linear relationships between the features and the target variable.
regressor.fit(X_train, y_train)
regressor.intercept_
regressor.n_support_
regressor.support_
regressor.support_vectors_
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test, y_pred)
print("r2_score: ",r_score)


