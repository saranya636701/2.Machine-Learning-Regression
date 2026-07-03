import pickle
loaded_model = pickle.load(open("finalized model_random_forest_regressor_model.sav", 'rb'))
result = loaded_model.predict([[165349.20, 136897.80, 471784.10, 0, 0]])
print(result)