import pickle
loaded_model = pickle.load(open('finalized model.sav', 'rb'))
result = loaded_model.predict([[13]])
print("Predicted salary for 13 years of experience: ", result)