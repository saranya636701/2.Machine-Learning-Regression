import pickle
loaded_model = pickle.load(open('finalized_model_DR_Grid.sav', 'rb'))
result = loaded_model.predict([[34, 28.5, 2, 1, 0]])
print("Result of loaded model:", result)