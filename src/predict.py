import pickle
import pandas as pd

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/ohe.pkl", "rb") as f:
    one_hot_enc = pickle.load(f)

# Load data
df = pd.read_csv("data/bank_predict.csv", sep=";")
df = df.drop(labels = ["default", "contact", "day", "month", "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)

# Apply the loaded OneHotEncoder to preprocess the new data
data_transformed = pd.DataFrame(one_hot_enc.transform(df), columns=one_hot_enc.get_feature_names_out())

# Predictions
predictions = model.predict(data_transformed)

# Add predictions to the dataframe
df['y_pred'] = ['yes' if pred == 1 else 'no' for pred in predictions]

print(df.head())

df.to_csv("data/bank_predict.csv", index=False)
