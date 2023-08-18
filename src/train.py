# Import
import pandas as pd
import seaborn as sns
import pickle
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/bank_proccess.csv")

# Model
X = df.drop("deposit", axis=1)
y = df["deposit"]

cat_cols = ["job", "marital", "education", "housing"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", drop="first"),
    cat_cols),
    remainder="passthrough")

X_train = one_hot_enc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=one_hot_enc.get_feature_names_out())

X_train.head(2)

X_test = pd.DataFrame(one_hot_enc.transform(X_test), columns=one_hot_enc.get_feature_names_out())
X_test.head(2)

model = LGBMClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="g")

print(f"Specificity: {specificity}")
print("Confusion Matrix:")

print(classification_report(y_test, y_pred))

# Save model
# Specify the file path where you want to save the pickle file
file_path = "models/model.pkl"

# Save the model as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(model, f)

file_path = "models/ohe.pkl"

# Save the OneHotEncoder as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(one_hot_enc, f)