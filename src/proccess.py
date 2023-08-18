# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Import data
df = pd.read_csv("data/bank.csv")
df.sample(3)

# Convert the column to category and map the values
dep_mapping = {"yes": 1, "no": 0}

# Convert the column to category and map the values
df["deposit"] = df["deposit"].astype("category").map(dep_mapping)

df = df.drop(labels = ["default", "contact", "day", "month", "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)

df.sample(3)

# Chack data
print("Shape")
print(df.shape)

print("\nTypes")
print(df.dtypes)

# Check missings
pd.DataFrame(df.isnull().sum()).T

msno.matrix(df)
plt.show()

cat_cols = ["job", "marital", "education", "housing"]
num_cols = ["age", "balance", "duration", "campaign"]

df.describe()

# Graphs
sns.set(style="whitegrid")
g = sns.countplot(data=df, x="job", hue="deposit")
g.set_xticklabels(g.get_xticklabels(), rotation=90);

sns.set(style="whitegrid")
g = sns.countplot(data=df, x="education", hue="deposit")
g.set_xticklabels(g.get_xticklabels(), rotation=90);

sns.set(style="whitegrid")
g = sns.countplot(data=df, x="housing", hue="deposit")
g.set_xticklabels(g.get_xticklabels(), rotation=90);

sns.set(style="whitegrid")
g = sns.countplot(data=df, x="marital", hue="deposit")
g.set_xticklabels(g.get_xticklabels(), rotation=90);

sns.pairplot(df)

def correlation_matrix(df, cols):
     fig = plt.gcf()
     fig.set_size_inches(8, 6)
     plt.xticks(fontsize=10)
     plt.yticks(fontsize=10)
     fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={"size": 12}, linecolor="w", cmap="RdBu")
     plt.show(block=True)

correlation_matrix(df, num_cols)

df.to_csv("data/bank_proccess.csv", index=False)