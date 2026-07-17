from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data,columns = data.feature_names)
print(df)

print(df["mean radius"])

print(f"Shape of data:{df.shape}")

print("Property of data set")

print(f"Describe:{df.describe()}")

print(f"Features in data:{data.feature_names}")