from scikitlearn.datasets import fetch_california_housing
import pandas  as pd


data = fetch_california_housing()
df = pd.DataFrame(data.data,columns=data.feature.names)

df["Price"]  = data.target
print(f"Shape:{data.shape}")
print(df.head())
print(df.describe())