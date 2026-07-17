from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

print("Strat the Model loading")
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)
print("Data in X")
# print(X)
y = data.target
print("Data in y")
# print(y)


print(f"Total Record in X: {X.shape[0]}") # row is dataset 
print(f"Total Record in y:{y.shape[0]}") 

print(f"Total Column in X:{X.shape[1]}")



# spliting the data
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    shuffle=True
    )
model = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=40,
    criterion='absolute_error'
)

model.fit(X_train,y_train)

