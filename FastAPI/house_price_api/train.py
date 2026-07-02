from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
import joblib

print("I am Loading Model")
data = fetch_california_housing()
# X : by convention in ml it is input data
X = pd.DataFrame(data.data,columns=data.feature_names)
y = data.target

print(f"Total record : {X.shape[0]}") # return rows of data

# data spiliting
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# model training 

model = RandomForestRegressor (
    n_estimators=100,
    random_state=42
)
model.fit(X_train,y_train)

# prediction accuracy

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred) # require two thing for calculat original price and predicted price
r2 = r2_score(y_test,y_pred) # r2 give us answer between 0 & 1 kind of percentage

print(f"Average Error :${mae * 100000:,.0f}")

joblib.dump(model,"house_model.joblib")
joblib.dump(list(X.columns),"house_features.joblib")