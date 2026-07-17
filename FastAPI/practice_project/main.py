import io
import joblib
import pandas as pd
from fastapi import HTTPException,FastAPI
from pydantic import BaseModel , Field


app = FastAPI()
model = joblib.load(filename="breast_cancer.joblib")
feature = joblib.load(filename="breast_cancer_feature.joblib")


# input validation schema


class BreastCancer(BaseModel):
    mean_radius : float = Field(gt=0,description="Mean Radius")
    mean_texture : float = Field(gt=0,description="Mean Tecture")
    mean_symmetry : float = Field(gt=0,description="Mean Symmetry")
    
    
    
    
# home
@app.get("/")
def homepage():
    return{
        "message":"Welcome breast Cancer pridiction",
        "status":"Running",
        "endpoint":"Send post request for prediction"
    }
    
    
@app.get("/feature")

def detail():
    return{
        "status":"Running",
        "model":"ExtraTreeRegressor",
        "feature":feature
    }
    
# prediction 
@app.post("/predict")
def predict(cancer:BreastCancer):
    try:
        input_data = pd.DataFrame([{
            "mean_radius":cancer.mean_radius,
            "mean_texture":cancer.mean_texture,
            "mean_symmetry":cancer.mean_symmetry
        }])
        predicted = model.predict(input_data)[0]
    
        return{
        "prediction":predicted
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction Failed:{str(e)}"
        )
    