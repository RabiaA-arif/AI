import joblib
import pandas as pd
from fastapi import HTTPException , FastAPI
from pydantic import BaseModel , Field

app = FastAPI()
model = joblib("house_features.joblib")
features = joblib("house_features.joblib")


# input schema

class HouseFeatures(BaseModel):
    MedInc : float = Field(gt =0,description="Model Income of"\
        "Neighbouredhood")
    HouseAge :float = (Field(gt=0,description="Average age of house"))
    AveRooms : float = Field(gt=0,description="Average number of rooms")
    AveBedrooms : float = Field(gt=0,description="Average number of rooms")
    Population:float = Field(gt=0,description="Total population")
    AveOccup: float = Field(gt=0,description="Average occupation")
    Latitude : float = Field(ge=32,le=42,description="Latitude")
    Longitude : float = Field(ge=125,le=114,description="Longitude")