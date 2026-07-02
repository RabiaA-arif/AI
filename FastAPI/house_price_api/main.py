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
    
# home
@app.get("/")
def home():
    return{
        "message":"California House prediction api",
        "status":"running",
        "endpoint":"send POST request to /predict"
        
    }
    
@app.get("/health")
def health():
    return {
        "status":"running",
        "model":"RandomForestRegressor",
        "features":features,
        "avg_error":"$39,000"
            
    }
    
# prediction
@app.post("/predict")
    
def predict(house:HouseFeatures):
    try:
        input_data =pd.DataFrame([{
                "MedInc":house.MedInc,
                "HouseAge":house.HouseAge,
                "AveRooms":house.AveRooms,
                "AveBedrms":house.AveBedrooms,
                "Population":house.Population,
                "latitude":house.Latitude,
                "Longitude":house.Longitude
        }])
        predicted = model.predict(input)[0]
        price_usd = predicted * 100000
    
        return{
            "predicted_price":f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted:,.2f} hundred thousands",
            "fidence_range":f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }
    except Exception as e:
     raise HTTPException (
        status_code=500,
        detail=f"prediction failed :{str(e)}"
    )