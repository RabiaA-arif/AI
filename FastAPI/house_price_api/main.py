import io 
import joblib
import pandas as pd
from fastapi import HTTPException , FastAPI,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel , Field

app = FastAPI()
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# input schema

class HouseFeatures(BaseModel):
    MedInc : float = Field(gt =0,description="Model Income of"/"Neighbouredhood")
    HouseAge :float = Field(gt=0,description="Average age of house")
    AveRooms : float = Field(gt=0,description="Average number of rooms")
    AveBedrms : float = Field(gt=0,description="Average number of rooms")
    Population:float = Field(gt=0,description="Total population")
    AveOccup: float = Field(gt=0,description="Average occupation")
    Latitude : float = Field(ge=32,le=42,description="Latitude")
    Longitude : float = Field(ge=-125,le=-114,description="Longitude")
    
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
                "AveBedrms":house.AveBedrms,
                "Population":house.Population,
                "AveOccup":house.AveOccup,
                "Latitude":house.Latitude,
                "Longitude":house.Longitude
        }])
        predicted = model.predict(input_data)[0]
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
     
     
@app.post("/predict-file")

async def predict_file(file:UploadFile=File(...)):
    # async ka mutlab ha as function ko wait kar sackta ha 
    # jub tak wait karna ha tub tak python apna dosra kam kar sacti ha
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a csv file only"
        )
        
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    required_columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup",
        "Latitude","Longitude"
    ]
    
    missing_column = [
        col for col in required_columns
        if col not in df.columns
    ]
    
    if missing_column:
        raise HTTPException(
            status_code=400,
            detail=f"These column are missign from your file{missing_column}"
        )
    if len(df) ==0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data"
        )
        
    try:
        predictions = model.predict(df[required_columns])
        
        df["predicted_price_usd"] = pd.Series(predictions * 100000).apply(lambda x : f"${x:,.0f}").values
        
        output = df.to_csv(index=False)
        
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition":"attachment; filename=predictions.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed:{str(e)}"
        )