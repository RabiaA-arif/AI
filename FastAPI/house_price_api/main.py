import joblib
import pandas as pd
from fastapi import HTTPException , FastAPI
from pydantic import BaseModel , Field

app = FastAPI()
model = joblib("house_features.joblib")
features = joblib("house_features.joblib")