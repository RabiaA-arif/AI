import io
import joblib
import pandas as pd
from fastapi import HTTPException,FastAPI
from pydantic import BaseModel


app = FastAPI()
model = joblib.load(filename="breast_cancer.joblib")
feature = joblib.load(filename="breast_cancer_feature.joblib")