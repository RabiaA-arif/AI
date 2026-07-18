import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "breast_cancer.joblib")
feature = joblib.load(BASE_DIR / "breast_cancer_feature.joblib")


class BreastCancer(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    mean_radius: float = Field(alias="mean radius", description="Mean Radius")
    mean_texture: float = Field(alias="mean texture", description="Mean Texture")
    mean_perimeter: float = Field(alias="mean perimeter", description="Mean Perimeter")
    mean_area: float = Field(alias="mean area", description="Mean Area")
    mean_smoothness: float = Field(alias="mean smoothness", description="Mean Smoothness")
    mean_compactness: float = Field(alias="mean compactness", description="Mean Compactness")
    mean_concavity: float = Field(alias="mean concavity", description="Mean Concavity")
    mean_concave_points: float = Field(alias="mean concave points", description="Mean Concave Points")
    mean_symmetry: float = Field(alias="mean symmetry", description="Mean Symmetry")
    mean_fractal_dimension: float = Field(alias="mean fractal dimension", description="Mean Fractal Dimension")
    radius_error: float = Field(alias="radius error", description="Radius Error")
    texture_error: float = Field(alias="texture error", description="Texture Error")
    perimeter_error: float = Field(alias="perimeter error", description="Perimeter Error")
    area_error: float = Field(alias="area error", description="Area Error")
    smoothness_error: float = Field(alias="smoothness error", description="Smoothness Error")
    compactness_error: float = Field(alias="compactness error", description="Compactness Error")
    concavity_error: float = Field(alias="concavity error", description="Concavity Error")
    concave_points_error: float = Field(alias="concave points error", description="Concave Points Error")
    symmetry_error: float = Field(alias="symmetry error", description="Symmetry Error")
    fractal_dimension_error: float = Field(alias="fractal dimension error", description="Fractal Dimension Error")
    worst_radius: float = Field(alias="worst radius", description="Worst Radius")
    worst_texture: float = Field(alias="worst texture", description="Worst Texture")
    worst_perimeter: float = Field(alias="worst perimeter", description="Worst Perimeter")
    worst_area: float = Field(alias="worst area", description="Worst Area")
    worst_smoothness: float = Field(alias="worst smoothness", description="Worst Smoothness")
    worst_compactness: float = Field(alias="worst compactness", description="Worst Compactness")
    worst_concavity: float = Field(alias="worst concavity", description="Worst Concavity")
    worst_concave_points: float = Field(alias="worst concave points", description="Worst Concave Points")
    worst_symmetry: float = Field(alias="worst symmetry", description="Worst Symmetry")
    worst_fractal_dimension: float = Field(alias="worst fractal dimension", description="Worst Fractal Dimension")


@app.get("/")
def homepage():
    return {
        "message": "Welcome breast Cancer prediction",
        "status": "Running",
        "endpoint": "Send post request for prediction"
    }


@app.get("/feature")
def detail():
    return {
        "status": "Running",
        "model": "ExtraTreeRegressor",
        "feature": feature
    }


@app.post("/predict")
def predict(cancer: BreastCancer):
    try:
        input_data = pd.DataFrame([cancer.model_dump(by_alias=True)])
        predicted = model.predict(input_data)[0]
        return {"prediction": float(predicted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction Failed: {str(e)}")
    