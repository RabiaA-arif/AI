from fastapi import FastAPI
from pydantic import BaseModel
# Health Insurance system


app = FastAPI()
class HealthInssurance(BaseModel):
    name: str
    age: int
    income: float
    health_condition: str
    complication: str
    
@app.post('/')

def insurance_approval(application:HealthInssurance):
    approved=(
    application.age < 60 and application.income > 40000 and 
    application.complication == "yes"
    )
    return {
        "Person Name:" : application.name,
        "Person Health Condition" :application.health_condition,
        "complication":application.complication,
        "Decision":"Aproved" if approved else "Rejected"
        
    }