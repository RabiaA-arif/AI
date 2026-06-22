from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LoanApplication(BaseModel):
    age: int
    salary: float
    loan_amount: float
    employment_year: int
    

@app.post('/predict')    

def predict_loan(application:LoanApplication):
    if application.salary > 50000 and application.employment_year > 5:
        decision = "Approved"
    else:
        decision = "Rejected"
    
    return {
        "application_age":application.age,
        "decision": decision
    }