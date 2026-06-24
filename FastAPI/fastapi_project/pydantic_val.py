from fastapi import FastAPI
from pydantic import BaseModel


##########error #############
# ImportError: cannot import name 'BaseModel' from partially initialized module 'pydantic'
# (most likely due to a circular import)
# reason : The error occurs because you have named your local file pydantic.py
    
apps=FastAPI()

class LoanApplication(BaseModel):
    name: str
    age: int
    income: float
    loan_amount: float
    employeement_years: int
    
@apps.post('/predict')

def predict_loan(application:LoanApplication):
    # model login
    approved = (
        application.income > 50000 and 
        application.employeement_years > 2 and
        application.age > 21
    )
    return{
        "applicant name" : application.name,
        "loan_amount" : application.loan_amount,
        "decision" : "approve" if approved else "rejected",
        "reviewe income": application.income
    }
    
    # using pydantic we validation data type with business logic
    
    # Assignment :Health insuurance prediction application
    # using pydantic model