from fastapi import FastAPI
from pydantic import BaseModel

application = FastAPI()

class StudentInfo(BaseModel):
    name: str
    department: str
    year: int
    favourite_subject:str
    

@application.post('/')
def student_information(info:StudentInfo):
    info.name
    info.department
    info.year
    info.favourite_subject
    
    return{
        # "_____Student Dashbored_____",
        "Name":info.name,
        "Department":info.department,
        "Year":info.year,
        "Favourite Subject":info.favourite_subject
        
    }
@application.post('t')
def semester_teacher():
    teacher_name:list
    return{
        "Teachers":teacher_name
    }