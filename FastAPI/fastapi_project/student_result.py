from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel


app=FastAPI()

class MarksSubmission(BaseModel):
    student_id: str
    marks:  int
    subject: str
students = {
    "01":{"name":"A","marks":87,"grade":"A"},
    "02":{"name":"B","marks":88,"grade":"A"},
    "03":{"name":"C","marks":70,"grade":"B"},
    "04":{"name":"D","marks":80,"grade":"B"}

}

@app.get("/student/{student_id}")

def get_student_id(student_id:str):
    if student_id not in students:
        raise HTTPException(
        status_code = 404,
        detail=f"Student With id {student_id} is not found"
        )
    # if student_id <= 0:
    #     raise HTTPException(
    #         status_code = 404,
    #         detail=f"This number is not valid id"
    #     )
    return students[student_id]



# raise HTTPException(
#     status_code = 404,
#     detail = "Not Found"
# )

@app.post("/marks-submit")
def submit_marks(submission:MarksSubmission):
    if submission.marks < 0 or submission.marks > 100:
        raise HTTPException(
            status_code= 400,
            detail={
                "error":"marks is between 0 and 100",
                "marks received": submission.marks,
                "fix":"entera marks between 0 and 100"
            }
            
        )
        
        # subject name empty
        
    if submission.subject.strip() == "":
        raise HTTPException(
            status_code=400,
            detail={
                "subject_name":"input the valid subject name",
                
            }
        )
        
    students[submission.student_id]["marks"] = submission.marks
    return{
        "message":"student marks is submitted succesfully",
        "student":students[submission.student_id]["name"],
        "subject":submission.subject,
        "marks":submission.marks
    }