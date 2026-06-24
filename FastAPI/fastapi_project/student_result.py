from fastapi import FastAPI

app=FastAPI()

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
    return students[student_id]



# raise HTTPException(
#     status_code = 404,
#     detail = "Not Found"
# )