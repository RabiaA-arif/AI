from fastapi import FASTAPI

app = FASTAPI()

@app.get("/")
def greet():
    return {"message":"welcome to our fastapi project"}