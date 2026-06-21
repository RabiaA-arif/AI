from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def greet():
    return {"message":"welcome to our fastapi project"}


@app.get("/intro")
def intro():
    return {"Introduction":"Hi i am Rabia Arif AI Engineer"}