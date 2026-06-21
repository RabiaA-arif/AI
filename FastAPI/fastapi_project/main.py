from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def greet():
    return {"message":"welcome to our fastapi project"}


@app.get("/intro")
def intro():
    return {"Introduction":"Hi i am Rabia Arif AI Engineer"}


@app.get("/customer")

def customer(customer_id: int):
    return {
        "customer id":customer_id,
        "customer_name":"Rabia",
        "customer_status":"Active"
    }