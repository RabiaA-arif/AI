from fastapi import FastAPI


my_application = FastAPI()

@my_application.get("/")

def home():
    return ("Welcome To our simple calculator home page")


@my_application.get("/add")

def add():
    return {"Addition:": 7 + 8 }


@my_application.get("/sub")

def subtraction():
    return {"Subtraction:" : 56 - 8 }

