from fastapi import FastAPI
app = FastAPI()

all_customer = [
    {"name":"sam","city":"umerkot","risk":"low"},
    {"name":"saman","city":"umerkot","risk":"high"},
    {"name":"sana","city":"hydrabad","risk":"medium"},
    {"name":"sara","city":"karachi","risk":"low"},
    {"name":"samam","city":"lahore","risk":"low"},
    {"name":"samra","city":"umerkot","risk":"high"}
]

@app.get("/customers")

def get_customer(city:str,risk:str):
    filter=[
        cust for cust in all_customer
        if cust["city"] == city and cust["risk"] == risk
    ]
    return {
        "city":city,
        "risk":risk,
        "count":len(filter),
        "results":filter
    }