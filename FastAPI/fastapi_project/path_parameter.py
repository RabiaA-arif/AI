from fastapi import FastAPI

app = FastAPI()

customer_risk_profiles = {
    1:{"name":"ali","risk":"high","score":1},
    2:{"name":"arsi","risk":"low","score":2},
    3:{"name":"jawad","risk":"medium","score":3}
}

@app.get("/customer/{id}")

def customer_risk(id: int):
    if id not in customer_risk_profiles:
        return {"Error":f"Customer id {id} is not found"}
    
    profile = customer_risk_profiles[id]
    return {
        "customer id": id,
        "name":profile["name"],
        "risk":profile["risk"],
        "score":profile["score"]
    }