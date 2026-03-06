from fastapi import FastAPI
from controllers import arbolController

app = FastAPI(title="Telco Customers Churn")
app.include_router(arbolController.router)

@app.get("/")
async def inicio():
    return {"mensaje": "corriendo"}