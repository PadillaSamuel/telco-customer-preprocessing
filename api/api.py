from fastapi import FastAPI
from controllers import arbolController, datasetController

app = FastAPI(title="Telco Customers Churn")
app.include_router(arbolController.router)
app.include_router(datasetController.router)

@app.get("/")
async def inicio():
    return {"mensaje": "corriendo"}