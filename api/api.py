from fastapi import FastAPI
from controllers import arbolController, datasetController, randomForestController

app = FastAPI(title="Telco Customers Churn")
app.include_router(arbolController.router)
app.include_router(randomForestController.router)
app.include_router(datasetController.router)

@app.get("/")
async def inicio():
    return {"mensaje": "corriendo"}
# uvicorn api.pai:app --host 127.0.0.1 --port 5000