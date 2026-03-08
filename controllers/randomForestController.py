from fastapi import APIRouter
from pydantic import BaseModel
from services.datasetService import dataService
from services.randomforestService import RandomForestService
import pandas as pd

router = APIRouter(prefix="/telco/randomforest", tags=["RandomForest"])

randomForestService = RandomForestService()
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

FIELD_NAME_MAP = {
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
}


class Hiperparametros(BaseModel):
    nEstimator: int = 100
    criterion: str = "entropy"
    maxDepth: int = 3
    classWeight: str = "balanced"
    maxFeatures: str = "sqrt"
    bootstrap: bool = True
    maxSamples: float = 2 / 3
    oobScore: bool = True


class TelcoCaracteristicas(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


@router.post("/entrenar")
async def entrenar(hiperparametros: Hiperparametros):
    reporte, outOfBag, matrizPNG, importanciaCaracteristicas, arboles = (
        randomForestService.entrenar(dataService, hiperparametros)
    )
    return {
        "reporte": reporte,
        "outOfBag": outOfBag,
        "matriz_base64": matrizPNG,
        "importancia_caracteristicas_base64": importanciaCaracteristicas,
        "arboles_base64": arboles,
    }


@router.post("/predecir")
async def predecir(caracteristicas: TelcoCaracteristicas):
    datos = caracteristicas.model_dump()
    diccionario = {}
    
    for key, value in datos.items():
        columnaModelo = FIELD_NAME_MAP.get(key, key)
        diccionario[columnaModelo] = value
    
    df = pd.DataFrame([diccionario])[FEATURE_COLUMNS]
    
    prediccion = randomForestService.predecir(df)
    
    return {"prediccion": str(prediccion[0])}
