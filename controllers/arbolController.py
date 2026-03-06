from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel
from services.decisionTreeService import ArbolService

router = APIRouter(
    prefix="/telco/arbol", tags=['Arbol Decisión']
)

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



class HiperparametrosArbol(BaseModel):
    criterion: str = "entropy"
    classWeight: str = "balanced"
    maxDepth: int = 3


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


arbolService = ArbolService()


@router.post("/entrenar")
async def entrenar(hiperparametros: HiperparametrosArbol):
    reporte, arbol, matriz = arbolService.entrenar(hiperparametros)
    return {"reporte": reporte, "arbol_base64": arbol, "matriz_base64": matriz}

@router.post("/predecir")
async def predecir(caracteristicas: TelcoCaracteristicas):
    datos = caracteristicas.model_dump()
    
    diccionario = {}
    for key, value in datos.items():
        columna_modelo = FIELD_NAME_MAP.get(key, key)
        diccionario[columna_modelo] = value
    df = pd.DataFrame([diccionario])[FEATURE_COLUMNS]
    
    prediccion = arbolService.predecir(df)
    
    resultado = prediccion[0] 
    
    return {"prediccion": str(resultado)}
