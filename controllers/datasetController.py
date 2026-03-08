from fastapi import APIRouter
import pandas as pd
from services.datasetService import dataService

router = APIRouter(prefix="/telco/dataset", tags=["Dataset"])


@router.put("/reducir/{numeroReducir}")
async def reducirDataset(numeroReducir: int):
    dataService.modificarDataset(numeroReducir)


@router.put("/reiniciar")
async def reiniciarDataset():
    dataService.reiniciarDataset()


@router.get("/metricas")
async def cargarMetricas():
    valoresUnicosChurn, valoresUnicosGenero, PastelContratos, datosNulos, boxplots = (
        dataService.cargarGraficas()
    )
    return {
        "valoresUnicosChurn": valoresUnicosChurn,
        "valoresUnicosGenero": valoresUnicosGenero,
        "pastelContratos": PastelContratos,
        "datosNulos": datosNulos,
        "boxplots": boxplots,
    }


@router.get("/obtener/{limite}")
async def obtenerDataset(limite: int):
    datos = dataService.listarDataset(limite)

    return {
        "longitud": len(dataService.dataset),
        "columnas": list(dataService.dataset.columns),
        "datos": datos,
    }
