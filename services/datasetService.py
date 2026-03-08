from pathlib import Path
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import base64
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


class DatasetService:
    def __init__(self, datasetCopia=DATA_PATH):
        self.dataset = pd.read_csv(datasetCopia)

    def modificarDataset(self, numero):
        if numero < len(self.dataset):
            self.dataset= self.dataset.drop(self.dataset.sample(numero).index).reset_index(drop=True)    
        return self.dataset

    def reiniciarDataset(self):
        self.dataset = pd.read_csv(DATA_PATH)
        return self.dataset

    def cargarGraficas(self):
        
        valoresUnicosChurn = self.dataset["Churn"].value_counts()
        valoresUnicosGenero = self.dataset["gender"].value_counts()
        valoresUnicosCotratos = self.dataset["Contract"].value_counts()
        
        PastelValoresUnicosContratos = self._graficaPastel(valoresUnicosCotratos)
        dfTemp = self.dataset.copy()
        
        columnasNumericas = ["tenure", "MonthlyCharges", "TotalCharges"]
        for col in columnasNumericas:
            dfTemp[col] = pd.to_numeric(dfTemp[col], errors="coerce")
        datosNulos = dfTemp["TotalCharges"].isnull().sum()
        
        dfTemp.dropna(subset=columnasNumericas)

        boxplots = self._graficarBoxplots(columnasNumericas, dfTemp)
        
        return (
            valoresUnicosChurn.to_dict(),
            valoresUnicosGenero.to_dict(),
            PastelValoresUnicosContratos,
            int(datosNulos),
            boxplots,
        )

    def leerDataset(self):
        return self.dataset

    def _imagen64(self):
        buffer = io.BytesIO()

        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img}"

    def _graficaPastel(self, valoresUnicos):
        plt.figure(figsize=(10, 10))
        plt.pie(
            valoresUnicos,
            labels=valoresUnicos.index,
            startangle=140,
            colors=["#ff9999", "#66b3ff", "#99ff99"],
            autopct="%1.1f%%",
        )
        return self._imagen64()

    def _graficarBoxplots(self, columnasNumericas, dfTemp):

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Distribución de Variables Numéricas")

        for i, col in enumerate(columnasNumericas):
            sns.boxplot(y=dfTemp[col], ax=axes[i], color="#69d2e7")
            axes[i].set_title(col)
            axes[i].set_ylabel("")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._imagen64()


dataService = DatasetService()
