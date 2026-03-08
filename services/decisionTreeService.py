import io
import matplotlib.pyplot as plt
import base64
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from pipeline.preprocessing import leerDataset, separarCaracteristicas, arbolDesicion
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)

class ArbolService:
    def __init__(self, modelo="models/decisionTree.joblib"):
        self.RUTA_MODELO = Path(modelo)
        self.pipeline = None

    def _cargarArbol(self):
        self.pipeline = joblib.load(self.RUTA_MODELO)

    def entrenar(self, service, hiperparametros):

        df = leerDataset(service)

        caracteristicas, etiquetas = separarCaracteristicas(df, "Churn")

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            caracteristicas, etiquetas, test_size=0.2
        )

        pipeline = arbolDesicion(
            hiperparametros.criterion,
            hiperparametros.classWeight,
            hiperparametros.maxDepth,
        )
        pipeline.fit(Xtrain, Ytrain)

        Ypredict = self.predecir(Xtest)

        reporte = classification_report(Ytest, Ypredict, output_dict=True)
        arbolPNG = self._graficarArbol(pipeline, hiperparametros.maxDepth)
        matrizPNG = self._graficarMatriz(Ytest, Ypredict, pipeline.classes_)
        self._guardarModelo(pipeline)
        return reporte, arbolPNG, matrizPNG

    def predecir(self, caracteristicas):
        self._cargarArbol()
        return self.pipeline.predict(caracteristicas)

    def _graficarMatriz(self, Ytest, Ypredict, classes):
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            Ytest, Ypredict, display_labels=classes, cmap=plt.cm.Blues, ax=ax
        )

        return self._imagen64()

    def _graficarArbol(self, pipeline, maxDepth):

        plt.figure(figsize=(20, 12))
        caracteristicasArbol = pipeline.named_steps[
            "preprocesador"
        ].get_feature_names_out()

        plot_tree(
            pipeline.named_steps["arbolDesicion"],
            feature_names=caracteristicasArbol,
            filled=True,
            rounded=True,
            fontsize=12,
            max_depth=maxDepth,
        )
        return self._imagen64()

    def _imagen64(self):
        buffer = io.BytesIO()

        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img}"

    def _guardarModelo(self, pipeline):
        self.RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.RUTA_MODELO)
