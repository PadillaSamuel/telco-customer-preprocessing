from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import base64
import io
from pathlib import Path
from pipeline.preprocessing import randomForest, separarCaracteristicas, leerDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn import tree

class RandomForestService:
    def __init__(self, modelo=""):
        self.RUTA_MODELO = Path(modelo)
        self.pipeline = None

    def _cargarRandomForest(self):
        self.pipeline = joblib.load(self.RUTA_MODELO)

    def entrenar(self, service, hiperparametros):
        df = leerDataset(service=service)
        caracteristicas, etiquetas = separarCaracteristicas(df, "Churn")

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            caracteristicas, etiquetas, test_size=0.2
        )
        pipeline = randomForest(
            hiperparametros.nEstimator,
            hiperparametros.criterion,
            hiperparametros.maxDepth,
            hiperparametros.classWeight,
            hiperparametros.maxFeatures,
            hiperparametros.bootstrap,
            hiperparametros.maxSamples,
            hiperparametros.oobScore,
        )
        pipeline.fit(Xtrain, Ytrain)

        Ypredict = self.predecir(Xtest)
        reporte = classification_report(Ytest, Ypredict, output_dict=True)
        outOfBag = getattr(pipeline.named_steps["randomForest"], "oob_score_", None)
        arboles = self._graficarArboles(pipeline, caracteristicas)

        matrizPNG = self._graficarMatriz(Ytest, Ypredict, pipeline.classes_)
        importanciaCaracteristicas = self._graficarImportanciaCaracteristicas(
            pipeline, caracteristicas.columns[:]
        )
        self._guardarModelo(pipeline)
        return reporte, outOfBag, matrizPNG, importanciaCaracteristicas, arboles

    def predecir(self, caracteristicas):
        self._cargarRandomForest()
        return self.pipeline.predict(caracteristicas)

    def _graficarArboles(self, pipeline, caracteristicas):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))

        axes = axes.flatten()

        estimadores = pipeline.named_steps["randomForest"].estimators_

        for i in range(10):
            tree.plot_tree(
                estimadores[i],
                feature_names=caracteristicas.columns[:],
                ax=axes[i],
                filled=True,
                max_depth=2,
                fontsize=8,
            )
        axes[i].set_title(f"Árbol {i+1}")
        plt.tight_layout()
        return self._imagen64()

    def _graficarImportanciaCaracteristicas(self, pipeline, caracteristicasNombres):
        importancias = pipeline.named_steps["randomForest"].feature_importances_
        indices = np.argsort(importancias)

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de las Variables")
        plt.barh(range(len(indices)), importancias[indices], align="center")
        plt.yticks(range(len(indices)), [caracteristicasNombres[i] for i in indices])
        plt.xlabel("Importancia Relativa")
        return self._imagen64()

    def _graficarMatriz(self, Ytest, Ypredict, classes):
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            Ytest, Ypredict, display_labels=classes, cmap=plt.cm.Blues, ax=ax
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
