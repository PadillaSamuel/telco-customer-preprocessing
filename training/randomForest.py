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
from services.datasetService import dataService

RUTA_MODELO = Path("models/randomForest.joblib")
nEstimator: int = 100
criterion: str = "entropy"
maxDepth: int = 3
classWeight: str = "balanced"
maxFeatures: str = "sqrt"
bootstrap: bool = True
maxSamples: float = 2 / 3
oobScore: bool = True


def graficar_arboles(pipeline):

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    estimadores = pipeline.named_steps["randomForest"].estimators_
    caracteristicas = pipeline[:-1].get_feature_names_out()

    for i in range(10):
        tree.plot_tree(
            estimadores[i],
            feature_names=caracteristicas,
            ax=axes[i],
            filled=True,
            max_depth=2,
            fontsize=8,
        )
        axes[i].set_title(f"Árbol {i+1}")

    plt.tight_layout()
    plt.show()


def graficar_importancia(pipeline):

    importancias = pipeline.named_steps["randomForest"].feature_importances_
    caracteristicas = pipeline[:-1].get_feature_names_out()

    indices = np.argsort(importancias)

    plt.figure(figsize=(10, 6))
    plt.title("Importancia de Variables")

    plt.barh(range(len(indices)), importancias[indices])
    plt.yticks(range(len(indices)), caracteristicas[indices])

    plt.xlabel("Importancia relativa")

    plt.tight_layout()
    plt.show()


def graficar_matriz(Ytest, Ypredict, clases):

    fig, ax = plt.subplots(figsize=(8, 8))

    ConfusionMatrixDisplay.from_predictions(
        Ytest, Ypredict, display_labels=clases, cmap=plt.cm.Blues, ax=ax
    )

    plt.show()


def main():
    df = leerDataset(service=dataService)
    caracteristicas, etiquetas = separarCaracteristicas(df, "Churn")

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        caracteristicas, etiquetas, test_size=0.2
    )
    pipeline = randomForest(
        nEstimator,
        criterion,
        maxDepth,
        classWeight,
        maxFeatures,
        bootstrap,
        maxSamples,
        oobScore,
    )
    pipeline.fit(Xtrain, Ytrain)

    joblib.dump(pipeline, RUTA_MODELO)

    Ypredict = pipeline.predict(Xtest)

    print(classification_report(y_true=Ytest, y_pred=Ypredict))

    oob = getattr(pipeline.named_steps["randomForest"], "oob_score_", None)
    print("OOB Score:", oob)

    graficar_arboles(pipeline)

    graficar_importancia(pipeline)

    graficar_matriz(Ytest, Ypredict, pipeline.named_steps["randomForest"].classes_)

    return


if __name__ == "__main__":
    main()
