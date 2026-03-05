import io
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from pipeline.preprocessing import leerDataset, separarCaracteristicas, arbolDesicion
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

def servicioEntrenarGraficar(hiperparametros):

    df = leerDataset()

    caracteristicas, etiquetas = separarCaracteristicas(df, "Churn")
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        caracteristicas, etiquetas, test_size=0.2
    )
    pipeline = arbolDesicion(
        hiperparametros.criterion, hiperparametros.classWeight, hiperparametros.maxDepth
    )
    pipeline.fit(Xtrain, Ytrain)

    Ypredict = pipeline.predict(Xtest)

    reporte = classification_report(Ytest, Ypredict, output_dict=True)

    bufferArbol = io.BytesIO()
    plt.figure(figsize=(20, 12))
    caracteristicasArbol = pipeline.named_steps[
        "preprocesador"
    ].get_features_names_out()

    plot_tree(
        pipeline.named_steps["arbolDesicion"],
        feature_names=caracteristicasArbol,
        filled=True,
        rounded=True,
        fontsize=12,
        max_depth=hiperparametros.maxDepth,
    )

    plt.savefig(bufferArbol, format="png", bbox_inches="tight")
    plt.close()
    arbolPNG = base64.b64encode(bufferArbol.getvalue()).decode("utf-8")

    matriz = confusion_matrix(Ytest, Ypredict)
    bufferMatriz = io.BytesIO()

    plt.figure(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=pipeline.classes_
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(bufferMatriz, format="png", bbox_inches="tight")
    plt.close()

    matrizPng = base64.b64encode(bufferMatriz.getvalue()).decode("utf-8")
    
    return reporte, arbolPNG, matrizPng
