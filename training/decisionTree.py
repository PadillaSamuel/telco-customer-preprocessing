import joblib 
from pathlib import Path
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from pipeline.preprocessing import  separarCaracteristicas,leerDataset, arbolDesicion
import matplotlib.pyplot as plt
from services.datasetService import dataService
RUTA_MODELO= Path('models/decisionTree.joblib')
CRITERION='entropy'
CLASS_WEIGHT= 'balanced'
MAX_DEPTH= 3

def main():    
    df  = leerDataset(dataService)
    caracteristicas, objetivo = separarCaracteristicas(df, 'Churn')
    
    Xtrain, Xtest, Ytrain,  Ytest = train_test_split(
        caracteristicas,
        objetivo, 
        random_state=1,
        test_size=0.2
    )
    
    pipeline = arbolDesicion(CRITERION, CLASS_WEIGHT, MAX_DEPTH)
    pipeline.fit(Xtrain, Ytrain)
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    
    yPredict= pipeline.predict(Xtest)
    print(classification_report(Ytest, yPredict))
    joblib.dump(pipeline, RUTA_MODELO)
    
    
    cm = confusion_matrix(Ytest, yPredict)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Matriz de Confusión - Predicción de Churn")
    plt.show()
    nombreColumnas = pipeline.named_steps['preprocesador'].get_feature_names_out()
    plt.figure(figsize=(20,10))
    plot_tree(
        pipeline.named_steps['arbolDesicion'], 
        feature_names= nombreColumnas, class_names=['No', 'Yes'],
        filled=True, max_depth=3) 
    plt.show()
    return 


if __name__ == "__main__":
    main()