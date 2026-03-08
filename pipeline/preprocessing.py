import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import os

caracteristicasNumericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
caracteristicasNominales = ['PaymentMethod', 'InternetService']
caracteristicasBinarias= ['gender' ,'SeniorCitizen' , 'Partner' ,'StreamingMovies', 
                                'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup', 
                                'OnlineSecurity' ,'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling']
caracteristicasOrdinales =['Contract']

def leerDataset(service):
    print(len(service.leerDataset()))
    return service.leerDataset()
    
def arbolDesicion(criterion, classWeight, maxDepth):
    preprocesador  = construirPipeline()
    arbolDesicion = DecisionTreeClassifier(random_state=1,criterion=criterion, class_weight=classWeight ,max_depth=maxDepth)
    pipeline = Pipeline(
        steps=[
            ('preprocesador', preprocesador),
            ('arbolDesicion', arbolDesicion)
        ]
    )
    return pipeline

def randomForest(nEstimators, criterion, maxDepth, classWeight, maxFeatures, bootstrap, maxSamples, oobScore):
    preprocesador= construirPipeline()
    randomForest = RandomForestClassifier(n_estimators=nEstimators, criterion= criterion, 
                                          max_depth=maxDepth, class_weight= classWeight, max_features=maxFeatures, 
                                          bootstrap=bootstrap, max_samples=maxSamples, oob_score=oobScore)
    pipeline = Pipeline(
        steps=[
            ('preprocesador' ,preprocesador), 
            ('randomForest', randomForest)
        ]
    )
    return pipeline

def construirPipeline():
    transformadorNumerico = Pipeline(
    steps=[
        # ('imputer', SimpleImputer(strategy= 'median')),
        ('scalar', StandardScaler())        
    ]
    )
    contractOrden = [['Month-to-month', 'One year', 'Two year']]
    
    transformadorOrdinal = Pipeline(
        steps=[
            ('ordinal', OrdinalEncoder(categories=contractOrden))         
        ]
    )

    transformadorNominal = Pipeline(
        steps=[
            ('oneHot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    transformadorBinario= Pipeline(
        steps=[
            ('binario', OrdinalEncoder())
        ]
    )
    preprocesador = ColumnTransformer(
    transformers=[
        ('binario', transformadorBinario, caracteristicasBinarias),
        ('numerico', transformadorNumerico, caracteristicasNumericas),
        ('ordinal', transformadorOrdinal, caracteristicasOrdinales), 
        ('nominal', transformadorNominal, caracteristicasNominales)
    ]
        )
    return preprocesador

def separarCaracteristicas(df, objetivo):   
    
    df= limpiarDatos(df)    
    # dfTargetYes= df[df[objetivo]=='Yes']
    # dfTargetNo= df[df[objetivo]=='No']
    
    # dfChurnNo = dfTargetNo.sample(len(dfTargetYes), random_state=1)
    # dfBalanceado = pd.concat([dfChurnNo, dfTargetYes]) 
    
    # target = dfBalanceado[objetivo]
    
    # caracteristicas = dfBalanceado.drop(columns=[objetivo, 'customerID'])
    caracteristicas= df.drop(columns=[objetivo, 'customerID'])
    target = df[objetivo]
    
    return caracteristicas, target

def limpiarDatos(df):
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors= 'coerce')
    
    df.dropna(inplace=True)
    
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')    
    
    columnasInternet = ['StreamingMovies', 'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity',]

    for col in columnasInternet:
        df[col]= df[col].replace('No internet service', 'No')
    
    return df