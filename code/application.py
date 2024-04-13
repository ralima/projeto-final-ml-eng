import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plot
import pycaret.classification as pycaret_classification

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Lançamentos do Kobe Bryant'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_name)

colunas = ['lat','lon','minutes_remaining', 'period', 'playoffs', 'shot_distance']

def plot_matriz_confusao(data_prod):
    # Remove linhas com NaN na coluna 'shot_made_flag'
    data_clean = data_prod.dropna(subset=['shot_made_flag'])
    
    # Convertendo os escores de previsão em classes preditas
    predicted_classes = (data_clean['predict_score'] > 0.5).astype(int)
    
    cm = confusion_matrix(data_clean['shot_made_flag'], predicted_classes)
    
    plot.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plot.title('Matriz de Confusão - Modelo de Produção')
    plot.xlabel('Predito')
    plot.ylabel('Verdadeiro')

    image = plot.gcf()
    image_name = 'matriz_confusao_producao.png'
    image.savefig(image_name, dpi=100)

    #plot.show()
    plot.close()
    return image_name

def plot_desempenho(data_prod):
    # Remove linhas com NaN na coluna 'shot_made_flag'
    data_clean = data_prod.dropna(subset=['shot_made_flag'])
    
    # Cálculo das métricas de desempenho na base de produção
    predicted_classes = (data_clean['predict_score'] > 0.5).astype(int)
    report = classification_report(data_clean['shot_made_flag'], predicted_classes, output_dict=True)
    # Convertendo o dicionário em um DataFrame para facilitar a visualização e o log
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    # Salvando o DataFrame como um arquivo CSV
    report_csv = 'classification_report.csv'
    report_df.to_csv(report_csv, index=True)
    return report_csv

    

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    Y = loaded_model.predict_proba(data_prod[colunas])[:, 1]
    
    data_prod['predict_score'] = Y
    prediction_file = '../data/processed/prediction_prod.parquet'
    data_prod.to_parquet(prediction_file)
    mlflow.log_artifact(prediction_file)
    
    # executa o plot da matriz de confusão e loga no mlflow
    image_name = plot_matriz_confusao(data_prod)
    mlflow.log_artifact(image_name)

    os.remove(image_name)
    
    # imprime o desempenho com os dados de peoducao
    report_name = plot_desempenho(data_prod)
    mlflow.log_artifact(report_name)
    os.remove(report_name)
    
    
    print(data_prod)
    
