import pandas as pd
import numpy as np

import matplotlib.pyplot as plot
import pycaret.classification as pycaret_classification

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient




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

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    Y = loaded_model.predict_proba(data_prod[colunas])[:, 1]
    
    data_prod['predict_score'] = Y
    prediction_file = '../data/processed/prediction_prod.parquet'
    data_prod.to_parquet(prediction_file)
    mlflow.log_artifact(prediction_file)
    
    print(data_prod)
    
