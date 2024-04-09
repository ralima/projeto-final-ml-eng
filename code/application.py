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
    
#     (retrain, [specificity_m, sensibility_m, precision_m],
#               [specificity_t, sensibility_t, precision_t] ) = alarm(data_novelty, pred_holdout, min_eff_alarm)
#     if retrain:
#         print('==> RETREINAMENTO NECESSARIO')
#     else:
#         print('==> RETREINAMENTO NAO NECESSARIO')
#     # LOG DE PARAMETROS DO MODELO
#     mlflow.log_param("min_eff_alarm", min_eff_alarm)

#     # LOG DE METRICAS GLOBAIS
#     mlflow.log_metric("Alarme Retreino", float(retrain))
#     mlflow.log_metric("Especificidade Controle", specificity_m)
#     mlflow.log_metric("Sensibilidade Controle", sensibility_m)
#     mlflow.log_metric("Precisao Controle", precision_m)
#     mlflow.log_metric("Especificidade Teste", specificity_t)
#     mlflow.log_metric("Sensibilidade Teste", sensibility_t)
#     mlflow.log_metric("Precisao Teste", precision_t)
#     # LOG ARTEFATO
#     var_name = 'volatile acidity' # 'alcohol'
#     data_drift_alarm(var_name, data_wine, pred_holdout, data_novelty)
#     plot_path = f'novidade_datadrift_{var_name}.png'
#     plt.savefig(plot_path)
#     mlflow.log_artifact(plot_path)

#     # LOG ARTEFATO
#     var_name = 'alcohol'
#     data_drift_alarm(var_name, data_wine, pred_holdout, data_novelty)
#     plot_path = f'novidade_datadrift_{var_name}.png'
#     plt.savefig(plot_path)
#     mlflow.log_artifact(plot_path)

# mlflow.end_run() 