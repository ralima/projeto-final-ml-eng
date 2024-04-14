import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plot
import matplotlib as mpl
import pycaret.classification as pycaret_classification
from pycaret.classification import setup, predict_model

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import seaborn as sns

if 'inline_rc' not in dir():
    inline_rc = dict(mpl.rcParams)

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
coluna_alvo = 'shot_made_flag'

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

# reset matplotlib

mpl.rcParams.update(inline_rc)
font = {'size'   : 14}
mpl.rc('font', **font)
lines = {'linewidth' : 3}
mpl.rc('lines', **lines)

def data_drift_alarm(var_name, dev_data, data_test, data_prod):
    plot.figure(figsize=(6,4))
    sns.kdeplot(dev_data[var_name], label='Desenvolvimento')
    sns.kdeplot(data_test[var_name], label='Teste')
    sns.kdeplot(data_prod[var_name], label='Produção')
    plot.grid()
    plot.legend(loc='best')
    plot.title(f'Distribuição Variável {var_name}')
    plot.ylabel('Densidade')
    plot.xlabel(f'Unidade de {var_name}')
    plot.tight_layout()
    
def alarmz(data_monitoring, testset, min_eff_alarm):
    cm = metrics.confusion_matrix(data_monitoring[coluna_alvo], data_monitoring['predict_score'])
    specificity_m = cm[0,0] / cm.sum(axis=1)[0]
    sensibility_m = cm[1,1] / cm.sum(axis=1)[1]
    precision_m   = cm[1,1] / cm.sum(axis=0)[1]

    cm = metrics.confusion_matrix(testset[coluna_alvo], testset['prediction_label'])
    specificity_t = cm[0,0] / cm.sum(axis=1)[0]
    sensibility_t = cm[1,1] / cm.sum(axis=1)[1]
    precision_t   = cm[1,1] / cm.sum(axis=0)[1]

    retrain = False
    for name, metric_m, metric_t in zip(['especificidade', 'sensibilidade', 'precisao'],
                                        [specificity_m, sensibility_m, precision_m],
                                        [specificity_t, sensibility_t, precision_t]):
        
        print(f'\t=> {name} de teste {metric_t} e de controle {metric_m}')
        if (metric_t-metric_m)/metric_t > min_eff_alarm:
            print(f'\t=> MODELO OPERANDO FORA DO ESPERADO')
            retrain = True
        else:
            print(f'\t=> MODELO OPERANDO DENTRO DO ESPERADO')
           
        
    return (retrain, [specificity_m, sensibility_m, precision_m],
                                        [specificity_t, sensibility_t, precision_t] )

def alarm(data_monitoring, testset, min_eff_alarm):
    # Aqui você converte os scores de previsão em classes preditas com um limiar de 0.5
    predicted_classes_monitoring = (data_monitoring['predict_score'] > 0.5).astype(int)
    predicted_classes_testset = (testset['predict_score'] > 0.5).astype(int)

    # Agora calcule a matriz de confusão usando as classes preditas
    cm = metrics.confusion_matrix(data_monitoring[coluna_alvo], predicted_classes_monitoring)
    specificity_m = cm[0,0] / (cm.sum(axis=1)[0] if cm.sum(axis=1)[0] != 0 else 1)
    sensibility_m = cm[1,1] / (cm.sum(axis=1)[1] if cm.sum(axis=1)[1] != 0 else 1)
    precision_m   = cm[1,1] / (cm.sum(axis=0)[1] if cm.sum(axis=0)[1] != 0 else 1)

    cm = metrics.confusion_matrix(testset[coluna_alvo], predicted_classes_testset)
    specificity_t = cm[0,0] / (cm.sum(axis=1)[0] if cm.sum(axis=1)[0] != 0 else 1)
    sensibility_t = cm[1,1] / (cm.sum(axis=1)[1] if cm.sum(axis=1)[1] != 0 else 1)
    precision_t   = cm[1,1] / (cm.sum(axis=0)[1] if cm.sum(axis=0)[1] != 0 else 1)

    retrain = False
    for name, metric_m, metric_t in zip(['especificidade', 'sensibilidade', 'precisao'],
                                        [specificity_m, sensibility_m, precision_m],
                                        [specificity_t, sensibility_t, precision_t]):
        
        print(f'\t=> {name} de teste {metric_t} e de controle {metric_m}')
        if (metric_t-metric_m)/metric_t > min_eff_alarm:
            print(f'\t=> MODELO OPERANDO FORA DO ESPERADO')
            retrain = True
        else:
            print(f'\t=> MODELO OPERANDO DENTRO DO ESPERADO')
        
    return (retrain, [specificity_m, sensibility_m, precision_m],
                     [specificity_t, sensibility_t, precision_t])
    

print('== ALARME DE RETREINAMENTO - BASE CONTROLE ==')
# 10% de desvio aceitavel na metrica. Deve ser estimado pelo conjunto de validacao cruzada. 
min_eff_alarm = 0.1 
mlflow.end_run()
with mlflow.start_run(experiment_id=experiment_id, run_name = 'MonitoramentoProducao'):
    
    model_uri = f"models:/modelo_kobe@staging"

    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_control = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet').dropna()
    setup(data=data_control, target='shot_made_flag')
    Y = loaded_model.predict_proba(data_control[colunas])[:, 1]
    data_control['predict_score'] = Y
    
    pred_holdout = pycaret_classification.predict_model(loaded_model, data=data_control, raw_score=True)
    
    (retrain, [specificity_m, sensibility_m, precision_m],
              [specificity_t, sensibility_t, precision_t] ) = alarm(data_control, pred_holdout, min_eff_alarm)
    if retrain:
        print('==> RETREINAMENTO NECESSARIO')
    else:
        print('==> RETREINAMENTO NAO NECESSARIO')
    # LOG DE PARAMETROS DO MODELO
    mlflow.log_param("min_eff_alarm", min_eff_alarm)

    # LOG DE METRICAS GLOBAIS
    mlflow.log_metric("Alarme Retreino", float(retrain))
    mlflow.log_metric("Especificidade Controle", specificity_m)
    mlflow.log_metric("Sensibilidade Controle", sensibility_m)
    mlflow.log_metric("Precisao Controle", precision_m)
    mlflow.log_metric("Especificidade Teste", specificity_t)
    mlflow.log_metric("Sensibilidade Teste", sensibility_t)
    mlflow.log_metric("Precisao Teste", precision_t)
    
    # LOG ARTEFATO
    var_name = 'shot_distance'
    data_drift_alarm(var_name, data_control, pred_holdout, data_control)
    plot_path = f'monitor_datadrift_{var_name}.png'
    plot.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)
    
    # LOG ARTEFATO
    var_name = coluna_alvo
    data_drift_alarm(var_name, data_control, pred_holdout, data_control)
    plot_path = f'monitor_datadrift_{var_name}.png'
    plot.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)
    

mlflow.end_run()   


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
    
