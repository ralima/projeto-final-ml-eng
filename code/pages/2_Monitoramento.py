import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################ Configuração inicial da página
st.set_page_config(
    page_title='Predicção de Arremeços do Kobe Bryant - Monitoramento',
    page_icon=':basketball:',
)
st.title("Monitoramento Modelo em produção")

################ Lida com os dados
prod_file = '../data/processed/prediction_prod.parquet'
df_prod = pd.read_parquet(prod_file)
df_prod = df_prod.dropna()

# Função para plotar a distribuição das previsões
def plot_predictions(df_prod):
    fignum = plt.figure(figsize=(10,6))
    sns.histplot(df_prod['predict_score'], color="red", label='Produção', kde=True)
    plt.title('Distribuição das Probabilidades de Arremesso - Prod')
    plt.xlabel('Probabilidade de Acertar a Cesta')
    plt.ylabel('Densidade')
    plt.legend()
    st.pyplot(fignum)
    
# Função para mostrar a matriz de confusão
def show_confusion_matrix(df):
    
    # Convertendo os escores de previsão em classes preditas
    # predicted_classes = (data_clean['predict_score'] > 0.5).astype(int)
    # cm = metrics.confusion_matrix(df['shot_made_flag'], predicted_classes)
    # st.write('Matriz de Confusão:')
    # st.dataframe(cm)
    
    ##### ---
    # Remove linhas com NaN na coluna 'shot_made_flag'
    fignum = plt.figure(figsize=(6,4))
    
    data_clean = df.dropna(subset=['shot_made_flag'])
    
    # Convertendo os escores de previsão em classes preditas
    predicted_classes = (data_clean['predict_score'] > 0.5).astype(int)
    
    cm = confusion_matrix(data_clean['shot_made_flag'], predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Matriz de Confusão - Modelo de Produção')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')

    st.pyplot(fignum)
    plt.close()

# Função para criar e mostrar a ROC Curve
def plot_roc_curve(df):
    fignum = plt.figure(figsize=(10,6))
    fpr, tpr, _ = roc_curve(df['shot_made_flag'], df['predict_score'])
    roc_auc = auc(fpr, tpr)
    #plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(fignum)
    

st.markdown('#### plot predictions')
plot_predictions(df_prod)

st.markdown('#### roc curve')
plot_roc_curve(df_prod)

st.markdown('#### confusion maxitrx')
show_confusion_matrix(df_prod)

st.markdown('#### dataframe')
st.dataframe(df_prod)



