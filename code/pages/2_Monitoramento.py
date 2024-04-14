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
st.title("Monitoramento do Modelo em produção")

################ Lida com os dados
prod_file = '../data/processed/prediction_prod.parquet'
df_prod = pd.read_parquet(prod_file)
df_prod = df_prod.dropna()

# Função para plotar a distribuição das previsões
def plot_previsoes(df_prod):
    fignum = plt.figure(figsize=(8,6))
    sns.histplot(df_prod['predict_score'], color="red", label='Produção', kde=True)
    plt.title('Distribuição das Probabilidades de Arremesso - Prod')
    plt.xlabel('Probabilidade de Acertar a Cesta')
    plt.ylabel('Densidade')
    plt.legend()
    st.pyplot(fignum)
    
# Função para mostrar a matriz de confusão
def plot_matriz_confusao(data_prod):
    fignum = plt.figure(figsize=(8,6))
    data_clean = data_prod.dropna(subset=['shot_made_flag'])
    
    # Convertendo os escores de previsão em classes preditas
    predicted_classes = (data_clean['predict_score'] > 0.5).astype(int)
    
    cm = confusion_matrix(data_clean['shot_made_flag'], predicted_classes)
    
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Matriz de Confusão - Modelo de Produção')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')

    st.pyplot(fignum)


# Função para criar e mostrar a Curva ROC
def plot_curva_roc(df):
    fignum = plt.figure(figsize=(8,6))
    fpr, tpr, _ = roc_curve(df['shot_made_flag'], df['predict_score'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    st.pyplot(fignum)
    
def plot_residual(df):
    # Residual Plot (Normalmente mais aplicável a regressão)
    residuals = df['shot_made_flag'] - df['predict_score']
    fig, ax = plt.subplots()
    sns.residplot(x=df['predict_score'], y=residuals, lowess=True, ax=ax, line_kws={'color': 'red', 'lw': 1})
    #sns.residplot(x=df['predict_score'], y=residuals, lowess=True, line_kws={'color': 'red', 'lw': #1})
    plt.title('Residual Plot')
    plt.xlabel('Probabilidades previstas')
    plt.ylabel('Residuais')
    plt.legend(loc="upper right")
    st.pyplot(fig)
    
col1, col2 = st.columns(2)
with col1:
    plot_previsoes(df_prod)
    plot_residual(df_prod)
with col2:
    plot_curva_roc(df_prod)
    plot_matriz_confusao(df_prod)

st.dataframe(df_prod)
