import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import classification_report

import pandas as pd
import seaborn as sns

################ Configuração inicial da página
st.set_page_config(
    page_title='Predicção de Arremeços do Kobe Bryant - Prod vs Dev',
    page_icon=':basketball:',
)

st.title("Comparação Dev X Prod")

################ Lida com os dados
dev_file = '../data/processed/prediction_test.parquet'
prod_file = '../data/processed/prediction_prod.parquet'

df_dev = pd.read_parquet(dev_file)
df_prod = pd.read_parquet(prod_file)

################ Plots comparando teste e produção

container_header = st.container()

with container_header:
    # Cria subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Saída do modelo em dados de desenvolvimento (teste)
    sns.kdeplot(df_dev['prediction_score_1'], label='Teste', ax=axes[0])
    axes[0].set_title('Distribuição de Desenvolvimento (Teste)')
    axes[0].set_ylabel('Densidade')
    axes[0].set_xlabel('Probabilidade de acertar a cesta')
    axes[0].set_xlim((0,1))
    axes[0].grid(True)
    axes[0].legend(loc='best')

    # Saída do modelo em dados de produção
    sns.kdeplot(df_prod['predict_score'], label='Produção', ax=axes[1])
    axes[1].set_title('Distribuição de Produção')
    axes[1].set_ylabel('Densidade')
    axes[1].set_xlabel('Probabilidade de acertar a cesta')
    axes[1].set_xlim((0,1))
    axes[1].grid(True)
    axes[1].legend(loc='best')

    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()

    # Mostra o plot no Streamlit
    st.pyplot(fig)
    
col1, col2 = st.columns(2)

with col1:    
    fignum = plt.figure(figsize=(6,4))
    sns.distplot(df_dev.prediction_score_1, label='Teste', ax = plt.gca())
    sns.distplot(df_prod.predict_score, label='Produção', ax = plt.gca())

    plt.title('Monitoramento de desvio de dados da saída do modelo')
    plt.ylabel('Distância')
    plt.xlabel('Probabilidade de arremeço acertar a cesta')
    plt.xlim((0,1))
    plt.grid(True)
    plt.legend(loc='best')

    st.pyplot(fignum)
    plt.close()

with col2:

    fignum = plt.figure(figsize=(6,4))
    sns.kdeplot(df_dev['prediction_score_1'], label='Teste', ax=plt.gca())
    sns.kdeplot(df_prod['predict_score'], label='Produção', ax=plt.gca())
    plt.title('Monitoramento de desvio de dados da saída do modelo')
    plt.ylabel('Distância')
    plt.xlabel('Probabilidade de arremeço acertar a cesta')
    plt.xlim((0,1))
    plt.grid(True)
    plt.legend(loc='best')

    st.pyplot(fignum)

footer_container = st.container()

with footer_container:
    # Calcula o relatório de classificação
    report = classification_report(df_dev.shot_made_flag, df_dev.prediction_label, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(2)
    st.dataframe(report_df) 
