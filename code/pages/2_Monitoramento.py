import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns

################ Configuração inicial da página
st.set_page_config(
    page_title='Predicção de Arremeços do Kobe Bryant - Monitoramento',
    page_icon=':basketball:',
)
st.title("Monitoramento")

################ Lida com os dados
dev_file = '../data/processed/prediction_test.parquet'
prod_file = '../data/processed/prediction_prod.parquet'

df_dev = pd.read_parquet(dev_file)
df_prod = pd.read_parquet(prod_file)



################ Plot básico de produção
fignum = plt.figure(figsize=(6,4))

#Saída do modelo em dados dev
sns.distplot(df_dev.prediction_score_1,
            label='Teste',
            ax = plt.gca())

#Saída do modelo em dados prod
sns.distplot(df_prod.predict_score,
            label='Produção',
            ax = plt.gca())

plt.title('Monitoramento de desvio de dados da saída do modelo')
plt.ylabel('Distância')
plt.xlabel('Probabilidade de arremeço acertar a cesta')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')


st.pyplot(fignum)

st.write(metrics.classification_report(df_dev.shot_made_flag, df_dev.prediction_label))

