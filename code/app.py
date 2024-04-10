import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

dev_file = '../data/processed/prediction_test.parquet'
prod_file = '../data/processed/prediction_prod.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Dados de arremessos para predicção.
""")

df_dev = pd.read_parquet(dev_file)
df_prod = pd.read_parquet(prod_file)



fignum = plt.figure(figsize=(6,4))

#Saída do modelo em dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax = plt.gca())

#Saída do modelo em dados prod
sns.distplot(df_prod.predict_score,
             label='Produção',
             ax = plt.gca())
# User wine

plt.title('Monitoramento de desvio de dados da saída do modelo')
plt.ylabel('Distância')
plt.xlabel('Probabilidade de arremeço acertar a cesta')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')


st.pyplot(fignum)

st.write(metrics.classification_report(df_dev.shot_made_flag, df_dev.prediction_label))
