import streamlit as st
import pandas as pd
import mlflow.pyfunc

################ variáveis necessárias para funcionamento do app
# caminho dos dados
dev_file = '../data/processed/prediction_test.parquet'
prod_file = '../data/processed/prediction_prod.parquet'

################ Predições do modelo
# Função para fazer predições
def make_prediction(input_data):
    df = pd.DataFrame(input_data, columns=['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance'])
    return model.predict(df)


################ Configuração inicial da página
st.set_page_config(
    page_title='Predicção de Arremeços do Kobe Bryant',
    page_icon=':basketball:',
)

# Configurações do MLFLOW na sidebar
st.sidebar.title("Configurações do MLflow")
tracking_uri = st.sidebar.text_input("MLflow Tracking URI", value='http://127.0.0.1:5001')
model_name = st.sidebar.text_input("Nome do Modelo", value='modelo_kobe')
model_version = st.sidebar.text_input("Versão do Modelo", value='23')

mlflow.set_tracking_uri(tracking_uri)

# Carregar o modelo do MLflow
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

st.title('Página Inicial')

# Armazenando o histórico de previsões na sessão
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

################ Lida com os dados


df_dev = pd.read_parquet(dev_file)
df_prod = pd.read_parquet(prod_file)

################ Importa o modelo do mlflow

#lat, log, minutes_remaining, period, playoffs, shot_distance 

######### containers da página

explicacao_container = st.container()
input_container = st.container()
resultado_container = st.container()
 

with explicacao_container:
    with st.expander("Explicação dos dados"):
        
        st.markdown("""
    #### Dados do Arremesso de Basquete

    Para cada arremesso, preencha os seguintes dados no formulário:

    - **Latitude (lat)**: Latitude do local do arremesso no formato decimal. Esse valor indica a posição norte-sul do arremesso na Terra.
    - **Longitude (log)**: Longitude do local do arremesso no formato decimal. Esse valor indica a posição leste-oeste do arremesso na Terra.
    - **Minutos Restantes (minutes_remaining)**: Número de minutos restantes no período (set) atual em que o arremesso ocorre. Essa informação ajuda a entender o momento do jogo em que o arremesso foi realizado.
    - **Período (period)**: Número do período (ou set) do jogo em que o arremesso ocorre. Um jogo de basquete típico tem quatro períodos de tempo regulamentar.
    - **Playoffs (playoffs)**: Indica se o lance ocorre durante os playoffs ou não. Insira `1` para sim e `0` para não.
    - **Distância do Lançamento (shot_distance)**: Distância, em metros, do ponto de arremesso até a cesta. Esta medida é crucial para análise de desempenho do arremessador.
    """)

with input_container:
    ic_col1, ic_col2, ic_col3 = st.columns(3)

    with ic_col1:
        # Slider para Latitude
        latitude = st.slider('Latitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.1)

        # Slider para Minutos Restantes
        minutes_remaining = st.slider('Minutos Restantes no Período', min_value=0, max_value=12, value=12)

    with ic_col2:
        # Slider para Longitude
        longitude = st.slider('Longitude', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)

        # Slider para Período
        period = st.slider('Período (Set)', min_value=1, max_value=4, value=1)

    with ic_col3:
        # Slider para Distância do Lançamento
        shot_distance = st.slider('Distância do Lançamento (em metros)', min_value=0, max_value=99, value=25)

        # usando captions vazias para melhorar o espaçamento
        st.caption('')
        st.caption('')
        # Checkbox para Playoffs
        playoffs = st.checkbox('Playoffs')
        
    submit = st.button('Prever lançamento')
    
    
    # Botão para realizar predição
    if submit:
        input_data = [[latitude, longitude, minutes_remaining, period, int(playoffs), shot_distance]]
        prediction = make_prediction(input_data)[0]
        result_text = "Cesta feita!" if prediction == 1 else "Cesta não feita."
        st.write(f"Predição de resultado: {result_text}")

        # Adicionar a predição ao histórico
        st.session_state.predictions.insert(0, {
            "Latitude": latitude,
            "Longitude": longitude,
            "Minutos Restantes": minutes_remaining,
            "Período": period,
            "Playoffs": playoffs,
            "Distância do Lançamento": shot_distance,
            "Resultado": result_text,
            "Versão do Modelo": model_version,
        })

# Exibindo o histórico de previsões
if st.session_state.predictions:
    df_predictions = pd.DataFrame(st.session_state.predictions)
    # Aplicar estilo para destacar a linha mais recente
    st.dataframe(df_predictions.style.apply(lambda x: ['background-color: lightgray' if x.name == 0 else '' for _ in x], axis=1))
else:
    st.write("Nenhuma predição foi realizada ainda.")

    