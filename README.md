# Introdução
Repositório de códigos e notebooks para o projeto final da matéria "Engenharia de machine learning" da pós-graduação MIT em inteligencia artificial do instituto infnet do aluno Rafael Lima Paulo.

# Dados
A base de dados utilizada nesse projeto, contida na pasta `data` também pode ser encontrada no kaggle através do link a seguir: https://www.kaggle.com/competitions/kobe-bryant-shot-selection/data

Nesta base constam os arremessos realizados pelo astro da NBA Kobe Bryant durante sua carreira.

# Como utilizar
É extremamente importante que o ambiente seja montado de maneira correta para o funcionamento deste projeto. O ambiente correto começa com a utilização do **python 3.10**.
### Estrutura

A pasta `data` contém os dados utilizados para execução do trabalho, assim como os dados gerados a partir do trabalho.

 - data.csv: O arquivo original, baixado do Kaggle para referencia, porém não utilizado no projeto.
 - raw: Contém os arquivos utilizados para execução do trabalho fornecidos pelo professor.
 - processed: Contém os arquivos gerados durante a execução do trabalho.

A pasta `code` contém todo o código necessário para avaliação e execução do modelo. Abaixo a lista dos arquivos mais importantes.

 - requirements.txt: Arquivo com as bibliotecas necessárias para execução do projeto. Essas bibliotecas precisam ser instaladas nas versões listadas para o correto funcionamento do projeto.
 - projeto_final.ipynb: Este é o notebook que contém toda a arquitetura inicial, processamento de dados, exploração, análise e treinamento dos modelos, assim como o monitoramento. É importante executar todos os blocos para  entender os outputs e verificar o resultado no mlflow.
 - application.py: Aqui você encontra a aplicação de produção, incluindo o monitoramento. é necessário executar este arquivo antes de abrir a aplicação no streamlit.
 - App.py: Aqui se encontra a página inicial do streamlit.
	 -  Contém a explicação dos dados e inclui a habilidade de testar o modelo mandando dados e recebendo a previsão. 
	 - Este arquivo precisa ser executado para abrir o streamlit.
	 - É importante garantir que as variáveis que indicam a versão e nome do modelo, assim como a url do mlflow contenham os valores corretos para que o modelo funcione.
 - pages/2_Monitoramento.py: Aqui se encontra a página de monitoramento visualizada no streamlit.
 -  pages/3_Comparação Dev e Prod.py: Aqui se encontram os plots comparando as estatísticas de teste e produção e pode ser visualizada no streamlit.

