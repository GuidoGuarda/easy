import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import openai
import bcrypt
import requests
import json
import certifi
from google.cloud import dialogflow_v2 as dialogflow
from textblob import TextBlob
from transformers import pipeline
from sklearn.linear_model import LinearRegression
from PIL import Image
from google.ads.googleads.client import GoogleAdsClient
from google_auth_oauthlib.flow import InstalledAppFlow
import json

# Para carregar um arquivo JSON
#with open('client_secret.json') as f:
   # dados = json.load(f)

# Para salvar dados em um arquivo JSON
#with open('client_secret.json', 'w') as f:
   # json.dump(dados, f, indent=4)

# Fazer uma requisição à API
#response = requests.get('https://googleads.googleapis.com/')
response = requests.get('https://googleads.googleapis.com/', verify=certifi.where())
print(response.content)

# Verificar o status da resposta e se contém dados
if response.status_code == 200 and response.text.strip():
    try:
        data = response.json()  # Tentar decodificar como JSON
        print(data)
    except json.JSONDecodeError:
        print("A resposta não está em formato JSON válido.")
else:
    print("A resposta da API está vazia ou ocorreu um erro na requisição.")

# Substitua o caminho pelo caminho correto para o seu arquivo client_secret.json
flow = InstalledAppFlow.from_client_secrets_file(
    'C:/scienceafter/Guido/integrations/client_secret.json',
    scopes=['https://www.googleapis.com/auth/adwords'])

# Executa o servidor local para o usuário autenticar
flow.run_local_server(port=8080, prompt='consent')

# Obtenha o refresh token
credentials = flow.credentials
print("Refresh Token:", credentials.refresh_token)

# from google.ads.google_ads.client import GoogleAdsClient


client = GoogleAdsClient.load_from_storage(r"C:\scienceafter\Guido\easyads1\integrations\google_ads.yaml")


# Criptografando
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Configuração da chave da API para chat bot
openai.api_key = "sua_chave_api"

def chat_with_bot(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}],
    )
    return response.choices[0].message['content']

def detect_intent_texts(project_id, session_id, texts, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )
        return response.query_result.fulfillment_text

def analisar_sentimento(texto):
    blob = TextBlob(texto)
    sentimento = blob.sentiment.polarity
    if sentimento > 0:
        return "Positivo"
    elif sentimento == 0:
        return "Neutro"
    else:
        return "Negativo"

# Título do aplicativo
st.title("Easy Ads Traffic Management")

# Upload de imagem como plano de fundo
uploaded_file = st.file_uploader("home.jpg", type=['jpg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

# Chatbot
user_input = st.text_input("Como posso ajudar você?")
if user_input:
    bot_response = chat_with_bot(user_input)
    st.write("Chatbot:", bot_response)

# Previsão de tendências futuras
X = np.array([1, 2, 3]).reshape(-1, 1)  # Períodos (meses)
y = np.array([1000, 1500, 1200])  # Impressões

model = LinearRegression().fit(X, y)
previsao = model.predict(np.array([[4]]))  # Previsão para o mês 4
st.write("Previsão de impressões para o mês 4:", previsao[0])

# Análise de sentimentos
texto = st.text_input("Digite um texto para análise de sentimento:")
if texto:
    resultado_sentimento = analisar_sentimento(texto)
    st.write("Análise de Sentimento:", resultado_sentimento)

# Análise Fácil de Campanhas
st.subheader("Análise Fácil de Campanhas")
cliente_id = st.text_input('Insira o ID do Cliente:')
if st.button('Carregar Dados'):
    st.write(f'Exibindo dados para o cliente: {cliente_id}')

    # Exemplo de dados
    data = {
        'campanha': ['Campanha 1', 'Campanha 2', 'Campanha 3'],
        'cliques': [100, 150, 80],
        'impressões': [1000, 1200, 900]
    }
    df = pd.DataFrame(data)

    # Gráfico de cliques por campanha
    fig = px.bar(df, x='campanha', y='cliques', title='Cliques por Campanha')
    st.plotly_chart(fig)

# Classificação de sentimento usando Transformers
classifier = pipeline("sentiment-analysis")
if st.button("Analisar Sentimento de Texto"):
    texto_analisar = "Este anúncio não teve nenhum impacto no público."
    resultado = classifier(texto_analisar)
    st.write("Resultado da Análise de Sentimento:", resultado)

# Simulação de dados de campanhas ao longo do tempo
data_tempo = {
    'data': ['2024-01-01', '2024-02-01', '2024-03-01'],
    'impressões': [1000, 1500, 1200],
    'cliques': [100, 180, 160]
}
df_tempo = pd.DataFrame(data_tempo)
df_tempo['data'] = pd.to_datetime(df_tempo['data'])

# Análise básica de tendências
df_tempo.set_index('data', inplace=True)
st.line_chart(df_tempo['impressões'], title='Impressões ao longo do tempo')
