from google_auth_oauthlib.flow import InstalledAppFlow

def main():
    # Inicia o fluxo OAuth para obter o token de atualização
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret.json',  # Substitua com o caminho para o arquivo JSON do client secret
        scopes=['https://www.googleapis.com/auth/adwords']
    )
    credentials = flow.run_local_server(port=0)

    print('Refresh token:', credentials.refresh_token)

if __name__ == '__main__':
    main()
