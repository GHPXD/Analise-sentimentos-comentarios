import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

def treinar_modelo():
    """
    Lê os dados do JSON, treina o modelo Naive Bayes e salva
    o modelo treinado e o vetorizador em arquivos .joblib.
    """
    # Carregando os dados do arquivo JSON
    with open('comentarios.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    comentarios_positivos = data['comentarios_positivos']
    comentarios_negativos = data['comentarios_negativos']

    # Combinando as listas e criando os rótulos (1 para positivo, 0 para negativo)
    comentarios = comentarios_positivos + comentarios_negativos
    rotulos = [1] * len(comentarios_positivos) + [0] * len(comentarios_negativos)

    # Vetorização dos comentários
    vetorizador = CountVectorizer()
    X = vetorizador.fit_transform(comentarios)

    # Treinando o modelo
    modelo = MultinomialNB()
    modelo.fit(X, rotulos)
    
    # Salvando o modelo e o vetorizador em arquivos
    dump(modelo, 'modelo_sentimento.joblib')
    dump(vetorizador, 'vetorizador.joblib')

    print("Modelo treinado e salvo com sucesso!")
    return modelo, vetorizador

def carregar_modelo_e_vetorizador():
    """
    Carrega o modelo e o vetorizador dos arquivos .joblib.
    """
    modelo = load('modelo_sentimento.joblib')
    vetorizador = load('vetorizador.joblib')
    return modelo, vetorizador

def prever_sentimento(modelo, vetorizador, novas_frases):
    """
    Usa um modelo e vetorizador carregados para prever o sentimento
    de uma lista de novas frases.
    """
    # Transformando as novas frases e fazendo previsões
    test = vetorizador.transform(novas_frases)
    previsoes = modelo.predict(test)

    return previsoes