# modelo.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from comentarios import comentarios_positivos, comentarios_negativos

def treinar_modelo():
    # Definindo os rótulos (1 para positivo, 0 para negativo)
    comentarios = comentarios_positivos + comentarios_negativos
    rotulos = [1] * len(comentarios_positivos) + [0] * len(comentarios_negativos)

    # Vetorização dos comentários
    vetorizador = CountVectorizer()
    X = vetorizador.fit_transform(comentarios)

    # Treinando o modelo
    modelo = MultinomialNB()
    modelo.fit(X, rotulos)

    return modelo, vetorizador

def prever_sentimento(modelo, vetorizador, novas_frases):
    # Transformando as novas frases e fazendo previsões
    test = vetorizador.transform(novas_frases)
    previsoes = modelo.predict(test)

    return previsoes