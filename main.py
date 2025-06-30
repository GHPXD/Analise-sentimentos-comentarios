from modelo import treinar_modelo, prever_sentimento, carregar_modelo_e_vetorizador
from colorama import Fore, Style
import matplotlib.pyplot as plt
import os

def main():
    """
    Ponto de entrada principal do script. Carrega ou treina o modelo,
    classifica um conjunto de frases e exibe os resultados.
    """
    # Verifica se o modelo já foi treinado. Se não, treina um novo.
    if not os.path.exists('modelo_sentimento.joblib'):
        print("Modelo não encontrado. Treinando um novo modelo...")
        modelo, vetorizador = treinar_modelo()
    else:
        print("Carregando modelo existente...")
        modelo, vetorizador = carregar_modelo_e_vetorizador()
        print("Modelo carregado com sucesso!")

    # Novas frases para classificar
    novas_frases = [
        "O produto chegou muito rápido, fiquei impressionado com a entrega.",
        "Infelizmente, o item veio com defeito e não funciona corretamente.",
        "A qualidade do material é excelente, muito satisfeito com a compra.",
        "O atendimento ao cliente foi péssimo, não consegui resolver meu problema.",
        "Adorei a compra, super recomendo para quem busca qualidade.",
        "O produto não tem nada a ver com o anunciado, uma grande decepção.",
        "Estou muito feliz com o produto, ele atendeu todas as minhas expectativas.",
        "Demorou demais para a entrega e o produto não correspondeu às expectativas.",
        "Surpreendente, o produto é ótimo e funciona perfeitamente!",
        "Não vale a pena, o produto quebrou depois de uma semana de uso."
    ]

    # Realizando previsões
    previsoes = prever_sentimento(modelo, vetorizador, novas_frases)

    contagem_positivas = 0
    contagem_negativas = 0

    print("\n" + "="*50)
    print("Analisando Sentimentos das Frases:")
    print("="*50 + "\n")

    # Exibindo as previsões de forma formatada
    for frase, previsao in zip(novas_frases, previsoes):
        if previsao == 1:
            print(f"{Fore.GREEN}'{frase}' → Positivo{Style.RESET_ALL}")
            contagem_positivas += 1
        else:
            print(f"{Fore.RED}'{frase}' → Negativo{Style.RESET_ALL}")
            contagem_negativas += 1

    print("\n" + "="*50)
    print("Resumo da Análise")
    print("="*50 + "\n")
    print(f"Total de Frases Positivas: {contagem_positivas}")
    print(f"Total de Frases Negativas: {contagem_negativas}")
    print("\n" + "="*50)

    # Gerando o gráfico
    labels = ['Positivas', 'Negativas']
    counts = [contagem_positivas, contagem_negativas]
    colors = ['#4CAF50', '#F44336'] # Verde e Vermelho

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=colors)
    plt.title('Distribuição de Sentimentos das Frases Analisadas')
    plt.ylabel('Número de Frases')
    plt.xlabel('Sentimento')
    plt.show()

if __name__ == "__main__":
    main()