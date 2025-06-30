from modelo import treinar_modelo, prever_sentimento
from colorama import Fore, Style
import matplotlib.pyplot as plt

def main():
    # Treinando o modelo
    modelo, vetorizador = treinar_modelo()

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
        "Não vale a pena, o produto quebrou depois de uma semana de uso.",
        "Produto de excelente qualidade, superou todas as minhas expectativas.",
        "O material é muito frágil, já estragou depois de poucas utilizações.",
        "Estou adorando o produto, com certeza compraria novamente.",
        "O produto não serve para o que eu precisava, muito decepcionante.",
        "Muito bem embalado e chegou intacto, recomendo para todos.",
        "A entrega foi rápida, mas o produto não vale o que cobram.",
        "O design é moderno e funcional, estou super satisfeito com a compra.",
        "A embalagem estava danificada, um verdadeiro descaso com o cliente.",
        "Produto bom, mas a entrega demorou mais do que o esperado.",
        "Fiquei muito insatisfeito com a compra, não vale o preço.",
        "O produto é exatamente o que eu queria, perfeito para o meu uso.",
        "Não cumpre o que promete, estou extremamente decepcionado.",
        "Produto ótimo, chegou antes do esperado e está funcionando perfeitamente.",
        "Tive uma péssima experiência, o produto não funcionou e o suporte não ajudou.",
        "Muito bonito, mas o material não é tão resistente quanto eu esperava.",
        "O produto chegou quebrado, um total desperdício de dinheiro.",
        "Muito satisfeito com a compra, foi um ótimo custo-benefício.",
        "Fiquei esperando mais de 10 dias e ainda não recebi meu pedido.",
        "A qualidade do produto é boa, mas o preço é um pouco elevado.",
        "Produto de excelente durabilidade, já usei várias vezes e continua ótimo."
    ]

    # Realizando previsões
    previsoes = prever_sentimento(modelo, vetorizador, novas_frases)

    # Contadores de frases positivas e negativas
    contagem_positivas = 0
    contagem_negativas = 0

    # Exibindo o cabeçalho
    print("Analisando Sentimentos das Frases:\n")
    print("-" * 50)

    # Exibindo as previsões de forma formatada
    for i, frase in enumerate(novas_frases):
        if previsoes[i] == 1:
            print(f"{Fore.GREEN}{frase} → É uma frase positiva{Style.RESET_ALL}")
            contagem_positivas += 1
        else:
            print(f"{Fore.RED}{frase} → É uma frase negativa{Style.RESET_ALL}")
            contagem_negativas += 1

    # Linha de separação
    print("\n" + "-" * 50)

    # Resumo das previsões
    print(f"\nResumo: {contagem_positivas} frase(s) positiva(s) e {contagem_negativas} frase(s) negativa(s).")
    print("-" * 50)

    # Gerando o gráfico
    labels = ['Positivas', 'Negativas']
    counts = [contagem_positivas, contagem_negativas]

    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Análise de Sentimentos das Frases')
    plt.ylabel('Número de Frases')
    plt.xlabel('Sentimento')
    plt.show()

if __name__ == "__main__":
    main()