import random
from collections import Counter


def gerar_jogo_mega_da_virada():
    numeros_por_jogo = 6
    numeros_sorteados = []

    # Adicione os números sorteados anteriormente
    numeros_sorteados.extend([
        [4, 5, 10, 34, 58, 59],  # 2022
        [12, 15, 23, 32, 33, 46],  # 2021
        [17, 20, 22, 35, 41, 42],  # 2020
        [3, 35, 38, 40, 57, 58],  # 2019
        [5, 10, 12, 18, 25, 33],  # 2018
        [3, 6, 10, 17, 34, 37],  # 2017
        [5, 11, 22, 24, 51, 53],  # 2016
        [2, 18, 31, 42, 51, 56],  # 2015
        [1, 5, 11, 16, 20, 56],  # 2014
        [20, 30, 36, 38, 47, 53],  # 2013
        [14, 32, 33, 36, 41, 52],  # 2012
        [3, 4, 29, 36, 45, 55],  # 2011
        [2, 10, 34, 37, 43, 50],  # 2010
        [10, 27, 40, 46, 49, 58],  # 2009
        [1, 11, 26, 51, 59, 60],  # 2008
    ])

    # Gerar um novo jogo baseado nos números sorteados
    novo_jogo = random.sample(range(1, 61), numeros_por_jogo)

    return novo_jogo


def gerar_multiplos_jogos(quantidade_jogos):
    jogos_gerados = []

    for _ in range(quantidade_jogos):
        novo_jogo = gerar_jogo_mega_da_virada()
        jogos_gerados.append(sorted(novo_jogo))

    return jogos_gerados


def analisar_numeros_sorteados(jogos_gerados):
    numeros_todos_jogos = [numero for jogo in jogos_gerados for numero in jogo]
    contagem_numeros = Counter(numeros_todos_jogos)
    contagem_numeros = dict(sorted(contagem_numeros.items()))

    return contagem_numeros


def gerar_jogo_com_base_na_frequencia(frequencia_numeros):
    numeros_mais_frequentes = [
        numero for numero, frequencia in frequencia_numeros.most_common(6)
    ]
    numeros_menos_frequentes = [
        numero
        for numero, frequencia in frequencia_numeros.most_common()[:-7:-1]
    ]

    jogo_gerado = numeros_mais_frequentes + numeros_menos_frequentes
    jogo_gerado = sorted(jogo_gerado)

    return jogo_gerado[:6]


def gerar_jogo_com_base_na_frequencia_corrigido(frequencia_numeros):
    numeros_ordenados_por_frequencia = sorted(
        frequencia_numeros, key=frequencia_numeros.get, reverse=True
    )

    numeros_mais_frequentes = numeros_ordenados_por_frequencia[:6]
    numeros_menos_frequentes = numeros_ordenados_por_frequencia[-6:]

    jogo_gerado = numeros_mais_frequentes + numeros_menos_frequentes
    jogo_gerado = sorted(jogo_gerado)

    return jogo_gerado[:6]


# Gerar 10 jogos
jogos_gerados = gerar_multiplos_jogos(10)

# Analisar a frequência dos números sorteados
frequencia_numeros = analisar_numeros_sorteados(jogos_gerados)

print("Jogos Gerados:")
for jogo in jogos_gerados:
    print(jogo)

print("\nFrequência dos Números Sorteados:")
for numero, frequencia in frequencia_numeros.items():
    print(f"Número {numero}: {frequencia} vezes")


# Chamada da função corrigida para gerar um jogo baseado na frequência dos números sorteados
jogo_baseado_na_frequencia_corrigido = (
    gerar_jogo_com_base_na_frequencia_corrigido(frequencia_numeros)
)
print("Jogo gerado com base na frequência dos números sorteados:")
print(jogo_baseado_na_frequencia_corrigido)
