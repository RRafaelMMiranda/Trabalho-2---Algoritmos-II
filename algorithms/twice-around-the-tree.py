from __future__ import annotations

import gc
import time
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np


def mst_prim(matriz: np.ndarray, raiz: int = 0) -> list[list[int]]:
    """Retorna a MST como lista de adjacência: filhos[v] = lista de filhos de v.

    Algoritmo de Prim sobre o grafo completo (matriz densa), O(N^2) tempo e O(N)
    memória extra: a árvore cresce a partir de `raiz`, e a cada passo entra o
    vértice fora da árvore com menor `chave` (menor aresta que o liga à árvore).
    Como a árvore já nasce enraizada, `pai[v]` dá direto a lista de filhos.
    Pesos 0 (pontos duplicados) não precisam de tratamento especial aqui.
    """
    n = matriz.shape[0]
    filhos: list[list[int]] = [[] for _ in range(n)]
    if n <= 1:
        return filhos

    pesos = np.asarray(matriz)
    chave = pesos[raiz].copy()              # menor peso conhecido até cada vértice
    pai = np.full(n, raiz, dtype=np.int64)  # vértice da árvore que atinge v com `chave[v]`
    fora = np.ones(n, dtype=bool)           # True enquanto v ainda não está na árvore
    fora[raiz] = False

    for _ in range(n - 1):
        candidatos = np.where(fora, chave, np.inf)
        u = int(np.argmin(candidatos))
        fora[u] = False
        filhos[int(pai[u])].append(u)

        melhora = fora & (pesos[u] < chave)
        chave[melhora] = pesos[u][melhora]
        pai[melhora] = u
    return filhos


def preordem(filhos: list[list[int]], raiz: int = 0) -> list[int]:
    """Percurso em pré-ordem da árvore, a partir da raiz.

    Iterativo (pilha explícita): evita o limite de recursão do Python, que uma
    MST em forma de caminho com N grande estouraria.
    """
    ordem: list[int] = []
    pilha = [raiz]
    while pilha:
        v = pilha.pop()
        ordem.append(v)
        pilha.extend(reversed(filhos[v]))  # mantém a ordem de inserção dos filhos
    return ordem


def custo_ciclo(ciclo: Sequence[int], matriz: np.ndarray) -> int:
    """Soma das distâncias das arestas do ciclo (aceita rota aberta ou fechada [r,...,r])."""
    n = len(ciclo)
    if n < 2:
        return 0
    fechada = ciclo[0] == ciclo[-1]
    limite = n - 1 if fechada else n
    custo = 0
    for i in range(limite):
        u = ciclo[i]
        v = ciclo[i + 1] if fechada else ciclo[(i + 1) % limite]
        custo += int(matriz[u][v])
    return custo


def medir_subrotina(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Executa `func` e devolve (resultado, tempo em segundos).

    Isola o coletor de lixo: faz gc.collect() antes e o desliga durante a medida,
    para que uma coleta no meio da subrotina não vire "tempo do algoritmo".
    Mede só tempo (sem tracemalloc, que deixa o código mais lento e distorceria o resultado).
    """
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        resultado = func(*args, **kwargs)
        tempo = time.perf_counter() - t0
    finally:
        gc.enable()
    return resultado, tempo


def twice_around_the_tree(matriz: np.ndarray, raiz: int = 0) -> dict:
    """Executa o algoritmo e devolve ciclo (fechado), custo e tempos por subrotina (s).

    tempos: "mst", "preordem", "custo" e "total" (= soma das três subrotinas).
    """
    matriz = np.asarray(matriz)
    tempos: dict[str, float] = {}

    filhos, tempos["mst"] = medir_subrotina(mst_prim, matriz, raiz)
    ordem, tempos["preordem"] = medir_subrotina(preordem, filhos, raiz)
    ciclo = ordem + [raiz]
    custo, tempos["custo"] = medir_subrotina(custo_ciclo, ciclo, matriz)
    tempos["total"] = tempos["mst"] + tempos["preordem"] + tempos["custo"]

    return {
        "ciclo": ciclo,
        "custo": custo,
        "tempos": tempos,
    }


def warm_start_twice_around_the_tree(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]],
) -> Tuple[int, List[int]]:
    """Interface padronizada de warm-start para o Branch-and-Bound: (custo, rota fechada)."""
    resultado = twice_around_the_tree(np.asarray(matriz_distancias))
    return int(resultado["custo"]), list(resultado["ciclo"])
