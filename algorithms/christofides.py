from __future__ import annotations

import gc
import math
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from utilities.warm_start import normalizar_rota, para_lista_distancias

# ============================================================================
# DATACLASSES PARA AUDITORIA E DATASET DE EXPERIMENTOS
# ============================================================================

@dataclass
class MetricasSubrotina:
    tempo_ms: float
    memoria_pico_kb: float
    memoria_alocada_kb: float

@dataclass
class ResultadoChristofides:
    custo: int
    tour: List[int]
    n: int
    qtd_vertices_impares: int
    mst: MetricasSubrotina
    matching: MetricasSubrotina
    euleriano: MetricasSubrotina
    shortcutting: MetricasSubrotina
    tempo_total_ms: float
    memoria_pico_total_kb: float

    def para_dict(self) -> Dict[str, Any]:
        """Converte para dicionario achatado, pronto para exportar para DataFrame/CSV."""
        return {
            "n": self.n,
            "custo": self.custo,
            "qtd_vertices_impares": self.qtd_vertices_impares,
            # Gargalo 1: MST
            "tempo_mst_ms": self.mst.tempo_ms,
            "mem_pico_mst_kb": self.mst.memoria_pico_kb,
            # Gargalo 2: Matching
            "tempo_matching_ms": self.matching.tempo_ms,
            "mem_pico_matching_kb": self.matching.memoria_pico_kb,
            # Sub-rotinas auxiliares
            "tempo_euleriano_ms": self.euleriano.tempo_ms,
            "mem_pico_euleriano_kb": self.euleriano.memoria_pico_kb,
            "tempo_shortcutting_ms": self.shortcutting.tempo_ms,
            "mem_pico_shortcutting_kb": self.shortcutting.memoria_pico_kb,
            # Totais
            "tempo_total_ms": self.tempo_total_ms,
            "mem_pico_total_kb": self.memoria_pico_total_kb,
        }

# ============================================================================
# HARNESS DE PROFILING CONTROLADO DE SUB-ROTINAS
# ============================================================================

def perf_subrotina(func: Callable, *args, **kwargs) -> Tuple[Any, MetricasSubrotina]:
    """
    Executa uma sub-rotina isolando o impacto do Garbage Collector e
    medindo tempo de CPU/relógio de alta resolução e pico real de memória.
    """
    gc.collect()  # Elimina lixo prévio para não falsear a medição de memória
    tracemalloc.start()
    t_inicio = time.perf_counter()

    resultado = func(*args, **kwargs)

    t_fim = time.perf_counter()
    mem_atual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metricas = MetricasSubrotina(
        tempo_ms=(t_fim - t_inicio) * 1000.0,
        memoria_pico_kb=mem_pico / 1024.0,
        memoria_alocada_kb=mem_atual / 1024.0,
    )
    return resultado, metricas

# ============================================================================
# SUB-ROTINAS DO ALGORITMO (PURAS E AUDITÁVEIS)
# ============================================================================

def subrotina_mst_prim(dist: List[List[int]]) -> List[Tuple[int, int]]:
    """
    GARGALO 1: Árvore Geradora Mínima (MST).
    Implementação nativa de Prim para grafos completos em O(N^2).
    Evita a criação de objetos pesados de grafo para N <= 75.
    """
    n = len(dist)
    if n <= 1:
        return []

    in_mst = [False] * n
    min_cost = [math.inf] * n
    parent = [-1] * n

    min_cost[0] = 0
    mst_edges: List[Tuple[int, int]] = []

    for _ in range(n):
        u = -1
        menor = math.inf
        for i in range(n):
            if not in_mst[i] and min_cost[i] < menor:
                menor = min_cost[i]
                u = i

        if u == -1:
            break

        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))

        row_u = dist[u]
        for v in range(n):
            if not in_mst[v] and row_u[v] < min_cost[v]:
                min_cost[v] = row_u[v]
                parent[v] = u

    return mst_edges

def subrotina_vertices_impares(mst_edges: Sequence[Tuple[int, int]], n: int) -> List[int]:
    """Identifica os vértices com grau ímpar em O(N)."""
    graus = [0] * n
    for u, v in mst_edges:
        graus[u] += 1
        graus[v] += 1
    return [v for v in range(n) if graus[v] % 2 != 0]

def subrotina_matching_blossom(
    impares: Sequence[int],
    dist: List[List[int]]
) -> List[Tuple[int, int]]:
    """
    GARGALO 2: Emparelhamento Perfeito de Peso Mínimo (MWPM).
    Usa NetworkX estritamente como suporte do algoritmo das Flores (Blossom) em O(|O|^3).
    Apenas o subgrafo induzido pelos vértices ímpares é instanciado.
    """
    if not impares:
        return []

    subgrafo = nx.Graph()
    m = len(impares)

    for i in range(m):
        u = impares[i]
        row_u = dist[u]
        for j in range(i + 1, m):
            v = impares[j]
            subgrafo.add_edge(u, v, weight=row_u[v])

    matching = nx.min_weight_matching(subgrafo, weight="weight")
    return list(matching)

def subrotina_circuito_euleriano(
    arestas: Sequence[Tuple[int, int]],
    n: int,
    start: int = 0
) -> List[int]:
    """
    Circuito Euleriano via Algoritmo de Hierholzer em O(|V| + |E|).
    Usa IDs de arestas para evitar remoções lentas em multígrafos.
    """
    num_arestas = len(arestas)
    if num_arestas == 0:
        return [start]

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for edge_id, (u, v) in enumerate(arestas):
        adj[u].append((v, edge_id))
        adj[v].append((u, edge_id))

    usada = [False] * num_arestas
    pilha = [start]
    circuito: List[int] = []

    while pilha:
        u = pilha[-1]
        while adj[u] and usada[adj[u][-1][1]]:
            adj[u].pop()

        if adj[u]:
            v, edge_id = adj[u].pop()
            usada[edge_id] = True
            pilha.append(v)
        else:
            circuito.append(pilha.pop())

    return circuito[::-1]

def subrotina_shortcutting(circuito: Sequence[int], n: int) -> List[int]:
    """Constroi o ciclo Hamiltoniano a partir do circuito Euleriano."""
    visitados = [False] * n
    tour: List[int] = []

    for v in circuito:
        if not visitados[v]:
            visitados[v] = True
            tour.append(v)

    tour.append(tour[0])
    return normalizar_rota(tour, n)

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def executar_christofides_com_auditoria(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]]
) -> ResultadoChristofides:
    """
    Executa o Christofides auditando individualmente cada sub-rotina crítica.
    Retorna o objeto com a rota e todos os dados estatísticos para regressão.
    """
    dist = para_lista_distancias(matriz_distancias)
    n = len(dist)

    gc.collect()
    tracemalloc.start()
    t_global_inicio = time.perf_counter()

    # --- 1. PROFILING DA MST (Gargalo 1) ---
    mst_edges, metrica_mst = perf_subrotina(subrotina_mst_prim, dist)

    # Identificação de ímpares
    impares = subrotina_vertices_impares(mst_edges, n)

    # --- 2. PROFILING DO MATCHING (Gargalo 2) ---
    matching_edges, metrica_matching = perf_subrotina(
        subrotina_matching_blossom, impares, dist
    )

    # Multígrafo H = T U M
    multigrafo_arestas = mst_edges + matching_edges

    # --- 3. PROFILING DO CIRCUITO EULERIANO ---
    circuito, metrica_euleriano = perf_subrotina(
        subrotina_circuito_euleriano, multigrafo_arestas, n, 0
    )

    # --- 4. PROFILING DO SHORTCUTTING ---
    tour_normalizado, metrica_shortcutting = perf_subrotina(
        subrotina_shortcutting, circuito, n
    )

    t_global_fim = time.perf_counter()
    _, mem_pico_global = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    custo = sum(dist[tour_normalizado[i]][tour_normalizado[i + 1]] for i in range(n))

    return ResultadoChristofides(
        custo=custo,
        tour=tour_normalizado,
        n=n,
        qtd_vertices_impares=len(impares),
        mst=metrica_mst,
        matching=metrica_matching,
        euleriano=metrica_euleriano,
        shortcutting=metrica_shortcutting,
        tempo_total_ms=(t_global_fim - t_global_inicio) * 1000.0,
        memoria_pico_total_kb=mem_pico_global / 1024.0,
    )

def warm_start_christofides(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]]
) -> Tuple[int, List[int]]:
    """
    Interface padronizada para warm-start no Branch-and-Bound:
    Retorna apenas (custo, rota_normalizada).
    """
    resultado = executar_christofides_com_auditoria(matriz_distancias)
    return resultado.custo, resultado.tour