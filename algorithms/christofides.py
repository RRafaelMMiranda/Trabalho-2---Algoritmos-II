from __future__ import annotations

import time
import tracemalloc
from typing import List, Sequence, Tuple, Union, Dict, Any
import networkx as nx
import numpy as np

# Importa as utilidades que o warm_start usa
# ajustar o caminho...
from utilities.warm_start import normalizar_rota, calcular_custo_rota, para_lista_distancias

def christofides_solver(
    dist: List[List[int]],
    coletar_metricas: bool = False
) -> Tuple[int, List[int], Dict[str, Any]]:
    """
    Executa o algoritmo de aproximacao de Christofides (garantia 1.5).
    Instrumentado para medir tempo e memoria de cada subrotina.
    """
    n = len(dist)
    if n <= 2:
        tour = list(range(n)) + [0]
        return calcular_custo_rota(tour, dist), normalizar_rota(tour, n), {}

    metricas: Dict[str, Any] = {
        'n': n,
        'tempo_mst': 0.0,
        'mem_mst_kb': 0.0,
        'tempo_impares': 0.0,
        'num_impares': 0,
        'tempo_matching': 0.0,
        'mem_matching_kb': 0.0,
        'tempo_euleriano': 0.0,
        'tempo_atalhos': 0.0,
        'tempo_total': 0.0,
    }

    t_inicio_total = time.perf_counter()

    # -------------------------------------------------------------
    # 1. Arvore Geradora Minima (MST) - O(N^2)
    # -------------------------------------------------------------
    if coletar_metricas:
        tracemalloc.start()
    t0 = time.perf_counter()

    G = nx.Graph()
    for i in range(n):
        row_i = dist[i]
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=row_i[j])
    
    mst = nx.minimum_spanning_tree(G, algorithm='prim')

    metricas['tempo_mst'] = time.perf_counter() - t0
    if coletar_metricas:
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metricas['mem_mst_kb'] = pico / 1024.0

    # -------------------------------------------------------------
    # 2. Identificar vertices de grau impar na MST
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    vertices_impares = [v for v, grau in mst.degree() if grau % 2 != 0]
    metricas['tempo_impares'] = time.perf_counter() - t0
    metricas['num_impares'] = len(vertices_impares)

    # -------------------------------------------------------------
    # 3. Minimum-Weight Perfect Matching (MWPM) - Gargalo O(|O|^3)
    # -------------------------------------------------------------
    if coletar_metricas:
        tracemalloc.start()
    t0 = time.perf_counter()

    subgrafo = nx.Graph()
    qtd_impares = len(vertices_impares)
    for i in range(qtd_impares):
        u = vertices_impares[i]
        row_u = dist[u]
        for j in range(i + 1, qtd_impares):
            v = vertices_impares[j]
            subgrafo.add_edge(u, v, weight=row_u[v])

    # Algoritmo de Blossom (Edmonds)
    matching = nx.min_weight_matching(subgrafo, maxcardinality=True)

    metricas['tempo_matching'] = time.perf_counter() - t0
    if coletar_metricas:
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metricas['mem_matching_kb'] = pico / 1024.0

    # -------------------------------------------------------------
    # 4. Multigrafo e Circuito Euleriano
    # -------------------------------------------------------------
    t0 = time.perf_counter()

    multigrafo = nx.MultiGraph()
    multigrafo.add_edges_from(mst.edges(data=True))
    for u, v in matching:
        multigrafo.add_edge(u, v, weight=dist[u][v])

    circuito_euleriano = list(nx.eulerian_circuit(multigrafo, source=0))
    metricas['tempo_euleriano'] = time.perf_counter() - t0

    # -------------------------------------------------------------
    # 5. Atalhos (Shortcutting) para formar ciclo Hamiltoniano
    # -------------------------------------------------------------
    t0 = time.perf_counter()

    visitados = set()
    tour_bruto = []
    for u, _ in circuito_euleriano:
        if u not in visitados:
            tour_bruto.append(u)
            visitados.add(u)

    # Se por desconexao algum vertice isolado ficou de fora (seguranca)
    if len(visitados) < n:
        for v in range(n):
            if v not in visitados:
                tour_bruto.append(v)
                visitados.add(v)

    tour_final = normalizar_rota(tour_bruto, n)
    metricas['tempo_atalhos'] = time.perf_counter() - t0

    # Consolidacao final
    metricas['tempo_total'] = time.perf_counter() - t_inicio_total
    custo = calcular_custo_rota(tour_final, dist)
    metricas['custo_tour'] = custo

    return custo, tour_final, metricas

# =====================================================================
# FUNCAO PADRAO DE WARM-START (para o seu colega importar no B&B)
# =====================================================================
def warm_start_christofides(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]]
) -> Tuple[int, List[int]]:
    """
    5. Christofides: Usa a solucao aproximada de Christofides como limite inicial.
    Retorna (custo, rota_normalizada).
    """
    dist = para_lista_distancias(matriz_distancias)
    custo, rota, _ = christofides_solver(dist, coletar_metricas=False)
    return custo, rota