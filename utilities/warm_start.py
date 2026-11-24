from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import numpy as np

from entities.node import Node


def para_lista_distancias(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]],
) -> List[List[int]]:
    """Converte a matriz de distancias para listas nativas do Python (acesso mais rapido em lacos)."""
    if isinstance(matriz_distancias, np.ndarray):
        return matriz_distancias.tolist()
    return [list(row) for row in matriz_distancias]


def calcular_custo_rota(
    tour: Sequence[int],
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]],
) -> int:
    """Calcula o custo de um ciclo Hamiltoniano (aceita rota fechada [0,...,0] ou aberta)."""
    if not tour:
        return 0
    dist = para_lista_distancias(matriz_distancias) if isinstance(matriz_distancias, np.ndarray) else matriz_distancias
    fechada = len(tour) > 1 and tour[0] == tour[-1]
    limite = len(tour) - 1 if fechada else len(tour)
    custo = 0
    for i in range(limite):
        u = tour[i]
        v = tour[i + 1] if fechada else tour[(i + 1) % limite]
        custo += int(dist[u][v])
    return custo


def normalizar_rota(tour: Sequence[int], n: int) -> List[int]:
    """Valida e normaliza a rota no formato fechado [0, ..., 0] com n + 1 elementos."""
    seq = list(tour)
    if len(seq) == n + 1 and seq[0] == seq[-1]:
        seq = seq[:-1]
    if len(seq) != n or len(set(seq)) != n:
        raise ValueError(f"Rota invalida: esperado ciclo com {n} vertices distintos.")
    idx0 = seq.index(0)
    return seq[idx0:] + seq[:idx0] + [0]


def warm_start_cold_start(dist: List[List[int]]) -> Tuple[int, List[int]]:
    """1. Cold Start / Trivial: rota ingenua 0 -> 1 -> ... -> n-1 -> 0."""
    n = len(dist)
    tour = list(range(n)) + [0]
    custo = sum(dist[tour[i]][tour[i + 1]] for i in range(n))
    return custo, tour


def warm_start_nearest_neighbor(dist: List[List[int]], start: int = 0) -> Tuple[int, List[int]]:
    """2. Busca Gulosa: Heuristica do Vizinho Mais Proximo (Nearest Neighbor), O(N^2)."""
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    curr = start
    custo = 0

    for _ in range(n - 1):
        row = dist[curr]
        best_v = min((v for v in range(n) if not visited[v]), key=lambda v: row[v])
        visited[best_v] = True
        tour.append(best_v)
        custo += row[best_v]
        curr = best_v

    custo += dist[curr][start]
    tour.append(start)
    return custo, normalizar_rota(tour, n)


def _cross(o: Node, a: Node, b: Node) -> float:
    return (a.posX - o.posX) * (b.posY - o.posY) - (a.posY - o.posY) * (b.posX - o.posX)


def _fecho_convexo(vertices: Sequence[Node]) -> List[int]:
    """Fecho convexo 2D via Monotone Chain de Andrew em O(N log N)."""
    n = len(vertices)
    if n <= 3:
        return list(range(n))

    ordenados = sorted(range(n), key=lambda i: (vertices[i].posX, vertices[i].posY))

    lower: List[int] = []
    for i in ordenados:
        while len(lower) >= 2 and _cross(vertices[lower[-2]], vertices[lower[-1]], vertices[i]) <= 0:
            lower.pop()
        lower.append(i)

    upper: List[int] = []
    for i in reversed(ordenados):
        while len(upper) >= 2 and _cross(vertices[upper[-2]], vertices[upper[-1]], vertices[i]) <= 0:
            upper.pop()
        upper.append(i)

    hull = lower[:-1] + upper[:-1]
    return hull if len(hull) >= 3 else [ordenados[0], ordenados[-1]]


def warm_start_convex_hull(dist: List[List[int]], vertices: Sequence[Node]) -> Tuple[int, List[int]]:
    """
    3. Heuristica de Insercao Baseada em Fecho Convexo (Convex Hull Insertion).
    Inicia com o fecho convexo e insere iterativamente os pontos internos
    na posicao de menor aumento de custo: d(u, v) + d(v, w) - d(u, w).
    """
    n = len(dist)
    if len(vertices) != n:
        raise ValueError(f"Numero de vertices ({len(vertices)}) difere da matriz ({n}).")

    tour = _fecho_convexo(vertices)
    in_tour = set(tour)
    remaining = [v for v in range(n) if v not in in_tour]

    while remaining:
        best_v, best_pos, best_delta = -1, -1, math.inf
        m = len(tour)
        for v in remaining:
            row_v = dist[v]
            for pos in range(m):
                u = tour[pos]
                w = tour[(pos + 1) % m]
                delta = row_v[u] + row_v[w] - dist[u][w]
                if delta < best_delta:
                    best_delta, best_v, best_pos = delta, v, pos + 1
        tour.insert(best_pos, best_v)
        remaining.remove(best_v)

    tour_norm = normalizar_rota(tour, n)
    custo = sum(dist[tour_norm[i]][tour_norm[i + 1]] for i in range(n))
    return custo, tour_norm
