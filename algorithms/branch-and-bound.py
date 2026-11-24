from __future__ import annotations

from dataclasses import dataclass, field
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from entities.node import Node
from utilities.warm_start import (
    calcular_custo_rota,
    normalizar_rota,
    para_lista_distancias,
    warm_start_cold_start,
    warm_start_convex_hull,
    warm_start_nearest_neighbor,
)


@dataclass
class BranchAndBoundResult:
    """Metricas coletadas na execucao do Branch-and-Bound."""

    best_cost: float
    best_tour: List[int]
    optimal: bool
    status: str  # "OPTIMAL" ou "NA" (quando excede o time_limit)
    initial_upper_bound: float
    initial_tour: List[int]
    root_lower_bound: int
    warm_start_strategy: str
    nodes_expanded: int
    nodes_pruned: int
    max_depth: int
    elapsed_time: float
    search_time: float
    subroutine_times: Dict[str, float] = field(default_factory=dict)
    peak_memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        """Exporta as metricas para dicionario plano (pronto para Pandas/CSV)."""
        row: Dict[str, Union[int, float, str, bool]] = {
            "status": self.status,
            "optimal": self.optimal,
            "best_cost": self.best_cost if self.optimal else "NA",
            "incumbent_cost": self.best_cost,
            "initial_upper_bound": self.initial_upper_bound,
            "root_lower_bound": self.root_lower_bound,
            "warm_start_strategy": self.warm_start_strategy,
            "nodes_expanded": self.nodes_expanded,
            "nodes_pruned": self.nodes_pruned,
            "max_depth": self.max_depth,
            "elapsed_time": self.elapsed_time,
            "search_time": self.search_time,
            "peak_memory_bytes": self.peak_memory_bytes,
        }
        for nome, tempo in self.subroutine_times.items():
            row[f"time_{nome}"] = tempo
        return row


def _executar_warm_start(
    dist: List[List[int]],
    vertices: Optional[Sequence[Node]],
    warm_start: str,
    initial_tour: Optional[Sequence[int]],
    initial_upper_bound: Optional[int],
    warm_start_fn: Optional[Callable[[List[List[int]]], Tuple[int, List[int]]]],
) -> Tuple[int, List[int], str]:
    n = len(dist)
    if warm_start_fn is not None:
        custo, tour = warm_start_fn(dist)
        return int(custo), normalizar_rota(tour, n), warm_start
    if initial_tour is not None:
        tour_norm = normalizar_rota(initial_tour, n)
        custo = int(initial_upper_bound) if initial_upper_bound is not None else calcular_custo_rota(tour_norm, dist)
        return custo, tour_norm, warm_start
    if initial_upper_bound is not None:
        return int(initial_upper_bound), [], warm_start

    ws = warm_start.lower()
    if ws in ("cold_start", "trivial"):
        custo, tour = warm_start_cold_start(dist)
        return custo, tour, "cold_start"
    if ws in ("nearest_neighbor", "greedy", "busca_gulosa"):
        custo, tour = warm_start_nearest_neighbor(dist)
        return custo, tour, "nearest_neighbor"
    if ws in ("convex_hull", "fecho_convexo"):
        if vertices is None:
            raise ValueError("A estrategia 'convex_hull' exige o parametro `vertices`.")
        custo, tour = warm_start_convex_hull(dist, vertices)
        return custo, tour, "convex_hull"
    raise ValueError(
        f"Warm-start desconhecido: '{warm_start}'. Use 'cold_start', 'nearest_neighbor', "
        "'convex_hull', ou passe `initial_tour`/`warm_start_fn`."
    )


def branch_and_bound_tsp(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]],
    vertices: Optional[Sequence[Node]] = None,
    warm_start: str = "nearest_neighbor",
    initial_tour: Optional[Sequence[int]] = None,
    initial_upper_bound: Optional[int] = None,
    warm_start_fn: Optional[Callable[[List[List[int]]], Tuple[int, List[int]]]] = None,
    time_limit: Optional[float] = 1800.0,
    track_memory: bool = True,
) -> BranchAndBoundResult:
    """
    Branch-and-Bound exato para o TSP Simetrico (DFS por extensao de caminho + LB via MST).

    - Estrategia de busca: Depth-First Search (DFS) visitando vizinhos em ordem
      crescente de distancia. Usa memoria O(N) fixa (sem copias de matrizes).
    - Lower Bound admissivel: para um caminho parcial (0 -> ... -> curr) de custo g
      e conjunto de vertices nao visitados U:
          LB = g + MST(U) + min_{u in U} d(curr, u) + min_{u in U} d(u, 0)
      (na raiz, onde curr == 0, usa as duas menores arestas distintas saindo de 0).
    """
    ja_rastreando = tracemalloc.is_tracing()
    if track_memory and not ja_rastreando:
        tracemalloc.start()

    t_inicio = time.perf_counter()
    deadline = t_inicio + time_limit if time_limit is not None else None
    dist = para_lista_distancias(matriz_distancias)
    n = len(dist)

    # Casos triviais
    if n <= 3:
        tour = [0] if n == 1 else (list(range(n)) + [0] if n >= 2 else [])
        custo = calcular_custo_rota(tour, dist)
        peak = tracemalloc.get_traced_memory()[1] if (track_memory and tracemalloc.is_tracing()) else 0
        if track_memory and not ja_rastreando and tracemalloc.is_tracing():
            tracemalloc.stop()
        return BranchAndBoundResult(
            best_cost=custo, best_tour=tour, optimal=True, status="OPTIMAL",
            initial_upper_bound=custo, initial_tour=tour, root_lower_bound=custo,
            warm_start_strategy=warm_start, nodes_expanded=1, nodes_pruned=0,
            max_depth=max(0, n - 1), elapsed_time=time.perf_counter() - t_inicio,
            search_time=0.0, subroutine_times={"warm_start": 0.0, "lower_bound_mst": 0.0},
            peak_memory_bytes=peak,
        )

    # 1. Warm-start (Upper Bound inicial)
    t_ws = time.perf_counter()
    initial_ub, initial_tour_norm, ws_label = _executar_warm_start(
        dist, vertices, warm_start, initial_tour, initial_upper_bound, warm_start_fn
    )
    tempo_warm_start = time.perf_counter() - t_ws

    # Ordena vizinhos por distancia crescente uma unica vez para guiar a DFS
    vizinhos_ordenados = [
        sorted((v for v in range(1, n) if v != u), key=lambda v, row=dist[u]: row[v])
        for u in range(n)
    ]

    # Estado unico mutavel (memoria O(N), zero copias de matrizes)
    best_cost: float = initial_ub
    best_tour: List[int] = list(initial_tour_norm)
    visited = [False] * n
    visited[0] = True
    path = [0] * n

    # Buffers pre-alocados para Prim (evita realocacoes e pressao no GC)
    in_mst = [False] * n
    min_edge = [0] * n
    unvisited_buf = [0] * n

    nodes_expanded = 0
    nodes_pruned = 0
    max_depth = 0
    timed_out = False
    tempo_lb = 0.0

    def lower_bound_mst(curr: int, g: int) -> int:
        nonlocal tempo_lb
        t0 = time.perf_counter()

        # Coleta os vertices nao visitados no buffer pre-alocado
        k = 0
        for v in range(1, n):
            if not visited[v]:
                unvisited_buf[k] = v
                k += 1

        if k == 0:
            tempo_lb += time.perf_counter() - t0
            return g + dist[curr][0]
        if k == 1:
            u = unvisited_buf[0]
            tempo_lb += time.perf_counter() - t0
            return g + dist[curr][u] + dist[u][0]

        # Conexoes de entrada (curr -> U) e retorno (U -> 0)
        row_curr = dist[curr]
        row_0 = dist[0]
        if curr == 0:
            # Na raiz, primeira e ultima arestas precisam ir para vertices distintos
            m1 = m2 = 10**18
            for idx in range(k):
                d = row_0[unvisited_buf[idx]]
                if d < m1:
                    m2, m1 = m1, d
                elif d < m2:
                    m2 = d
            conn = m1 + m2
        else:
            min_from_curr = 10**18
            min_to_0 = 10**18
            for idx in range(k):
                u = unvisited_buf[idx]
                dc = row_curr[u]
                if dc < min_from_curr:
                    min_from_curr = dc
                d0 = row_0[u]
                if d0 < min_to_0:
                    min_to_0 = d0
            conn = min_from_curr + min_to_0

        # Corte rapido O(k) antes de rodar Prim O(k^2)
        if g + conn >= best_cost:
            tempo_lb += time.perf_counter() - t0
            return g + conn

        # MST sobre U usando Prim O(k^2) sem alocacoes dinamicas
        start_u = unvisited_buf[0]
        row_start = dist[start_u]
        in_mst[start_u] = True
        for idx in range(1, k):
            u = unvisited_buf[idx]
            in_mst[u] = False
            min_edge[u] = row_start[u]

        mst_cost = 0
        for _ in range(k - 1):
            best_u = -1
            best_d = 10**18
            for idx in range(1, k):
                u = unvisited_buf[idx]
                if not in_mst[u] and min_edge[u] < best_d:
                    best_d = min_edge[u]
                    best_u = u
            in_mst[best_u] = True
            mst_cost += best_d
            if g + conn + mst_cost >= best_cost:
                tempo_lb += time.perf_counter() - t0
                return g + conn + mst_cost

            row_bu = dist[best_u]
            for idx in range(1, k):
                u = unvisited_buf[idx]
                if not in_mst[u]:
                    d = row_bu[u]
                    if d < min_edge[u]:
                        min_edge[u] = d

        tempo_lb += time.perf_counter() - t0
        return g + conn + mst_cost

    def dfs(curr: int, depth: int, g: int) -> None:
        nonlocal best_cost, best_tour, nodes_expanded, nodes_pruned, max_depth, timed_out

        if timed_out:
            return
        if deadline is not None and (nodes_expanded & 255) == 0 and time.perf_counter() >= deadline:
            timed_out = True
            return

        if depth > max_depth:
            max_depth = depth

        if depth == n - 1:
            # Ultimo vertice: fecha o ciclo de volta a 0
            for v in vizinhos_ordenados[curr]:
                if not visited[v]:
                    total = g + dist[curr][v] + dist[v][0]
                    nodes_expanded += 1
                    if total < best_cost:
                        best_cost = total
                        path[depth] = v
                        best_tour = path + [0]
                    else:
                        nodes_pruned += 1
                    return
            return

        nodes_expanded += 1
        row_curr = dist[curr]

        for nxt in vizinhos_ordenados[curr]:
            if visited[nxt]:
                continue
            novo_g = g + row_curr[nxt]
            if novo_g >= best_cost:
                nodes_pruned += 1
                continue

            visited[nxt] = True
            path[depth] = nxt

            lb = lower_bound_mst(nxt, novo_g)
            if lb >= best_cost:
                nodes_pruned += 1
            else:
                dfs(nxt, depth + 1, novo_g)

            visited[nxt] = False
            if timed_out:
                return

    # 2. Busca Branch-and-Bound
    t_busca = time.perf_counter()
    root_lb = lower_bound_mst(0, 0)
    if root_lb >= best_cost:
        nodes_pruned += 1
    else:
        dfs(0, 1, 0)
    agora = time.perf_counter()

    peak_mem = 0
    if track_memory and tracemalloc.is_tracing():
        _, peak_mem = tracemalloc.get_traced_memory()
        if not ja_rastreando:
            tracemalloc.stop()

    optimal = not timed_out
    search_time = agora - t_busca
    return BranchAndBoundResult(
        best_cost=best_cost,
        best_tour=best_tour,
        optimal=optimal,
        status="OPTIMAL" if optimal else "NA",
        initial_upper_bound=initial_ub,
        initial_tour=initial_tour_norm,
        root_lower_bound=root_lb,
        warm_start_strategy=ws_label,
        nodes_expanded=nodes_expanded,
        nodes_pruned=nodes_pruned,
        max_depth=max_depth,
        elapsed_time=agora - t_inicio,
        search_time=search_time,
        subroutine_times={
            "warm_start": tempo_warm_start,
            "lower_bound_mst": tempo_lb,
            "search_overhead": max(0.0, search_time - tempo_lb),
        },
        peak_memory_bytes=peak_mem,
    )
