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


def _carregar_twice_around_the_tree():
    """O arquivo tem hifens no nome (nao importavel com `import`), entao carregamos via importlib."""
    import importlib.util
    import sys
    from pathlib import Path

    nome = "twice_around_the_tree"
    if nome in sys.modules:
        return sys.modules[nome]
    caminho = Path(__file__).resolve().parent / "twice-around-the-tree.py"
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


# Carregado na importacao (fora do cronometro do warm-start), para o tempo medido
# da estrategia nao incluir o custo de importar o SciPy.
_TAT = _carregar_twice_around_the_tree()


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
    if ws in ("christofides", "cristofides"):
        from algorithms.christofides import warm_start_christofides

        custo, tour = warm_start_christofides(dist)
        return int(custo), normalizar_rota(tour, n), "christofides"
    if ws in ("twice_around_the_tree", "twice-around-the-tree", "tat", "dobro_da_arvore"):
        custo, tour = _TAT.warm_start_twice_around_the_tree(dist)
        return int(custo), normalizar_rota(tour, n), "twice_around_the_tree"
    raise ValueError(
        f"Warm-start desconhecido: '{warm_start}'. Use 'cold_start', 'nearest_neighbor', "
        "'convex_hull', 'twice_around_the_tree', 'christofides', ou passe `initial_tour`/`warm_start_fn`."
    )


def branch_and_bound_tsp(
    matriz_distancias: Union[np.ndarray, Sequence[Sequence[int]]],
    vertices: Optional[Sequence[Node]] = None,
    warm_start: str = "nearest_neighbor",
    initial_tour: Optional[Sequence[int]] = None,
    initial_upper_bound: Optional[int] = None,
    warm_start_fn: Optional[Callable[[List[List[int]]], Tuple[int, List[int]]]] = None,
    time_limit: Optional[float] = 1800.0,
    break_symmetry: bool = True,
    use_mst_cache: bool = True,
    max_mst_cache_size: int = 65536,
    track_memory: bool = False,
) -> BranchAndBoundResult:
    """
    Branch-and-Bound exato para o TSP Simetrico (DFS por extensao de caminho + LB via MST).

    - Busca: DFS com backtracking in-place (sem copias de matrizes) e vizinhos
      visitados em ordem crescente de distancia.
    - Lower Bound admissivel: para caminho parcial (0 -> ... -> curr) de custo g
      e conjunto de vertices nao visitados U:
          LB = g + MST(U) + min_{u in U} d(curr, u) + min_{u in U} d(u, 0)
      (na raiz, onde curr == 0, usa as duas menores arestas distintas saindo de 0).
    - Otimizacoes enxutas:
      1) Quebra de simetria de reversao (path[1] < path[n-1]) em O(1) via bitmask;
      2) Cache limitado de MST(U) por bitmask (evita recalcular Prim em subproblemas repetidos);
      3) Prim com swap-remove sobre buffers pre-alocados e aborto antecipado;
      4) `track_memory=False` por padrao para nao contaminar medicoes de tempo.
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

    # Rank por proximidade a origem (alinha a quebra de simetria com a ordem gulosa da DFS)
    ordem_origem = vizinhos_ordenados[0]
    rank_origem = [0] * n
    bit_vertice = [0] * n
    for r, v in enumerate(ordem_origem):
        rank_origem[v] = r
        bit_vertice[v] = 1 << r

    # Estado unico mutavel (zero copias de matrizes)
    best_cost: float = initial_ub
    best_tour: List[int] = list(initial_tour_norm)
    visited = [False] * n
    visited[0] = True
    path = [0] * n

    # Buffers pre-alocados para o Prim com swap-remove + cache limitado de MST(U)
    unvisited_buf = [0] * n
    ativos = [0] * n
    chaves = [0] * n
    mst_cache: Dict[int, int] = {}

    nodes_expanded = 0
    nodes_pruned = 0
    max_depth = 0
    timed_out = False
    tempo_lb = 0.0

    def lower_bound_mst(curr: int, g: int, mask: int) -> int:
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
        base_lb = g + conn
        if base_lb >= best_cost:
            tempo_lb += time.perf_counter() - t0
            return base_lb

        # Consulta cache limitado de MST(U) pela bitmask dos nao visitados
        orcamento = best_cost - base_lb
        if use_mst_cache and mask in mst_cache:
            mst_cost = mst_cache[mask]
        else:
            # Prim denso O(k^2) com swap-remove e aborto antecipado
            m = k - 1
            row_start = dist[unvisited_buf[0]]
            for idx in range(m):
                u = unvisited_buf[idx + 1]
                ativos[idx] = u
                chaves[idx] = row_start[u]

            mst_cost = 0
            aborted = False
            while m > 0:
                best_i = 0
                best_d = chaves[0]
                for idx in range(1, m):
                    d = chaves[idx]
                    if d < best_d:
                        best_d = d
                        best_i = idx

                mst_cost += best_d
                if mst_cost >= orcamento:
                    aborted = True
                    break

                u = ativos[best_i]
                m -= 1
                ativos[best_i] = ativos[m]
                chaves[best_i] = chaves[m]

                row_u = dist[u]
                for idx in range(m):
                    d = row_u[ativos[idx]]
                    if d < chaves[idx]:
                        chaves[idx] = d

            # Armazena no cache somente custos exatos (nao abortados)
            if not aborted and use_mst_cache and max_mst_cache_size > 0:
                if len(mst_cache) >= max_mst_cache_size:
                    mst_cache.clear()
                mst_cache[mask] = mst_cost

        tempo_lb += time.perf_counter() - t0
        return base_lb + mst_cost

    def dfs(curr: int, depth: int, g: int, mask: int) -> None:
        nonlocal best_cost, best_tour, nodes_expanded, nodes_pruned, max_depth, timed_out

        if timed_out:
            return
        if deadline is not None and (nodes_expanded & 4095) == 0 and time.perf_counter() >= deadline:
            timed_out = True
            return

        if depth > max_depth:
            max_depth = depth

        if depth == n - 1:
            # Ultimo vertice direto via bitmask em O(1)
            ultimo = ordem_origem[mask.bit_length() - 1]
            total = g + dist[curr][ultimo] + dist[ultimo][0]
            nodes_expanded += 1
            if total < best_cost:
                best_cost = total
                path[depth] = ultimo
                best_tour = path + [0]
            else:
                nodes_pruned += 1
            return

        nodes_expanded += 1
        row_curr = dist[curr]

        for nxt in vizinhos_ordenados[curr]:
            if visited[nxt]:
                continue

            new_mask = mask ^ bit_vertice[nxt]

            # Quebra de simetria (rank_origem[path[1]] < rank_origem[path[n-1]]) em O(1):
            # preserva a orientacao cujo primeiro passo saindo de 0 e mais curto
            if break_symmetry:
                rank_v1 = rank_origem[nxt] if depth == 1 else rank_origem[path[1]]
                if (new_mask.bit_length() - 1) <= rank_v1:
                    nodes_pruned += 1
                    continue

            novo_g = g + row_curr[nxt]
            if novo_g >= best_cost:
                nodes_pruned += 1
                continue

            visited[nxt] = True
            path[depth] = nxt

            lb = lower_bound_mst(nxt, novo_g, new_mask)
            if lb >= best_cost:
                nodes_pruned += 1
            else:
                dfs(nxt, depth + 1, novo_g, new_mask)

            visited[nxt] = False
            if timed_out:
                return

    # 2. Busca Branch-and-Bound
    t_busca = time.perf_counter()
    mask_inicial = (1 << (n - 1)) - 1  # bits 0..n-2 ligados (um por vertice em ordem_origem)
    root_lb = lower_bound_mst(0, 0, mask_inicial)
    if root_lb >= best_cost:
        nodes_pruned += 1
    else:
        dfs(0, 1, 0, mask_inicial)
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
