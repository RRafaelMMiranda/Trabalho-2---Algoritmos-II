from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from utilities.tsplib_reader import carregar_instancia_tsplib
from utilities.warm_start import calcular_custo_rota

PASTA_INSTANCIAS = RAIZ_PROJETO / "data" / "tsplib"
CSV_PADRAO = RAIZ_PROJETO / "results" / "metricas_branch_and_bound.csv"

INSTANCIAS_ALVO = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22",
    "gr24", "fri26", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70",
]

ESTRATEGIAS_PADRAO = [
    "cold_start",
    "nearest_neighbor",
    "convex_hull",
    "twice_around_the_tree",
    "christofides",
]

COLUNAS_CSV = [
    "instancia",
    "n",
    "tipo_metrica",
    "warm_start_strategy",
    "status",
    "optimal",
    "valido",
    "best_cost",
    "incumbent_cost",
    "initial_upper_bound",
    "root_lower_bound",
    "gap_inicial_pct",
    "nodes_expanded",
    "nodes_pruned",
    "max_depth",
    "elapsed_time",
    "search_time",
    "time_warm_start",
    "time_lower_bound_mst",
    "time_search_overhead",
    "peak_memory_bytes",
]


def _carregar_modulo_bnb():
    caminho = RAIZ_PROJETO / "algorithms" / "branch-and-bound.py"
    spec = importlib.util.spec_from_file_location("branch_and_bound", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["branch_and_bound"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def validar_ciclo_hamiltoniano(tour: List[int], n: int, dist: np.ndarray, custo_reportado: float) -> bool:
    """Garante que a solucao e um ciclo Hamiltoniano valido com custo consistente."""
    if not tour or len(tour) != n + 1:
        return False
    if tour[0] != 0 or tour[-1] != 0:
        return False
    if len(set(tour[:-1])) != n:
        return False
    return calcular_custo_rota(tour, dist) == int(custo_reportado)


def carregar_ja_executados(caminho_csv: Path) -> Set[Tuple[str, str]]:
    """Retorna pares (instancia, warm_start_strategy) ja gravados para suportar --resume."""
    feitos: Set[Tuple[str, str]] = set()
    if not caminho_csv.exists():
        return feitos
    with open(caminho_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row.get("instancia")
            ws = row.get("warm_start_strategy")
            if inst and ws:
                feitos.add((inst, ws))
    return feitos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bateria de experimentos do Branch-and-Bound com os 5 warm-starts."
    )
    parser.add_argument(
        "--instancias",
        nargs="+",
        default=INSTANCIAS_ALVO,
        help="Lista de instancias TSPLIB (sem .tsp). Padrao: todas as 14 instancias.",
    )
    parser.add_argument(
        "--estrategias",
        nargs="+",
        choices=ESTRATEGIAS_PADRAO,
        default=ESTRATEGIAS_PADRAO,
        help="Estrategias de warm-start a executar. Padrao: todas as 5.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1800.0,
        help="Limite de tempo por execucao em segundos (padrao: 1800s = 30 min).",
    )
    parser.add_argument(
        "--medir-memoria",
        action="store_true",
        help="Ativa tracemalloc para medir pico de memoria (desligado por padrao para nao distorcer tempo).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CSV_PADRAO,
        help=f"Caminho do CSV de saida (padrao: {CSV_PADRAO.relative_to(RAIZ_PROJETO)}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula combinacoes (instancia, estrategia) ja presentes no CSV de saida.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bb = _carregar_modulo_bnb()

    caminho_csv: Path = args.output
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    ja_feitos = carregar_ja_executados(caminho_csv) if args.resume else set()
    modo_arquivo = "a" if (args.resume and caminho_csv.exists()) else "w"

    print("=" * 114)
    print("BATERIA DE TESTES - BRANCH-AND-BOUND (DCC207 UFMG)")
    print(f"Estrategias: {', '.join(args.estrategias)} | Time limit: {args.time_limit:.0f}s | Memoria: {args.medir_memoria}")
    print("=" * 114)
    print(
        f"{'Instancia':<11} | {'N':>3} | {'WarmStart':<17} | {'Status':<7} | "
        f"{'UB0':>7} | {'OPT/Inc':>7} | {'Exp':>9} | {'Prun':>10} | {'t_ws(ms)':>8} | {'Tempo(s)':>9}"
    )
    print("-" * 114)

    with open(caminho_csv, modo_arquivo, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if modo_arquivo == "w":
            writer.writeheader()
            f.flush()

        for nome in args.instancias:
            caminho_tsp = PASTA_INSTANCIAS / f"{nome}.tsp"
            if not caminho_tsp.exists():
                print(f"{nome:<11} | [ARQUIVO NAO ENCONTRADO EM {caminho_tsp}]")
                continue

            matriz, n, nodes, metadados = carregar_instancia_tsplib(caminho_tsp)
            tipo_metrica = metadados.get("edge_weight_type", "EUC_2D")

            for ws in args.estrategias:
                if (nome, ws) in ja_feitos:
                    print(f"{nome:<11} | {n:>3} | {ws:<17} | [PULADO --resume]")
                    continue

                if ws == "convex_hull" and nodes is None:
                    print(f"{nome:<11} | {n:>3} | {ws:<17} | SKIP (instancia {tipo_metrica} sem coordenadas 2D)")
                    continue

                res = bb.branch_and_bound_tsp(
                    matriz_distancias=matriz,
                    vertices=nodes,
                    warm_start=ws,
                    time_limit=args.time_limit,
                    track_memory=args.medir_memoria,
                )

                valido = validar_ciclo_hamiltoniano(res.best_tour, n, matriz, res.best_cost)
                gap_pct = (
                    100.0 * (res.initial_upper_bound - res.best_cost) / res.best_cost
                    if res.optimal and res.best_cost > 0
                    else "NA"
                )
                t_ws_ms = res.subroutine_times.get("warm_start", 0.0) * 1000.0

                print(
                    f"{nome:<11} | {n:>3} | {res.warm_start_strategy:<17} | {res.status:<7} | "
                    f"{int(res.initial_upper_bound):>7} | {int(res.best_cost):>7} | "
                    f"{res.nodes_expanded:>9} | {res.nodes_pruned:>10} | "
                    f"{t_ws_ms:>8.2f} | {res.elapsed_time:>9.4f}"
                )

                linha: Dict[str, object] = res.to_dict()
                linha.update({
                    "instancia": nome,
                    "n": n,
                    "tipo_metrica": tipo_metrica,
                    "valido": valido,
                    "gap_inicial_pct": round(gap_pct, 4) if isinstance(gap_pct, float) else gap_pct,
                })
                writer.writerow({col: linha.get(col, "") for col in COLUNAS_CSV})
                f.flush()

    print("=" * 114)
    print(f"\n[SUCESSO] Metricas salvas em '{caminho_csv}'.")


if __name__ == "__main__":
    main()
