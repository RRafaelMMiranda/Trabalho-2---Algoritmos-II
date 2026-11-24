from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import statistics
import sys
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from utilities.tsplib_reader import carregar_instancia_tsplib
from utilities.warm_start import calcular_custo_rota

PASTA_INSTANCIAS = RAIZ_PROJETO / "data" / "tsplib"
CSV_PADRAO = RAIZ_PROJETO / "results" / "twice" / "metricas_twice_around_the_tree.csv"

INSTANCIAS_ALVO = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22",
    "gr24", "fri26", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70",
]

# Custo do tour otimo conhecido de cada instancia (TSPLIB). Usado so para o gap;
# conferir contra a pagina da TSPLIB antes de citar no relatorio.
OTIMOS_TSPLIB = {
    "burma14": 3323, "ulysses16": 6859, "gr17": 2085, "gr21": 2707,
    "ulysses22": 7013, "gr24": 1272, "fri26": 937, "bays29": 2020,
    "dantzig42": 699, "swiss42": 1273, "att48": 10628, "eil51": 426,
    "berlin52": 7542, "st70": 675,
}

SUBROTINAS = ("mst", "preordem", "custo", "total")

COLUNAS_CSV = [
    "instancia", "n", "tipo_metrica", "custo", "otimo_conhecido", "gap_pct", "valido",
    "repeticoes",
    "tempo_mst_ms", "tempo_preordem_ms", "tempo_custo_ms", "tempo_total_ms",
    "mem_pico_mst_kb", "mem_pico_preordem_kb", "mem_pico_custo_kb", "mem_pico_total_kb",
]


def _carregar_modulo_tat():
    """O arquivo tem hifens no nome (nao importavel com `import`), entao carregamos via importlib."""
    caminho = RAIZ_PROJETO / "algorithms" / "twice-around-the-tree.py"
    spec = importlib.util.spec_from_file_location("twice_around_the_tree", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["twice_around_the_tree"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def validar_ciclo_hamiltoniano(tour: List[int], n: int, dist: np.ndarray, custo_reportado: int) -> bool:
    """Garante que a solucao e um ciclo Hamiltoniano estrito e valida o custo."""
    if len(tour) != n + 1:
        return False
    if tour[0] != 0 or tour[-1] != 0:
        return False
    if len(set(tour[:-1])) != n:
        return False
    return calcular_custo_rota(tour, dist) == custo_reportado


def pico_memoria_kb(func: Callable, *args: Any) -> Tuple[Any, float]:
    """Executa `func` sob tracemalloc e devolve (resultado, pico de memoria em KB).

    Uma sessao de tracemalloc por chamada (start/stop sem aninhamento), e sempre em passada
    separada da medicao de tempo, porque o tracemalloc deixa o codigo mais lento.
    """
    gc.collect()
    tracemalloc.start()
    try:
        resultado = func(*args)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return resultado, pico / 1024.0


def medir_instancia(tat, matriz: np.ndarray, repeticoes: int) -> Dict[str, Any]:
    """Mede tempo (mediana de `repeticoes` execucoes) e pico de memoria de cada subrotina."""
    # Aquecimento: absorve inicializacoes preguicosas (imports internos do SciPy, caches)
    tat.twice_around_the_tree(matriz)

    # --- Passada 1: TEMPO (sem tracemalloc) ---
    amostras_ms: Dict[str, List[float]] = {k: [] for k in SUBROTINAS}
    resultado: Dict[str, Any] = {}
    for _ in range(repeticoes):
        resultado = tat.twice_around_the_tree(matriz)
        for k in SUBROTINAS:
            amostras_ms[k].append(resultado["tempos"][k] * 1000.0)
    tempos_ms = {k: statistics.median(v) for k, v in amostras_ms.items()}

    # --- Passada 2: MEMORIA (com tracemalloc, cada subrotina isolada) ---
    filhos, mem_mst = pico_memoria_kb(tat.mst_prim, matriz, 0)
    ordem, mem_pre = pico_memoria_kb(tat.preordem, filhos, 0)
    _, mem_custo = pico_memoria_kb(tat.custo_ciclo, ordem + [0], matriz)
    _, mem_total = pico_memoria_kb(tat.twice_around_the_tree, matriz)

    return {
        "ciclo": resultado["ciclo"],
        "custo": resultado["custo"],
        "tempos_ms": tempos_ms,
        "mem_kb": {"mst": mem_mst, "preordem": mem_pre, "custo": mem_custo, "total": mem_total},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bateria de experimentos do Twice-around-the-tree (tempo e memoria por subrotina)."
    )
    parser.add_argument(
        "--instancias", nargs="+", default=INSTANCIAS_ALVO,
        help="Instancias TSPLIB (sem .tsp). Padrao: as 14 instancias.",
    )
    parser.add_argument(
        "--repeticoes", type=int, default=20,
        help="Execucoes por instancia para a mediana do tempo (padrao: 20).",
    )
    parser.add_argument(
        "--output", type=Path, default=CSV_PADRAO,
        help=f"CSV de saida (padrao: {CSV_PADRAO.relative_to(RAIZ_PROJETO)}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tat = _carregar_modulo_tat()

    print("=" * 104)
    print("BATERIA DE TESTES - TWICE-AROUND-THE-TREE (DCC207 UFMG)")
    print(f"Repeticoes por instancia: {args.repeticoes} (mediana do tempo)")
    print("=" * 104)
    print(
        f"{'Instancia':<11} | {'N':>3} | {'Custo':>7} | {'Gap(%)':>7} | {'MST(ms)':>8} | "
        f"{'Pre(ms)':>8} | {'Total(ms)':>9} | {'MST(KB)':>8} | {'Pico(KB)':>8} | Status"
    )
    print("-" * 104)

    linhas: List[Dict[str, Any]] = []
    for nome in args.instancias:
        caminho = PASTA_INSTANCIAS / f"{nome}.tsp"
        if not caminho.exists():
            print(f"{nome:<11} | [ARQUIVO NAO ENCONTRADO EM {caminho}]")
            continue
        try:
            matriz, n, _, metadados = carregar_instancia_tsplib(caminho)
            matriz = np.asarray(matriz)

            m = medir_instancia(tat, matriz, args.repeticoes)
            valido = validar_ciclo_hamiltoniano(m["ciclo"], n, matriz, m["custo"])

            otimo = OTIMOS_TSPLIB.get(nome)
            gap_pct = round(100.0 * (m["custo"] - otimo) / otimo, 4) if otimo else "NA"
            t, mem = m["tempos_ms"], m["mem_kb"]

            print(
                f"{nome:<11} | {n:>3} | {m['custo']:>7} | "
                f"{(f'{gap_pct:.2f}' if otimo else 'NA'):>7} | {t['mst']:>8.3f} | "
                f"{t['preordem']:>8.3f} | {t['total']:>9.3f} | {mem['mst']:>8.1f} | "
                f"{mem['total']:>8.1f} | {'OK' if valido else 'FALHA'}"
            )

            linhas.append({
                "instancia": nome,
                "n": n,
                "tipo_metrica": metadados.get("edge_weight_type", "EUC_2D"),
                "custo": m["custo"],
                "otimo_conhecido": otimo if otimo else "NA",
                "gap_pct": gap_pct,
                "valido": valido,
                "repeticoes": args.repeticoes,
                "tempo_mst_ms": t["mst"],
                "tempo_preordem_ms": t["preordem"],
                "tempo_custo_ms": t["custo"],
                "tempo_total_ms": t["total"],
                "mem_pico_mst_kb": mem["mst"],
                "mem_pico_preordem_kb": mem["preordem"],
                "mem_pico_custo_kb": mem["custo"],
                "mem_pico_total_kb": mem["total"],
            })
        except Exception as e:  # uma instancia com erro nao derruba a bateria
            print(f"{nome:<11} | [ERRO NA EXECUCAO: {e}]")

    print("=" * 104)

    if linhas:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
            writer.writeheader()
            writer.writerows(linhas)
        print(f"\n[SUCESSO] Base de metricas salva em '{args.output}'.")


if __name__ == "__main__":
    main()
