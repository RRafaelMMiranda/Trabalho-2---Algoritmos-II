from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from algorithms.christofides import executar_christofides_com_auditoria
from utilities.tsplib_reader import carregar_instancia_tsplib
from utilities.warm_start import calcular_custo_rota

PASTA_INSTANCIAS = Path("./data/tsplib")

INSTANCIAS_ALVO = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22",
    "gr24", "fri26", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70"
]

def validar_ciclo_hamiltoniano(tour: List[int], n: int, dist: np.ndarray, custo_reportado: int) -> bool:
    """Garante que a solucao e um ciclo Hamiltoniano estrito e valida o custo."""
    if len(tour) != n + 1:
        return False
    if tour[0] != 0 or tour[-1] != 0:
        return False
    if len(set(tour[:-1])) != n:
        return False
    custo_calculado = calcular_custo_rota(tour, dist)
    return custo_calculado == custo_reportado

def main():
    print("=" * 90)
    print("BATERIA DE TESTES - ALGORITMO DE CHRISTOFIDES (DCC207 UFMG)")
    print("=" * 90)
    print(f"{'Instância':<12} | {'N':<4} | {'|O|':<4} | {'Custo':<8} | {'Puro(ms)':<10} | {'MST(ms)':<8} | {'Match(ms)':<9} | {'Pico(KB)':<8} | {'Status'}")
    print("-" * 90)

    resultados = []

    for nome in INSTANCIAS_ALVO:
        caminho = PASTA_INSTANCIAS / f"{nome}.tsp"
        if not caminho.exists():
            print(f"{nome:<12} | [ARQUIVO NÃO ENCONTRADO EM {caminho}]")
            continue

        try:
            # 1. Leitura centralizada via modulo utilitario
            matriz, n, _, metadados = carregar_instancia_tsplib(caminho)

            # 2. Execucao de Christofides com profiling das sub-rotinas
            res = executar_christofides_com_auditoria(matriz)

            # 3. Validacao formal
            valido = validar_ciclo_hamiltoniano(res.tour, n, matriz, res.custo)
            status = "OK" if valido else "FALHA"

            # Tempo puro = soma das sub-rotinas (sem o overhead do gc.collect)
            tempo_puro_ms = (
                res.mst.tempo_ms +
                res.matching.tempo_ms +
                res.euleriano.tempo_ms +
                res.shortcutting.tempo_ms
            )

            print(
                f"{nome:<12} | {n:<4} | {res.qtd_vertices_impares:<4} | {res.custo:<8} | "
                f"{tempo_puro_ms:<10.2f} | {res.mst.tempo_ms:<8.2f} | "
                f"{res.matching.tempo_ms:<9.2f} | {res.memoria_pico_total_kb:<8.1f} | {status}"
            )

            d = res.para_dict()
            d["instancia"] = nome
            d["tempo_algoritmo_puro_ms"] = tempo_puro_ms
            d["tipo_metrica"] = metadados.get("edge_weight_type", "EUC_2D")
            resultados.append(d)

        except Exception as e:
            print(f"{nome:<12} | [ERRO NA EXECUÇÃO: {e}]")

    print("=" * 90)

    # Exporta para CSV
    if resultados:
        caminho_csv = Path("metricas_christofides.csv")
        campos = list(resultados[0].keys())
        with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\n[SUCESSO] Base de metricas salva em '{caminho_csv}'.")

if __name__ == "__main__":
    main()