from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# 1. FUNÇÕES DOS MODELOS TEÓRICOS ASSINTÓTICOS
# ============================================================================

def modelo_quadratico(n: np.ndarray, a: float, c: float) -> np.ndarray:
    """Modelo teórico para MST: T(N) = a * N^2 + c (c é o overhead inicial fixo)."""
    return a * (n ** 2) + c

def modelo_cubico(n: np.ndarray, a: float, c: float) -> np.ndarray:
    """Modelo teórico para Matching / Christofides: T(N) = a * N^3 + c."""
    return a * (n ** 3) + c

def modelo_polinomial_completo(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Modelo cúbico com termo quadrático para Tempo Total: T(N) = a*N^3 + b*N^2 + c."""
    return a * (n ** 3) + b * (n ** 2) + c

# ============================================================================
# 2. CARREGAMENTO E NORMALIZAÇÃO DOS DADOS
# ============================================================================

def carregar_dados() -> pd.DataFrame:
    caminhos = [
        Path("metricas_christofides.csv"),
        Path("resultados_christofides.csv"),
    ]
    caminho_csv = next((p for p in caminhos if p.exists()), None)

    if caminho_csv is None:
        raise FileNotFoundError(
            "Nenhum arquivo CSV encontrado. Certifique-se de que 'metricas_christofides.csv' "
            "ou 'resultados_christofides.csv' está no diretório atual."
        )

    df = pd.read_csv(caminho_csv)

    # Padronização de nomes de colunas caso haja pequenas variações
    mapa_colunas = {
        "N": "n",
        "tempo_total": "tempo_total_ms",
        "tempo_mst": "tempo_mst_ms",
        "tempo_matching": "tempo_matching_ms",
    }
    df.rename(columns=mapa_colunas, inplace=True)
    df.sort_values(by="n", inplace=True)
    return df

# ============================================================================
# 3. AJUSTE DE CURVAS E CÁLCULO DE MÉTRICAS (R² E RMSE)
# ============================================================================

def calcular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o RMSE compatível com versões antigas e novas do scikit-learn."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ajustar_modelo(
    x: np.ndarray,
    y: np.ndarray,
    func_modelo,
    p0: Tuple = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Ajusta a função teórica aos dados e extrai parâmetros, R² e RMSE."""
    popt, _ = curve_fit(func_modelo, x, y, p0=p0, maxfev=10000)
    y_pred = func_modelo(x, *popt)
    r2 = float(r2_score(y, y_pred))
    rmse = calcular_rmse(y, y_pred)
    return popt, y_pred, r2, rmse

# ============================================================================
# 4. GERAÇÃO DE GRÁFICO (PADRÃO ARTIGO CIENTÍFICO SBC)
# ============================================================================

def configurar_estilo_cientifico():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })

def plotar_resultados(df: pd.DataFrame, resultados: Dict[str, dict]):
    configurar_estilo_cientifico()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle(
        "Modelagem de Complexidade Empírica - Algoritmo de Christofides (TSPLIB)",
        fontweight="bold",
        y=1.02
    )

    n_continuo = np.linspace(df["n"].min(), df["n"].max(), 200)

    # --- Painel 1: MST (Quadrático O(N^2)) ---
    ax = axes[0]
    res = resultados["mst"]
    y_curve = modelo_quadratico(n_continuo, *res["popt"])
    
    ax.scatter(df["n"], df["tempo_mst_ms"], color="#1f77b4", label="Dados Empíricos", edgecolors="k", zorder=4)
    ax.plot(
        n_continuo, y_curve, color="#0b4f8a", linestyle="-",
        label=f"Ajuste: $aN^2 + c$\n$R^2 = {res['r2']:.4f}$\n$RMSE = {res['rmse']:.3f}$ ms"
    )
    ax.set_title("Árvore Geradora Mínima (MST)")
    ax.set_xlabel("Número de Vértices ($N$)")
    ax.set_ylabel("Tempo de Execução (ms)")
    ax.legend(loc="upper left")

    # --- Painel 2: Matching (Cúbico O(N^3)) ---
    ax = axes[1]
    res = resultados["matching"]
    y_curve = modelo_cubico(n_continuo, *res["popt"])

    ax.scatter(df["n"], df["tempo_matching_ms"], color="#d62728", label="Dados Empíricos", edgecolors="k", zorder=4)
    ax.plot(
        n_continuo, y_curve, color="#8b0000", linestyle="-",
        label=f"Ajuste: $aN^3 + c$\n$R^2 = {res['r2']:.4f}$\n$RMSE = {res['rmse']:.3f}$ ms"
    )
    ax.set_title("Emparelhamento Perfeito (Matching)")
    ax.set_xlabel("Número de Vértices ($N$)")
    ax.set_ylabel("Tempo de Execução (ms)")
    ax.legend(loc="upper left")

    # --- Painel 3: Tempo Total (Dominado por O(N^3)) ---
    ax = axes[2]
    res = resultados["total"]
    y_curve = modelo_cubico(n_continuo, *res["popt"])

    ax.scatter(df["n"], df["tempo_total_ms"], color="#2ca02c", label="Dados Empíricos", edgecolors="k", zorder=4)
    ax.plot(
        n_continuo, y_curve, color="#135a13", linestyle="-",
        label=f"Ajuste: $aN^3 + c$\n$R^2 = {res['r2']:.4f}$\n$RMSE = {res['rmse']:.3f}$ ms"
    )
    ax.set_title("Tempo Total (Christofides)")
    ax.set_xlabel("Número de Vértices ($N$)")
    ax.set_ylabel("Tempo de Execução (ms)")
    ax.legend(loc="upper left")

    plt.tight_layout()

    # Salva em PNG de alta definição e em PDF vetorial para o LaTeX
    caminho_png = Path("complexidade_empirica_christofides.png")
    caminho_pdf = Path("complexidade_empirica_christofides.pdf")

    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[SUCESSO] Gráficos exportados:")
    print(f" -> Imagem PNG (300 DPI): {caminho_png}")
    print(f" -> Gráfico Vetorial PDF (padrão SBC): {caminho_pdf}")

# ============================================================================
# 5. EXECUÇÃO PRINCIPAL E EXIBIÇÃO DE MÉTRICAS
# ============================================================================

def main():
    df = carregar_dados()
    x = df["n"].to_numpy(dtype=float)

    resultados = {}

    # 1. Ajuste MST (O(N^2))
    popt_mst, _, r2_mst, rmse_mst = ajustar_modelo(x, df["tempo_mst_ms"].to_numpy(), modelo_quadratico)
    resultados["mst"] = {"popt": popt_mst, "r2": r2_mst, "rmse": rmse_mst}

    # 2. Ajuste Matching (O(N^3))
    popt_match, _, r2_match, rmse_match = ajustar_modelo(x, df["tempo_matching_ms"].to_numpy(), modelo_cubico)
    resultados["matching"] = {"popt": popt_match, "r2": r2_match, "rmse": rmse_match}

    # 3. Ajuste Tempo Total (O(N^3))
    popt_total, _, r2_total, rmse_total = ajustar_modelo(x, df["tempo_total_ms"].to_numpy(), modelo_cubico)
    resultados["total"] = {"popt": popt_total, "r2": r2_total, "rmse": rmse_total}

    # Exibição no console
    print("=" * 80)
    print("RESULTADOS DA REGRESSÃO NÃO LINEAR (MODELAGEM ASSINTÓTICA)")
    print("=" * 80)
    print(f"{'Subrotina / Componente':<25} | {'Modelo Teórico':<15} | {'R²':<10} | {'RMSE (ms)':<10} | {'Equação Ajustada'}")
    print("-" * 80)
    print(
        f"{'MST (Prim)':<25} | {'O(N^2)':<15} | {r2_mst:<10.4f} | {rmse_mst:<10.4f} | "
        f"T(N) = {popt_mst[0]:.2e}·N² + {popt_mst[1]:.2e}"
    )
    print(
        f"{'Matching (Blossom)':<25} | {'O(N^3)':<15} | {r2_match:<10.4f} | {rmse_match:<10.4f} | "
        f"T(N) = {popt_match[0]:.2e}·N³ + {popt_match[1]:.2e}"
    )
    print(
        f"{'Tempo Total':<25} | {'O(N^3)':<15} | {r2_total:<10.4f} | {rmse_total:<10.4f} | "
        f"T(N) = {popt_total[0]:.2e}·N³ + {popt_total[1]:.2e}"
    )
    print("=" * 80)

    # Plota e exporta figuras
    plotar_resultados(df, resultados)

if __name__ == "__main__":
    main()