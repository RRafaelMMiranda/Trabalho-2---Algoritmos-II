"""Modelagem de complexidade empirica do Twice-around-the-tree (TP2, DCC207).

Le o CSV gerado por experiments/run_twice_around_the_tree.py e ajusta, por
subrotina (MST, pre-ordem, custo do ciclo e total), curvas teoricas de tempo e de
memoria em funcao de N, reportando R^2, RMSE e AICc para cada modelo candidato.

Uso (a partir de qualquer pasta):
    python twice-around-the-tree/analise_complexidade.py
    python twice-around-the-tree/analise_complexidade.py --sintetico   # N ate 2000, ver abaixo

A opcao --sintetico roda o MST (Prim) em matrizes aleatorias com N bem maior que o das
instancias TSPLIB (14..70). Serve para checar o regime assintotico e para isolar o
efeito da copia de matriz (dtype int64 -> float64) sobre a memoria.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import statistics
import sys
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

PASTA_SCRIPT = Path(__file__).resolve().parent
RAIZ_PROJETO = PASTA_SCRIPT.parent
CSV_PADRAO = RAIZ_PROJETO / "results" / "twice" / "metricas_twice_around_the_tree.csv"

# ============================================================================
# 1. MODELOS TEORICOS
# ============================================================================

def const(n, c):
    return np.full_like(n, c, dtype=float)

def linear(n, a, c):
    return a * n + c

def quadratico(n, a, c):
    return a * n**2 + c

def quadratico_completo(n, a, b, c):
    return a * n**2 + b * n + c

def lei_de_potencia(n, k, p):
    return k * n**p

# ============================================================================
# 2. AJUSTE E METRICAS
# ============================================================================

def aicc(y: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """AIC corrigido para amostras pequenas (menor = melhor compromisso ajuste/complexidade)."""
    n = len(y)
    rss = float(np.sum((y - y_pred) ** 2))
    rss = max(rss, 1e-300)
    base = n * np.log(rss / n) + 2 * k
    return float(base + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else float("inf")

def ajustar(x: np.ndarray, y: np.ndarray, modelo: Callable, p0=None) -> Dict:
    popt, _ = curve_fit(modelo, x, y, p0=p0, maxfev=20000)
    y_pred = modelo(x, *popt)
    return {
        "popt": popt,
        "r2": float(r2_score(y, y_pred)) if np.ptp(y) > 0 else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "aicc": aicc(y, y_pred, len(popt)),
    }

def expoente_loglog(x: np.ndarray, y: np.ndarray) -> float:
    """Expoente empirico p de T ~ N^p (regressao linear em escala log-log)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

def comparar(x, y, candidatos: List[Tuple[str, Callable, tuple]]) -> List[Dict]:
    saida = []
    for nome, f, p0 in candidatos:
        r = ajustar(x, y, f, p0)
        r["modelo"] = nome
        r["f"] = f
        saida.append(r)
    return saida

# ============================================================================
# 3. ANALISE DO CSV (TSPLIB)
# ============================================================================

# (rotulo, coluna, unidade, candidatos). O primeiro candidato e o modelo teorico esperado.
TEMPO = [
    ("MST (Prim)", "tempo_mst_ms", "ms", [
        ("$aN^2+c$", quadratico, (1e-4, 0.0)),
        ("$aN^2+bN+c$", quadratico_completo, (1e-4, 1e-3, 0.0)),
        ("$aN+c$", linear, (1e-2, 0.0)),
    ]),
    ("Pré-ordem", "tempo_preordem_ms", "ms", [
        ("$aN+c$", linear, (1e-3, 0.0)),
        ("$aN^2+c$", quadratico, (1e-6, 0.0)),
    ]),
    ("Custo do ciclo", "tempo_custo_ms", "ms", [
        ("$aN+c$", linear, (1e-3, 0.0)),
        ("$aN^2+c$", quadratico, (1e-6, 0.0)),
    ]),
    ("Total", "tempo_total_ms", "ms", [
        ("$aN^2+c$", quadratico, (1e-4, 0.0)),
        ("$aN^2+bN+c$", quadratico_completo, (1e-4, 1e-3, 0.0)),
        ("$aN+c$", linear, (1e-2, 0.0)),
    ]),
]

MEMORIA = [
    ("MST (Prim)", "mem_pico_mst_kb", "KB", [
        ("$aN^2+bN+c$", quadratico_completo, (1e-3, 0.1, 0.0)),
        ("$aN+c$", linear, (0.5, 0.0)),
    ]),
    ("Pré-ordem", "mem_pico_preordem_kb", "KB", [
        ("$aN+c$", linear, (0.01, 0.0)),
    ]),
    ("Custo do ciclo", "mem_pico_custo_kb", "KB", [
        ("$c$ (constante)", const, (0.2,)),
    ]),
    ("Total", "mem_pico_total_kb", "KB", [
        ("$aN^2+bN+c$", quadratico_completo, (1e-3, 0.1, 0.0)),
        ("$aN+c$", linear, (0.5, 0.0)),
    ]),
]

def tabela(df: pd.DataFrame, blocos, titulo: str) -> Tuple[pd.DataFrame, Dict]:
    x = df["n"].to_numpy(dtype=float)
    linhas, guardado = [], {}
    for rotulo, col, un, cands in blocos:
        y = df[col].to_numpy(dtype=float)
        res = comparar(x, y, cands)
        p = expoente_loglog(x, y) if np.ptp(y) > 0 else float("nan")
        melhor = min(res, key=lambda r: r["aicc"])
        guardado[col] = {"rotulo": rotulo, "unidade": un, "res": res, "melhor": melhor["modelo"], "p": p}
        for r in res:
            linhas.append({
                "grandeza": titulo, "subrotina": rotulo, "modelo": r["modelo"].replace("$", ""),
                "R2": r["r2"], "RMSE": r["rmse"], "unidade": un, "AICc": r["aicc"],
                "coeficientes": ", ".join(f"{c:.3e}" for c in r["popt"]),
                "expoente_loglog": p, "melhor_AICc": r["modelo"] == melhor["modelo"],
            })
    return pd.DataFrame(linhas), guardado

# ============================================================================
# 4. GRAFICOS
# ============================================================================

def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
        "lines.linewidth": 1.8, "lines.markersize": 6, "axes.grid": True,
        "grid.linestyle": "--", "grid.alpha": 0.5,
    })

CORES = ["#0b4f8a", "#c0392b", "#7f8c8d"]
ESTILOS = ["-", "--", ":"]

def painel(ax, df, col, info, ylabel):
    x = df["n"].to_numpy(dtype=float)
    xs = np.linspace(x.min(), x.max(), 300)
    ax.scatter(x, df[col], color="#1f77b4", edgecolors="k", zorder=5, label="Dados empíricos")
    for i, r in enumerate(info["res"]):
        marca = " (menor AICc)" if r["modelo"] == info["melhor"] else ""
        ax.plot(xs, r["f"](xs, *r["popt"]), color=CORES[i % 3], linestyle=ESTILOS[i % 3],
                label=f"{r['modelo']}{marca}\n$R^2={r['r2']:.4f}$, RMSE$={r['rmse']:.3g}$")
    ax.set_title(info["rotulo"])
    ax.set_xlabel("Número de vértices ($N$)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")

def figura(df, guardado, blocos, titulo, ylabel, nome, layout):
    estilo()
    fig, axes = plt.subplots(*layout, figsize=(6.2 * layout[1], 4.6 * layout[0]), dpi=300)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(titulo, fontweight="bold")
    for ax, (_, col, _, _) in zip(axes, blocos):
        painel(ax, df, col, guardado[col], ylabel)
    for ax in axes[len(blocos):]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PASTA_SCRIPT / f"{nome}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f" -> {nome}.png / {nome}.pdf")

# ============================================================================
# 5. EXPERIMENTO SINTETICO (regime assintotico + efeito da copia de dtype)
# ============================================================================

def carregar_modulo_tat():
    caminho = RAIZ_PROJETO / "algorithms" / "twice-around-the-tree.py"
    spec = importlib.util.spec_from_file_location("twice_around_the_tree", caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["twice_around_the_tree"] = mod
    spec.loader.exec_module(mod)
    return mod

def matriz_aleatoria(n: int, dtype, rng) -> np.ndarray:
    pts = rng.random((n, 2)) * 1000.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return np.rint(d).astype(dtype)

def pico_kb(func, *args) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        func(*args)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pico / 1024.0

def experimento_sintetico(repeticoes: int = 7) -> pd.DataFrame:
    tat = carregar_modulo_tat()
    rng = np.random.default_rng(42)
    linhas = []
    for n in (14, 30, 50, 70, 100, 200, 300, 500, 750, 1000, 1500, 2000):
        m_int = matriz_aleatoria(n, np.int64, rng)   # como o projeto gera (criar_matriz_distancias)
        m_flt = m_int.astype(np.float64)             # sem a conversao dentro de mst_prim
        tat.mst_prim(m_flt)                          # aquecimento
        ts = [tat.medir_subrotina(tat.mst_prim, m_flt)[1] * 1000 for _ in range(repeticoes)]
        linhas.append({
            "n": n,
            "tempo_mst_ms": statistics.median(ts),
            "mem_mst_int64_kb": pico_kb(tat.mst_prim, m_int),
            "mem_mst_float64_kb": pico_kb(tat.mst_prim, m_flt),
        })
        print(f"   N={n:<5} tempo={linhas[-1]['tempo_mst_ms']:.3f} ms")
    return pd.DataFrame(linhas)

def analisar_sintetico(df: pd.DataFrame):
    x = df["n"].to_numpy(dtype=float)
    print("\n" + "=" * 92)
    print("EXPERIMENTO SINTÉTICO (MST/Prim, matrizes aleatórias, N até 2000)")
    print("=" * 92)
    res_t = comparar(x, df["tempo_mst_ms"].to_numpy(), [
        ("$aN^2+bN+c$", quadratico_completo, (1e-5, 1e-3, 0.0)),
        ("$aN+c$", linear, (1e-2, 0.0)),
    ])
    res_i = comparar(x, df["mem_mst_int64_kb"].to_numpy(), [
        ("$aN^2+bN+c$", quadratico_completo, (1e-3, 0.1, 0.0)),
        ("$aN+c$", linear, (0.5, 0.0)),
    ])
    res_f = comparar(x, df["mem_mst_float64_kb"].to_numpy(), [
        ("$aN^2+bN+c$", quadratico_completo, (1e-6, 0.1, 0.0)),
        ("$aN+c$", linear, (0.1, 0.0)),
    ])
    for nome, res, col in (("tempo MST", res_t, "tempo_mst_ms"),
                           ("memória MST (entrada int64)", res_i, "mem_mst_int64_kb"),
                           ("memória MST (entrada float64)", res_f, "mem_mst_float64_kb")):
        p = expoente_loglog(x, df[col].to_numpy())
        print(f"\n{nome}  | expoente log-log p = {p:.3f}")
        for r in res:
            print(f"   {r['modelo'].replace('$', ''):<14} R2={r['r2']:.5f}  RMSE={r['rmse']:.4g}  "
                  f"AICc={r['aicc']:.1f}  coef=[{', '.join(f'{c:.3e}' for c in r['popt'])}]")

    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=300)
    fig.suptitle("Verificação em regime assintótico — MST (Prim), instâncias sintéticas", fontweight="bold")
    xs = np.linspace(x.min(), x.max(), 300)
    ax = axes[0]
    ax.scatter(x, df["tempo_mst_ms"], color="#1f77b4", edgecolors="k", zorder=5, label="Dados empíricos")
    for i, r in enumerate(res_t):
        ax.plot(xs, r["f"](xs, *r["popt"]), color=CORES[i], linestyle=ESTILOS[i],
                label=f"{r['modelo']}\n$R^2={r['r2']:.4f}$, RMSE$={r['rmse']:.3g}$")
    ax.set_title("Tempo"); ax.set_xlabel("Número de vértices ($N$)"); ax.set_ylabel("Tempo (ms)")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.scatter(x, df["mem_mst_int64_kb"], color="#c0392b", edgecolors="k", zorder=5, label="Entrada int64 (como no projeto)")
    ax.scatter(x, df["mem_mst_float64_kb"], color="#27ae60", edgecolors="k", marker="s", zorder=5, label="Entrada float64 (sem cópia)")
    ax.plot(xs, 8 * xs**2 / 1024, color="#7f8c8d", linestyle=":", label="$8N^2$ bytes (uma cópia float64 da matriz)")
    ax.set_title("Pico de memória"); ax.set_xlabel("Número de vértices ($N$)"); ax.set_ylabel("Pico (KB)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PASTA_SCRIPT / f"complexidade_sintetica_twice.{ext}", bbox_inches="tight")
    plt.close(fig)
    df.to_csv(PASTA_SCRIPT / "metricas_sinteticas_twice.csv", index=False)
    print("\n -> complexidade_sintetica_twice.png / .pdf e metricas_sinteticas_twice.csv")

# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Regressão de complexidade empírica do Twice-around-the-tree.")
    ap.add_argument("--csv", type=Path, default=CSV_PADRAO)
    ap.add_argument("--sintetico", action="store_true",
                    help="Roda também o MST em instâncias aleatórias com N até 2000.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("n").reset_index(drop=True)
    print(f"{len(df)} instâncias lidas de {args.csv} (N de {df['n'].min()} a {df['n'].max()})")

    t_df, t_g = tabela(df, TEMPO, "tempo")
    m_df, m_g = tabela(df, MEMORIA, "memória")
    todos = pd.concat([t_df, m_df], ignore_index=True)

    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print("\n" + "=" * 110)
    print("REGRESSÃO NÃO LINEAR (scipy.optimize.curve_fit) — R², RMSE e AICc por modelo candidato")
    print("=" * 110)
    for g in ("tempo", "memória"):
        print(f"\n--- {g.upper()} ---")
        sub = todos[todos["grandeza"] == g].drop(columns="grandeza")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    todos.to_csv(PASTA_SCRIPT / "resultados_ajuste_twice.csv", index=False)

    print("\nGráficos:")
    figura(df, t_g, TEMPO, "Complexidade empírica de tempo — Twice-around-the-tree (TSPLIB)",
           "Tempo (ms)", "complexidade_tempo_twice", (2, 2))
    figura(df, m_g, MEMORIA, "Complexidade empírica de memória — Twice-around-the-tree (TSPLIB)",
           "Pico de memória (KB)", "complexidade_memoria_twice", (2, 2))

    if args.sintetico:
        print("\nExecutando experimento sintético (pode levar alguns segundos)...")
        analisar_sintetico(experimento_sintetico())

if __name__ == "__main__":
    main()
