from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CSV_COLD_START = RAIZ_PROJETO / "results" / "bnb" / "metricas_bnb_cold_start.csv"
CSV_MEMORIA = RAIZ_PROJETO / "results" / "bnb" / "metricas_bnb_memoria.csv"
DIR_SAIDA = Path(__file__).resolve().parent

# Otimos oficiais conhecidos da TSPLIB (usados para calcular o gap da 1-arvore na raiz)
OTIMOS_TSPLIB = {
    "burma14": 3323,
    "ulysses16": 6859,
    "gr17": 2085,
    "gr21": 2707,
    "ulysses22": 7013,
    "gr24": 1272,
    "fri26": 937,
    "bays29": 2020,
    "dantzig42": 699,
    "swiss42": 1273,
    "att48": 10628,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
}


def modelo_exponencial(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * n)


def modelo_quadratico(n: np.ndarray, a: float, c: float) -> np.ndarray:
    return a * (n ** 2) + c


def calcular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def configurar_estilo_cientifico() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
    })


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(CSV_COLD_START, keep_default_na=False)
    for col in [
        "n", "incumbent_cost", "initial_upper_bound", "root_lower_bound",
        "nodes_expanded", "nodes_pruned", "max_depth",
        "elapsed_time", "search_time", "time_warm_start",
        "time_lower_bound_mst", "time_search_overhead",
    ]:
        df[col] = pd.to_numeric(df[col])

    df["opt_tsplib"] = df["instancia"].map(OTIMOS_TSPLIB)
    df["gap_lb0_pct"] = 100.0 * (df["opt_tsplib"] - df["root_lower_bound"]) / df["opt_tsplib"]
    df["mst_pct"] = 100.0 * df["time_lower_bound_mst"] / df["search_time"]
    df["overhead_pct"] = 100.0 - df["mst_pct"]
    df["us_por_no_exp"] = 1e6 * df["time_lower_bound_mst"] / df["nodes_expanded"]

    if CSV_MEMORIA.exists():
        df_mem = pd.read_csv(CSV_MEMORIA, keep_default_na=False)
        df_mem["peak_memory_kb"] = pd.to_numeric(df_mem["peak_memory_bytes"]) / 1024.0
        df = df.merge(df_mem[["instancia", "peak_memory_kb"]], on="instancia", how="left")
    else:
        df["peak_memory_kb"] = np.nan

    df.sort_values(by=["n", "instancia"], inplace=True)
    return df.reset_index(drop=True)


def main() -> None:
    df = carregar_dados()
    sol = df[df["status"] == "OPTIMAL"].copy()
    na_df = df[df["status"] != "OPTIMAL"].copy()

    # 1. Modelos univariados em N (nas 6 resolvidas)
    xs = sol["n"].to_numpy(dtype=float)
    ys_t = sol["elapsed_time"].to_numpy(dtype=float)
    ys_log_t = np.log10(ys_t)
    ys_log_n = np.log10(sol["nodes_expanded"].to_numpy(dtype=float))
    gs = sol["gap_lb0_pct"].to_numpy(dtype=float)

    popt_exp, _ = curve_fit(modelo_exponencial, xs, ys_t, p0=(1.0, 0.05), maxfev=10000)
    pred_exp = modelo_exponencial(xs, *popt_exp)
    r2_exp = float(r2_score(ys_t, pred_exp))
    rmse_exp = calcular_rmse(ys_t, pred_exp)

    lr_n_only = LinearRegression().fit(xs.reshape(-1, 1), ys_log_n)
    pred_n_only = lr_n_only.predict(xs.reshape(-1, 1))
    r2_logn_n = float(r2_score(ys_log_n, pred_n_only))
    rmse_logn_n = calcular_rmse(ys_log_n, pred_n_only)

    # 2. Modelos multivariados (N + Gap_LB0) nas 6 resolvidas
    X_multi = np.column_stack([xs, gs])
    lr_logn_multi = LinearRegression().fit(X_multi, ys_log_n)
    pred_logn_multi = lr_logn_multi.predict(X_multi)
    r2_logn_multi = float(r2_score(ys_log_n, pred_logn_multi))
    rmse_logn_multi = calcular_rmse(ys_log_n, pred_logn_multi)

    lr_logt_multi = LinearRegression().fit(X_multi, ys_log_t)
    pred_logt_multi = lr_logt_multi.predict(X_multi)
    r2_logt_multi = float(r2_score(ys_log_t, pred_logt_multi))
    rmse_logt_multi = calcular_rmse(ys_log_t, pred_logt_multi)

    # 3. Custo por no (MST) vs N^2 (todas as 14 instancias)
    x_all = df["n"].to_numpy(dtype=float)
    y_us = df["us_por_no_exp"].to_numpy(dtype=float)
    popt_q, _ = curve_fit(modelo_quadratico, x_all, y_us, maxfev=10000)
    pred_q = modelo_quadratico(x_all, *popt_q)
    r2_us = float(r2_score(y_us, pred_q))
    rmse_us = calcular_rmse(y_us, pred_q)

    print("=" * 94)
    print("MODELAGEM DE COMPLEXIDADE EMPIRICA - BRANCH-AND-BOUND (COLD START / BASELINE)")
    print("=" * 94)
    print(f"{'Variavel Alvo':<28} | {'Preditores':<16} | {'R2':<8} | {'RMSE':<12} | {'Amostra'}")
    print("-" * 94)
    print(f"{'Tempo Total (s)':<28} | {'a*exp(b*N)':<16} | {r2_exp:<8.4f} | {rmse_exp:<9.3f} s  | 6 resolvidas")
    print(f"{'log10(Nos Expandidos)':<28} | {'N':<16} | {r2_logn_n:<8.4f} | {rmse_logn_n:<9.4f}    | 6 resolvidas")
    print(f"{'log10(Nos Expandidos)':<28} | {'N + Gap_LB0':<16} | {r2_logn_multi:<8.4f} | {rmse_logn_multi:<9.4f}    | 6 resolvidas")
    print(f"{'log10(Tempo Total)':<28} | {'N + Gap_LB0':<16} | {r2_logt_multi:<8.4f} | {rmse_logt_multi:<9.4f}    | 6 resolvidas")
    print(f"{'MST por No Exp. (us)':<28} | {'a*N^2 + c':<16} | {r2_us:<8.4f} | {rmse_us:<9.2f} us | 14 instancias")
    print("-" * 94)
    print(
        f"Equacao log10(Tempo): {lr_logt_multi.coef_[0]:.4f}*N + "
        f"{lr_logt_multi.coef_[1]:.4f}*Gap_LB0 + ({lr_logt_multi.intercept_:.4f})"
    )
    print(
        f"Equacao log10(Nos):   {lr_logn_multi.coef_[0]:.4f}*N + "
        f"{lr_logn_multi.coef_[1]:.4f}*Gap_LB0 + ({lr_logn_multi.intercept_:.4f})"
    )
    print(
        f"Participacao da MST no tempo de busca: min={df['mst_pct'].min():.1f}%, "
        f"media={df['mst_pct'].mean():.1f}%, max={df['mst_pct'].max():.1f}%"
    )
    if df["peak_memory_kb"].notna().any():
        print(
            f"Pico de memoria (tracemalloc): min={df['peak_memory_kb'].min():.1f} KB, "
            f"media={df['peak_memory_kb'].mean():.1f} KB, max={df['peak_memory_kb'].max():.1f} KB"
        )
    print("=" * 94)

    # ========================================================================
    # GERAÇÃO DA FIGURA (3 PAINEIS - PADRAO SBC)
    # ========================================================================
    configurar_estilo_cientifico()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)
    fig.suptitle(
        "Análise Empírica e Profiling - Branch-and-Bound Exato (DFS + MST, TSPLIB)",
        fontweight="bold",
        y=1.02,
    )

    # --- Painel 1: Tempo Total vs N (escala log + teto de 1800s) ---
    ax = axes[0]
    n_cont = np.linspace(xs.min(), xs.max(), 150)
    ax.scatter(sol["n"], sol["elapsed_time"], color="#1f77b4", edgecolors="k", zorder=5, label="Resolvidas (Ótimo)")
    ax.scatter(
        na_df["n"], na_df["elapsed_time"], color="#d62728", marker="^", s=55,
        edgecolors="k", zorder=5, label="Timeout (NA, 1800 s)"
    )
    ax.plot(
        n_cont, modelo_exponencial(n_cont, *popt_exp), color="#0b4f8a", linestyle="-",
        label=f"Ajuste $ae^{{bN}}$ (resolvidas)\n$R^2={r2_exp:.4f}$, RMSE={rmse_exp:.2f} s"
    )
    ax.axhline(1800.0, color="#d62728", linestyle=":", linewidth=1.3, alpha=0.8)

    offsets_p1 = {
        "burma14": (5, -10),
        "ulysses16": (-6, 8),
        "gr17": (5, -11),
        "gr21": (5, -8),
        "gr24": (5, -11),
        "fri26": (5, 3),
        "ulysses22": (-14, -16),
    }
    for _, row in df[df["instancia"].isin(offsets_p1)].iterrows():
        dx, dy = offsets_p1[row["instancia"]]
        ax.annotate(
            row["instancia"],
            (row["n"], row["elapsed_time"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.8,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.08, 5500)
    ax.set_title("Tempo de Execução e Ponto de Colapso")
    ax.set_xlabel("Número de Vértices ($N$)")
    ax.set_ylabel("Tempo Total (s, escala log)")
    ax.legend(loc="lower right")

    # --- Painel 2: Regressao Multivariada (N + Gap_LB0) ---
    ax = axes[1]
    ax.scatter(ys_log_t, pred_logt_multi, color="#2ca02c", edgecolors="k", s=55, zorder=5, label="Instâncias Resolvidas")
    lims = [min(ys_log_t.min(), pred_logt_multi.min()) - 0.25, max(ys_log_t.max(), pred_logt_multi.max()) + 0.35]
    ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.75, label="Identidade ($y = x$)")

    offsets_p2 = {
        "burma14": (8, -4),
        "gr21": (8, -4),
        "gr17": (-64, 14),
        "gr24": (-24, -30),
        "fri26": (-76, 20),
        "ulysses16": (-34, -64),
    }
    for i, (_, row) in enumerate(sol.iterrows()):
        dx, dy = offsets_p2.get(row["instancia"], (6, 4))
        usar_seta = row["instancia"] in ("gr17", "gr24", "fri26", "ulysses16")
        ax.annotate(
            f"{row['instancia']} ({row['gap_lb0_pct']:.0f}%)",
            (ys_log_t[i], pred_logt_multi[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.6,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7) if usar_seta else None,
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Regressão: $\\log_{10}(T) \\sim N + \\mathrm{Gap}_{LB_0}$")
    ax.set_xlabel("$\\log_{10}(\\mathrm{Tempo\\ Observado\\ [s]})$")
    ax.set_ylabel("$\\log_{10}(\\mathrm{Tempo\\ Predito\\ [s]})$")
    ax.text(
        0.05, 0.94,
        f"$R^2 = {r2_logt_multi:.4f}$\n$\\mathrm{{RMSE}} = {rmse_logt_multi:.3f}$\n(vs. $R^2={r2_exp:.4f}$ só com $N$)",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.legend(loc="lower right")

    # --- Painel 3: Profiling de Subrotinas (% MST vs Overhead) ---
    ax = axes[2]
    idx = np.arange(len(df))
    ax.bar(idx, df["mst_pct"], color="#1f77b4", edgecolor="k", linewidth=0.5, label="Subrotina MST (Prim)")
    ax.bar(
        idx, df["overhead_pct"], bottom=df["mst_pct"],
        color="#ff7f0e", edgecolor="k", linewidth=0.5, label="Overhead DFS / Poda"
    )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{r['instancia']}\n({int(r['n'])})" for _, r in df.iterrows()], rotation=55, ha="right", fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_title("Decomposição do Tempo de Busca por Subrotina")
    ax.set_ylabel("Participação no Tempo de Busca (%)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    caminho_png = DIR_SAIDA / "complexidade_empirica_bnb.png"
    caminho_pdf = DIR_SAIDA / "complexidade_empirica_bnb.pdf"
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[SUCESSO] Figuras salvas em:")
    print(f" -> {caminho_png}")
    print(f" -> {caminho_pdf}")


if __name__ == "__main__":
    main()
