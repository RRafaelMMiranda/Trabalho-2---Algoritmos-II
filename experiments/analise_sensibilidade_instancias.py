from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from algorithms.christofides import subrotina_mst_prim
from utilities.tsplib_reader import carregar_instancia_tsplib
from utilities.warm_start import _fecho_convexo, para_lista_distancias

PASTA_TSPLIB = RAIZ_PROJETO / "data" / "tsplib"
DIR_SAIDA_CSV = RAIZ_PROJETO / "results" / "features_instancias_sensibilidade.csv"
DIR_FIGURAS = RAIZ_PROJETO / "bnb" / "analise-complexidade"

INSTANCIAS_ALVO = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22",
    "gr24", "fri26", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70",
]

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


def calcular_entropia_espacial(xs: np.ndarray, ys: np.ndarray, bins: int = 4) -> float:
    """
    Entropia de Shannon normalizada em [0, 1] sobre uma grade bins x bins no bounding box.
    Valores menores indicam forte agrupamento (clustering); valores proximos de 1 indicam
    distribuicao espacial mais uniforme.
    """
    hist, _, _ = np.histogram2d(xs, ys, bins=bins)
    probs = hist.flatten()
    probs = probs[probs > 0] / probs.sum()
    entropia = -np.sum(probs * np.log2(probs))
    entropia_max = math.log2(min(len(xs), bins * bins))
    return float(entropia / entropia_max) if entropia_max > 0 else 0.0


def extrair_features_instancia(nome: str) -> Dict[str, object]:
    caminho = PASTA_TSPLIB / f"{nome}.tsp"
    matriz, n, nodes, meta = carregar_instancia_tsplib(caminho)
    dist_list = para_lista_distancias(matriz)

    xs = np.array([p.posX for p in nodes], dtype=float)
    ys = np.array([p.posY for p in nodes], dtype=float)

    # 1. Razao de aspecto (aspect ratio) do bounding box
    dx = float(xs.max() - xs.min())
    dy = float(ys.max() - ys.min())
    menor_lado = min(dx, dy)
    maior_lado = max(dx, dy)
    aspect_ratio = maior_lado / menor_lado if menor_lado > 1e-9 else maior_lado

    # 2. Proporcao de pontos pertencentes ao fecho convexo
    fecho = _fecho_convexo(nodes)
    hull_size = len(fecho)
    hull_ratio = hull_size / n

    # 3. Entropia espacial normalizada (clustering vs uniformidade)
    entropia_espacial = calcular_entropia_espacial(xs, ys, bins=4)

    # 4. Estatisticas da Arvore Geradora Minima (MST)
    mst_edges = subrotina_mst_prim(dist_list)
    graus = np.zeros(n, dtype=int)
    pesos_mst = []
    for u, v in mst_edges:
        graus[u] += 1
        graus[v] += 1
        pesos_mst.append(dist_list[u][v])

    mst_grau_max = int(graus.max())
    mst_prop_impares = float(np.sum(graus % 2 != 0) / n)
    mst_prop_folhas = float(np.sum(graus == 1) / n)
    mst_cv_arestas = float(np.std(pesos_mst) / np.mean(pesos_mst)) if np.mean(pesos_mst) > 0 else 0.0

    # 5. Coeficiente de variacao das distancias ao vizinho mais proximo (heterogeneidade local)
    nn_dists = []
    for i in range(n):
        row = dist_list[i]
        nn_dists.append(min(row[j] for j in range(n) if j != i))
    nn_arr = np.array(nn_dists, dtype=float)
    cv_vizinho_proximo = float(nn_arr.std() / nn_arr.mean()) if nn_arr.mean() > 0 else 0.0

    return {
        "instancia": nome,
        "n": n,
        "tipo_metrica": meta.get("edge_weight_type", "EUC_2D"),
        "origem_coordenadas": meta.get("origem_coordenadas", "NODE_COORD_SECTION"),
        "opt_tsplib": OTIMOS_TSPLIB[nome],
        "aspect_ratio": round(aspect_ratio, 4),
        "hull_size": hull_size,
        "hull_ratio": round(hull_ratio, 4),
        "entropia_espacial": round(entropia_espacial, 4),
        "mst_grau_max": mst_grau_max,
        "mst_prop_impares": round(mst_prop_impares, 4),
        "mst_prop_folhas": round(mst_prop_folhas, 4),
        "mst_cv_arestas": round(mst_cv_arestas, 4),
        "cv_vizinho_proximo": round(cv_vizinho_proximo, 4),
    }


def carregar_gaps_experimentais() -> pd.DataFrame:
    df_cs = pd.read_csv(RAIZ_PROJETO / "results" / "bnb" / "metricas_bnb_cold_start.csv", keep_default_na=False)
    df_nn = pd.read_csv(RAIZ_PROJETO / "results" / "bnb" / "nearest_bnb_nearest_neighbor.csv", keep_default_na=False)
    df_ch = pd.read_csv(RAIZ_PROJETO / "results" / "bnb" / "metricas_bnb_convex_hull.csv", keep_default_na=False)
    df_cr = pd.read_csv(RAIZ_PROJETO / "results" / "bnb" / "metricas_bnb_christofides.csv", keep_default_na=False)
    df_ta = pd.read_csv(RAIZ_PROJETO / "results" / "twice" / "metricas_twice_around_the_tree.csv", keep_default_na=False)

    base = pd.DataFrame({"instancia": INSTANCIAS_ALVO})
    base["opt"] = base["instancia"].map(OTIMOS_TSPLIB)

    def _map_num(df: pd.DataFrame, col: str) -> pd.Series:
        m = dict(zip(df["instancia"], pd.to_numeric(df[col])))
        return base["instancia"].map(m)

    base["lb0"] = _map_num(df_cs, "root_lower_bound")
    base["custo_nn"] = _map_num(df_nn, "initial_upper_bound")
    base["custo_ch"] = _map_num(df_ch, "initial_upper_bound")
    base["custo_tat"] = _map_num(df_ta, "custo")
    base["custo_chr"] = _map_num(df_cr, "initial_upper_bound")

    base["gap_lb0_pct"] = (100.0 * (base["opt"] - base["lb0"]) / base["opt"]).round(2)
    base["gap_nn_pct"] = (100.0 * (base["custo_nn"] - base["opt"]) / base["opt"]).round(2)
    base["gap_ch_pct"] = (100.0 * (base["custo_ch"] - base["opt"]) / base["opt"]).round(2)
    base["gap_tat_pct"] = (100.0 * (base["custo_tat"] - base["opt"]) / base["opt"]).round(2)
    base["gap_chr_pct"] = (100.0 * (base["custo_chr"] - base["opt"]) / base["opt"]).round(2)
    base["gap_medio_aprox_pct"] = ((base["gap_tat_pct"] + base["gap_chr_pct"]) / 2.0).round(2)

    return base.drop(columns=["opt"])


def configurar_estilo_cientifico() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13.5,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    })


def main() -> None:
    feat_rows = [extrair_features_instancia(nome) for nome in INSTANCIAS_ALVO]
    df_feat = pd.DataFrame(feat_rows)
    df_gaps = carregar_gaps_experimentais()
    df = df_feat.merge(df_gaps, on="instancia")

    DIR_SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_SAIDA_CSV, index=False)

    print("=== TABELA DE FEATURES E GAPS (14 INSTANCIAS) ===")
    cols_show = [
        "instancia", "n", "aspect_ratio", "hull_ratio", "entropia_espacial",
        "mst_grau_max", "mst_prop_impares", "cv_vizinho_proximo",
        "gap_ch_pct", "gap_chr_pct", "gap_nn_pct", "gap_tat_pct", "gap_lb0_pct",
    ]
    print(df[cols_show].to_string(index=False))

    # 5 features canonicas do enunciado + proporcao de impares na MST
    features = [
        "aspect_ratio",
        "hull_ratio",
        "entropia_espacial",
        "mst_grau_max",
        "cv_vizinho_proximo",
        "mst_prop_impares",
    ]
    feat_labels = [
        "Aspect Ratio",
        "Prop. Fecho Conv.",
        "Entropia Espacial",
        "Grau Máx. MST",
        "CV Viz. Próximo",
        "Prop. Ímpares MST",
    ]

    targets = [
        ("gap_ch_pct", "Convex Hull"),
        ("gap_chr_pct", "Christofides"),
        ("gap_nn_pct", "Nearest Neighbor"),
        ("gap_tat_pct", "Twice-Around"),
        ("gap_lb0_pct", "Gap LB0 (1-Tree)"),
    ]

    # Matriz de correlacao de Pearson (r)
    corr_mat = np.zeros((len(features), len(targets)))
    print("\n=== CORRELACOES DE PEARSON (r) E SPEARMAN (rho) ===")
    for i, fcol in enumerate(features):
        vals = []
        for j, (tcol, tname) in enumerate(targets):
            r, p = pearsonr(df[fcol], df[tcol])
            rho, _ = spearmanr(df[fcol], df[tcol])
            corr_mat[i, j] = r
            vals.append(f"{tname}: r={r:+.3f} (rho={rho:+.3f})")
        print(f"{fcol:<20} | " + " | ".join(vals))

    # Regressao Multipla Padronizada + Arvore de Decisao (5 features canonicas do enunciado)
    canon_feats = [
        "aspect_ratio",
        "hull_ratio",
        "entropia_espacial",
        "mst_grau_max",
        "cv_vizinho_proximo",
    ]
    canon_labels = [
        "Aspect Ratio",
        "Prop. Fecho Conv.",
        "Entropia Espacial",
        "Grau Máx. MST",
        "CV Viz. Próximo",
    ]

    X = df[canon_feats].to_numpy(dtype=float)
    X_std = StandardScaler().fit_transform(X)

    print("\n=== REGRESSAO MULTIPLA PADRONIZADA (5 FEATURES CANONICAS) ===")
    betas: Dict[str, np.ndarray] = {}
    reg_metrics: Dict[str, Dict[str, float]] = {}
    for tcol, tname in [
        ("gap_ch_pct", "Convex Hull"),
        ("gap_chr_pct", "Christofides"),
        ("gap_nn_pct", "Nearest Neighbor"),
        ("gap_tat_pct", "Twice-Around"),
        ("gap_medio_aprox_pct", "Média Aprox. (TAT+Chr)"),
        ("gap_lb0_pct", "Gap LB0 (1-Tree)"),
    ]:
        y = df[tcol].to_numpy(dtype=float)
        lr = LinearRegression().fit(X_std, y)
        pred = lr.predict(X_std)
        r2 = float(r2_score(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        betas[tcol] = lr.coef_
        reg_metrics[tcol] = {"r2": r2, "rmse": rmse}
        coef_str = ", ".join(f"{k}={v:+.2f}" for k, v in zip(canon_feats, lr.coef_))
        print(f"{tname:<24} | R2={r2:.4f} | RMSE={rmse:5.2f} pp | {coef_str}")

    print("\n=== ARVORE DE DECISAO (max_depth=2) ===")
    tree_imps: Dict[str, np.ndarray] = {}
    for tcol, tname in [
        ("gap_chr_pct", "Christofides"),
        ("gap_tat_pct", "Twice-Around"),
        ("gap_medio_aprox_pct", "Média Aprox. (TAT+Chr)"),
        ("gap_lb0_pct", "Gap LB0 (1-Tree)"),
    ]:
        y = df[tcol].to_numpy(dtype=float)
        dt = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=42).fit(X, y)
        pred = dt.predict(X)
        r2 = float(r2_score(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        tree_imps[tcol] = dt.feature_importances_
        print(f"\n[{tname}] R2={r2:.4f}, RMSE={rmse:.2f} pp")
        print("Importancias:", dict(zip(canon_feats, np.round(dt.feature_importances_, 3))))
        print(export_text(dt, feature_names=canon_feats, decimals=3))

    # ========================================================================
    # FIGURA DE ANALISE DE SENSIBILIDADE (3 PAINEIS - PADRAO SBC)
    # ========================================================================
    configurar_estilo_cientifico()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), dpi=300)
    fig.suptitle(
        "Análise de Sensibilidade Topológica e Instâncias Adversariais (TSPLIB, 14 Instâncias)",
        fontweight="bold",
        y=1.02,
    )

    # --- Painel 1: Heatmap de Correlacao de Pearson ---
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t[1] for t in targets], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feat_labels, fontsize=8.5)
    for i in range(len(features)):
        for j in range(len(targets)):
            val = corr_mat[i, j]
            cor_txt = "white" if abs(val) > 0.50 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=cor_txt, fontsize=8.2, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlação de Pearson ($r$)", fontsize=9)
    ax.set_title("Correlação: Features vs. Gaps (%)")

    # --- Painel 2: Pesos Padronizados da Regressao Multipla (Christofides e Twice-Around) ---
    ax = axes[1]
    y_pos = np.arange(len(canon_labels))
    h = 0.26
    ax.barh(y_pos - h, betas["gap_chr_pct"], height=h, color="#1f77b4", edgecolor="k", linewidth=0.5,
            label=f"Christofides ($R^2={reg_metrics['gap_chr_pct']['r2']:.2f}$)")
    ax.barh(y_pos, betas["gap_tat_pct"], height=h, color="#ff7f0e", edgecolor="k", linewidth=0.5,
            label=f"Twice-Around ($R^2={reg_metrics['gap_tat_pct']['r2']:.2f}$)")
    ax.barh(y_pos + h, betas["gap_lb0_pct"], height=h, color="#2ca02c", edgecolor="k", linewidth=0.5,
            label=f"Gap $\\mathrm{{LB}}_0$ ($R^2={reg_metrics['gap_lb0_pct']['r2']:.2f}$)")
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(canon_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Coeficiente Padronizado ($\\beta$, em p.p. por $\\sigma$)")
    ax.set_title("Peso das Features (Regressão Múltipla)")
    ax.set_xlim(-6.8, 5.2)
    ax.legend(loc="lower left", fontsize=7.8)

    # --- Painel 3: Instancias Adversariais (Aspect Ratio vs Gap Medio Aproximado + corte Arvore de Decisao) ---
    ax = axes[2]
    sc = ax.scatter(
        df["aspect_ratio"],
        df["gap_medio_aprox_pct"],
        c=df["cv_vizinho_proximo"],
        cmap="viridis",
        s=72,
        edgecolors="k",
        zorder=5,
    )
    cbar3 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar3.set_label("CV Vizinho Próximo", fontsize=9)

    # Corte principal da Arvore de Decisao (aspect_ratio = 1.921)
    ax.axvline(1.921, color="#d62728", linestyle=":", linewidth=1.4, label="Corte Árvore ($AR = 1.92$)")

    # Linha de tendencia linear
    z = np.polyfit(df["aspect_ratio"], df["gap_medio_aprox_pct"], 1)
    x_line = np.linspace(df["aspect_ratio"].min(), df["aspect_ratio"].max(), 100)
    r_ar, _ = pearsonr(df["aspect_ratio"], df["gap_medio_aprox_pct"])
    ax.plot(x_line, np.polyval(z, x_line), "k--", linewidth=1.2, alpha=0.75, label=f"Tendência ($r = {r_ar:+.2f}$)")

    offsets = {
        "eil51": (6, 2),
        "dantzig42": (6, 2),
        "bays29": (-8, -13),
        "gr21": (6, -3),
        "berlin52": (6, 2),
        "att48": (6, -3),
        "gr24": (-32, -3),
        "swiss42": (6, -3),
        "fri26": (-26, -12),
        "st70": (6, -4),
        "burma14": (6, -3),
        "gr17": (6, -3),
        "ulysses22": (-56, 6),
        "ulysses16": (-56, -10),
    }
    for _, row in df.iterrows():
        dx, dy = offsets.get(row["instancia"], (5, 4))
        ax.annotate(
            row["instancia"],
            (row["aspect_ratio"], row["gap_medio_aprox_pct"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
        )

    ax.set_xlim(0.75, 4.35)
    ax.set_ylim(5.0, 32.5)
    ax.set_xlabel("Razão de Aspecto do Bounding Box ($AR$)")
    ax.set_ylabel("Gap Médio Aproximado: (TAT + Chr)/2 (%)")
    ax.set_title("Mapa de Instâncias Adversariais")
    ax.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    png_path = DIR_FIGURAS / "analise_sensibilidade_instancias.png"
    pdf_path = DIR_FIGURAS / "analise_sensibilidade_instancias.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[SUCESSO] Dataset salvo em: {DIR_SAIDA_CSV}")
    print(f"[SUCESSO] Figuras salvas em:\n -> {png_path}\n -> {pdf_path}")


if __name__ == "__main__":
    main()
