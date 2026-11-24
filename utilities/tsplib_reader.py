from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from entities.node import Node
from utilities.distance import criar_matriz_distancias


def _ler_secao_coordenadas(linhas: List[str], idx_inicio: int) -> List[Node]:
    """Extrai lista de Node a partir de NODE_COORD_SECTION ou DISPLAY_DATA_SECTION."""
    nodes: List[Node] = []
    for linha in linhas[idx_inicio + 1:]:
        topo = linha.upper()
        if topo in ("EOF", "") or topo.endswith("_SECTION"):
            break
        partes = linha.split()
        if len(partes) >= 3:
            try:
                x = float(partes[1])
                y = float(partes[2])
                nodes.append(Node(x, y))
            except ValueError:
                continue
    return nodes


def _reconstruir_coordenadas_mds(matriz: np.ndarray) -> List[Node]:
    """
    Projeta uma matriz de distancias simetrica em 2D via Classical MDS (Torgerson).
    Usado apenas para instancias EXPLICIT sem NODE_COORD_SECTION nem DISPLAY_DATA_SECTION,
    permitindo executar heuristicas geometricas (ex.: Fecho Convexo) sem falhas,
    mantendo os custos exatos da matriz original.
    """
    n = matriz.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [Node(0.0, 0.0)]

    d2 = matriz.astype(float) ** 2
    b = -0.5 * (d2 - d2.mean(axis=1, keepdims=True) - d2.mean(axis=0, keepdims=True) + d2.mean())
    autovalores, autovetores = np.linalg.eigh(b)
    ordem = np.argsort(autovalores)[::-1]

    l1 = max(0.0, float(autovalores[ordem[0]]))
    l2 = max(0.0, float(autovalores[ordem[1]]))
    xs = autovetores[:, ordem[0]] * np.sqrt(l1)
    ys = autovetores[:, ordem[1]] * np.sqrt(l2)
    return [Node(float(xs[i]), float(ys[i])) for i in range(n)]


def carregar_instancia_tsplib(
    caminho_arquivo: Union[str, Path]
) -> Tuple[np.ndarray, int, List[Node], Dict[str, Any]]:
    """
    Le uma instancia da TSPLIB e retorna:
    - matriz_distancias (np.ndarray int)
    - dimensao (int)
    - lista de vertices (List[Node]):
        1) NODE_COORD_SECTION (quando disponivel),
        2) DISPLAY_DATA_SECTION (em instancias EXPLICIT como bays29/dantzig42),
        3) Projecao 2D via Classical MDS (em instancias EXPLICIT puras como gr17/gr24/fri26/swiss42).
    - metadados (dict contendo nome, tipo, metrica e origem_coordenadas)
    """
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    metadados: Dict[str, Any] = {
        "nome": caminho.stem,
        "dimensao": 0,
        "edge_weight_type": "EUC_2D",
        "edge_weight_format": "FUNCTION",
    }

    idx_coord = -1
    idx_matrix = -1
    idx_display = -1

    for i, linha in enumerate(linhas):
        if ":" in linha:
            partes = [p.strip() for p in linha.split(":", 1)]
            chave = partes[0].upper()
            valor = partes[1].upper()

            if chave == "NAME":
                metadados["nome"] = partes[1].strip()
            elif chave == "DIMENSION":
                metadados["dimensao"] = int(valor)
            elif chave == "EDGE_WEIGHT_TYPE":
                metadados["edge_weight_type"] = valor
            elif chave == "EDGE_WEIGHT_FORMAT":
                metadados["edge_weight_format"] = valor

        linha_up = linha.upper()
        if "NODE_COORD_SECTION" in linha_up:
            idx_coord = i
        elif "EDGE_WEIGHT_SECTION" in linha_up:
            idx_matrix = i
        elif "DISPLAY_DATA_SECTION" in linha_up:
            idx_display = i

    dimensao = metadados["dimensao"]
    tipo_peso = metadados["edge_weight_type"]

    # Caso 1: Instancia com Coordenadas (EUC_2D, ATT, GEO)
    if idx_coord != -1:
        nodes = _ler_secao_coordenadas(linhas, idx_coord)

        if dimensao == 0 or len(nodes) != dimensao:
            dimensao = len(nodes)
            metadados["dimensao"] = dimensao

        metadados["origem_coordenadas"] = "NODE_COORD_SECTION"
        matriz = criar_matriz_distancias(nodes, metric=tipo_peso)
        return matriz, dimensao, nodes, metadados

    # Caso 2: Instancia com Matriz Explicita (gr17, swiss42)
    elif idx_matrix != -1 or tipo_peso == "EXPLICIT":
        valores: List[int] = []
        inicio = idx_matrix + 1 if idx_matrix != -1 else 0
        for linha in linhas[inicio:]:
            if linha.upper() in ("EOF", "DISPLAY_DATA_SECTION"):
                break
            for val in linha.split():
                try:
                    valores.append(int(float(val)))
                except ValueError:
                    continue

        formato = metadados.get("edge_weight_format", "").upper()
        matriz = np.zeros((dimensao, dimensao), dtype=int)

        if formato == "LOWER_DIAG_ROW":
            k = 0
            for i in range(dimensao):
                for j in range(i + 1):
                    matriz[i, j] = valores[k]
                    matriz[j, i] = valores[k]
                    k += 1
        elif formato == "UPPER_ROW":
            k = 0
            for i in range(dimensao - 1):
                for j in range(i + 1, dimensao):
                    matriz[i, j] = valores[k]
                    matriz[j, i] = valores[k]
                    k += 1
        else:
            # FULL_MATRIX ou deducao pelo numero de valores
            if len(valores) >= dimensao * dimensao:
                matriz = np.array(valores[:dimensao * dimensao], dtype=int).reshape((dimensao, dimensao))
            elif len(valores) == dimensao * (dimensao + 1) // 2:
                k = 0
                for i in range(dimensao):
                    for j in range(i + 1):
                        matriz[i, j] = valores[k]
                        matriz[j, i] = valores[k]
                        k += 1

        if idx_display != -1:
            nodes_display = _ler_secao_coordenadas(linhas, idx_display)
            if len(nodes_display) == dimensao:
                metadados["origem_coordenadas"] = "DISPLAY_DATA_SECTION"
                return matriz, dimensao, nodes_display, metadados

        metadados["origem_coordenadas"] = "MDS_2D"
        nodes_mds = _reconstruir_coordenadas_mds(matriz)
        return matriz, dimensao, nodes_mds, metadados

    raise ValueError(f"Estrutura da instancia '{caminho.name}' nao reconhecida.")