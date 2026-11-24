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

def carregar_instancia_tsplib(
    caminho_arquivo: Union[str, Path]
) -> Tuple[np.ndarray, int, Optional[List[Node]], Dict[str, Any]]:
    """
    Le uma instancia da TSPLIB e retorna:
    - matriz_distancias (np.ndarray int)
    - dimensao (int)
    - lista de vertices (List[Node] se baseado em coordenadas, ou None se for EXPLICIT)
    - metadados (dict contendo nome, tipo, metrica, etc.)
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

        if "NODE_COORD_SECTION" in linha.upper():
            idx_coord = i
        elif "EDGE_WEIGHT_SECTION" in linha.upper():
            idx_matrix = i

    dimensao = metadados["dimensao"]
    tipo_peso = metadados["edge_weight_type"]

    # Caso 1: Instancia com Coordenadas (EUC_2D, ATT, GEO)
    if idx_coord != -1:
        nodes: List[Node] = []
        for linha in linhas[idx_coord + 1:]:
            if linha.upper() in ("EOF", ""):
                break
            partes = linha.split()
            if len(partes) >= 3:
                x = float(partes[1])
                y = float(partes[2])
                nodes.append(Node(x, y))

        if dimensao == 0 or len(nodes) != dimensao:
            dimensao = len(nodes)
            metadados["dimensao"] = dimensao

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

        return matriz, dimensao, None, metadados

    raise ValueError(f"Estrutura da instancia '{caminho.name}' nao reconhecida.")