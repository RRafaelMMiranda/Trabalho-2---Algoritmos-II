from __future__ import annotations

import math
from typing import List, Sequence
import numpy as np

from entities.node import Node

PI = 3.141592
RAIO_TERRA = 6378.388  # km (especificação padrão TSPLIB para GEO)

def distancia_euclidiana(verticeV: Node, verticeU: Node) -> int:
    """Calculo padrao EUC_2D com arredondamento TSPLIB (nint)."""
    dx = verticeV.posX - verticeU.posX
    dy = verticeV.posY - verticeU.posY
    return int(math.floor(math.sqrt(dx * dx + dy * dy) + 0.5))

def distancia_pseudo_euclidiana_att(verticeV: Node, verticeU: Node) -> int:
    """Calculo ATT (pseudo-euclidiano usado em att48)."""
    dx = verticeV.posX - verticeU.posX
    dy = verticeV.posY - verticeU.posY
    rij = math.sqrt((dx * dx + dy * dy) / 10.0)
    tij = int(math.floor(rij + 0.5))
    return tij + 1 if tij < rij else tij

def _coord_para_radianos_geo(valor: float) -> float:
    deg = int(valor)
    minuto = valor - deg
    return PI * (deg + 5.0 * minuto / 3.0) / 180.0

def distancia_geografica_geo(verticeV: Node, verticeU: Node) -> int:
    """Calculo GEO oficial da TSPLIB (ulysses16, bays29, etc.)."""
    lat_v = _coord_para_radianos_geo(verticeV.posX)
    lon_v = _coord_para_radianos_geo(verticeV.posY)
    lat_u = _coord_para_radianos_geo(verticeU.posX)
    lon_u = _coord_para_radianos_geo(verticeU.posY)

    q1 = math.cos(lon_v - lon_u)
    q2 = math.cos(lat_v - lat_u)
    q3 = math.cos(lat_v + lat_u)

    termo = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    termo = max(-1.0, min(1.0, termo))  # Evita erro numerico fora de [-1, 1]
    return int(RAIO_TERRA * math.acos(termo) + 1.0)

def calcular_distancia(u: Node, v: Node, metric: str = "EUC_2D") -> int:
    """Despacha o calculo conforme a metrica especificada no cabecalho da TSPLIB."""
    metric = metric.upper().strip()
    if metric == "EUC_2D":
        return distancia_euclidiana(u, v)
    elif metric == "ATT":
        return distancia_pseudo_euclidiana_att(u, v)
    elif metric == "GEO":
        return distancia_geografica_geo(u, v)
    else:
        # Fallback padrao
        return distancia_euclidiana(u, v)

def criar_matriz_distancias(vetor_vertices: Sequence[Node], metric: str = "EUC_2D") -> np.ndarray:
    """Gera a matriz simetrica N x N a partir dos nos e da metrica da instancia."""
    n = len(vetor_vertices)
    matriz = np.zeros((n, n), dtype=int)
    for i in range(n):
        u = vetor_vertices[i]
        for j in range(i + 1, n):
            v = vetor_vertices[j]
            d = calcular_distancia(u, v, metric=metric)
            matriz[i, j] = d
            matriz[j, i] = d
    return matriz