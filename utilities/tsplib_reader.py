import os
from entities.node import Node

def ler_instancia_tsp(caminho_arquivo: str) -> list:
    """
    Lê uma instância da TSPLIB no formato EUC_2D e retorna uma lista de nós.
    """
    vertices = []
    lendo_coords = False

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha_limpa = linha.strip()
            if not linha_limpa:
                continue

            # Detecta o início das coordenadas
            if linha_limpa == "NODE_COORD_SECTION":
                lendo_coords = True
                continue
            elif linha_limpa in ("EOF", "EOF\n"):
                break

            if lendo_coords:
                partes = linha_limpa.split()
                if len(partes) >= 3:
                    # partes[0] é o id (1, 2, 3...) - ignoramos pois o índice na lista já é o id
                    x = float(partes[1])
                    y = float(partes[2])
                    
                    # Cria o Node apenas com posX e posY
                    vertices.append(Node(posX=x, posY=y))

    return vertices