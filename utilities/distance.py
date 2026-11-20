import math
import numpy as np
from entities.node import *

def distancia_euclidiana(verticeV: Node, verticeU: Node) -> int:
    vX = verticeV.posX
    vY = verticeV.posY
    uX = verticeU.posX
    uY = verticeU.posY
    dist = math.floor(math.sqrt(((vX - uX) ** 2) + ((vY - uY)** 2)) + 0.5)
    return dist


#Vou criar uma matriz de distâncias entre dois vértices, a fim de facilitar os acessos (O(1)) sempre que precisarmos da informação.
#O index do vértice no vetor é o seu valor na entrada - 1.
def criar_matriz_distancias(vetor_vertices: list):
    N = len(vetor_vertices)
    #Inicializa uma matriz N * N com zeros.
    matriz_distancias = np.zeros((N, N), dtype=int)
    for i in range(N):
        #O i+1 garante que a posição d_ii sempre continuará zero
        for j in range(i+1, N):
                dist = distancia_euclidiana(vetor_vertices[i], vetor_vertices[j])
                matriz_distancias[i, j] = dist
                matriz_distancias[j, i] = dist

    return matriz_distancias

