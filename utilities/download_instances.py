import gzip
import io
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Instâncias solicitadas
INSTANCIAS = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22",
    "gr24", "fri26", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70"
]

PASTA_DESTINO = Path("./data/tsplib")

# URLs dos repositórios
URL_HEIDELBERG = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
URL_GITHUB_MIRROR = "https://raw.githubusercontent.com/mastqe/tsplib/master/"

# Header simulando navegador comum para não ser barrado por WAF/firewall
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def baixar_arquivo(url: str) -> bytes:
    """Faz requisição HTTP segura com timeout e headers customizados."""
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=15) as resposta:
        return resposta.read()

def obter_conteudo_instancia(nome: str) -> str:
    """
    Tenta obter o arquivo .tsp testando as seguintes alternativas:
    1. Heidelberg .tsp direto
    2. Heidelberg .tsp.gz (e descompacta em memória)
    3. Mirror do GitHub
    """
    tentativas = [
        (f"{URL_HEIDELBERG}{nome}.tsp", False),
        (f"{URL_HEIDELBERG}{nome}.tsp.gz", True),
        (f"{URL_GITHUB_MIRROR}{nome}.tsp", False),
    ]

    for url, compactado in tentativas:
        try:
            dados = baixar_arquivo(url)
            if compactado:
                with gzip.GzipFile(fileobj=io.BytesIO(dados)) as gz:
                    return gz.read().decode("utf-8", errors="replace")
            else:
                return dados.decode("utf-8", errors="replace")
        except (HTTPError, URLError):
            continue

    raise RuntimeError(f"Não foi possível baixar {nome} de nenhuma fonte conhecida.")

def main():
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)
    print(f"Iniciando download de {len(INSTANCIAS)} instâncias para '{PASTA_DESTINO}'...\n")

    for idx, nome in enumerate(INSTANCIAS, 1):
        caminho_arquivo = PASTA_DESTINO / f"{nome}.tsp"

        if caminho_arquivo.exists() and caminho_arquivo.stat().st_size > 0:
            print(f"[{idx}/{len(INSTANCIAS)}] {nome}.tsp já existe localmente. Pulando.")
            continue

        print(f"[{idx}/{len(INSTANCIAS)}] Baixando {nome}...", end="", flush=True)

        try:
            conteudo = obter_conteudo_instancia(nome)
            with open(caminho_arquivo, "w", encoding="utf-8") as f:
                f.write(conteudo)
            print(" [OK]")
        except Exception as e:
            print(f" [FALHOU: {e}]")

        # Intervalo educado entre downloads para evitar bloqueio por IP
        time.sleep(1.0)

    print("\nDownload concluído com sucesso!")

if __name__ == "__main__":
    main()