"""
Exercise 9.2 — Disease Proximity and Drug Ranking
Completed for: RAZVAN24RR
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Tuple

import networkx as nx
import pandas as pd
import numpy as np

# --------------------------
# Config
# --------------------------
HANDLE = "RAZVAN24RR"

# Input: Tabelul drug-gene generat anterior
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Input: Lista genelor bolii
DISEASE_GENES_TXT = Path(f"data/work/{HANDLE}/lab09/disease_genes_{HANDLE}.txt")

# Output directory & file
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    """Verifică existența fișierului."""
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul la: {path.absolute()}")


def load_bipartite_graph_or_build() -> nx.Graph:
    """
    Reconstruim graful din CSV pentru a fi siguri ca avem cea mai recenta versiune
    si pentru a evita problemele de compatibilitate cu pickle.
    """
    print(f"[INFO] Reconstruire graf din {DRUG_GENE_CSV}...")
    df = pd.read_csv(DRUG_GENE_CSV)
    
    G = nx.Graph()
    drugs = df['drug'].unique()
    genes = df['gene'].unique()
    
    # Adaugam noduri cu atribute
    G.add_nodes_from(drugs, bipartite="drug")
    G.add_nodes_from(genes, bipartite="gene")
    
    # Adaugam muchii
    edges = list(zip(df['drug'], df['gene']))
    G.add_edges_from(edges)
    
    return G


def load_disease_genes(path: Path) -> Set[str]:
    """Incarca lista de gene (una pe linie) intr-un set."""
    print(f"[INFO] Incarcare gene boala din {path}...")
    with open(path, 'r') as f:
        # Strip whitespace si ignora liniile goale
        genes = {line.strip() for line in f if line.strip()}
    return genes


def get_drug_nodes(B: nx.Graph) -> List[str]:
    """Returneaza lista tuturor medicamentelor din graf."""
    return [n for n, d in B.nodes(data=True) if d.get("bipartite") == "drug"]


def compute_drug_disease_distance(
    B: nx.Graph,
    drug: str,
    disease_genes: Set[str],
    mode: str = "mean",
    max_dist: int = 5,
) -> float:
    """
    Calculeaza distanta de la 'drug' la setul 'disease_genes'.
    Daca o gena este 'unreachable' sau prea departe, primeste penalizare (max_dist + 1).
    """
    # Calculam lungimile drumurilor de la medicament catre toate celelalte noduri
    # (pana la o adancime maxima pentru eficienta)
    lengths = nx.single_source_shortest_path_length(B, drug, cutoff=max_dist)
    
    distances = []
    for gene in disease_genes:
        if gene in lengths:
            distances.append(lengths[gene])
        else:
            # Penalizare pentru genele la care nu se poate ajunge
            distances.append(max_dist + 1)
    
    if not distances:
        return float(max_dist + 1)
    
    if mode == "mean":
        return np.mean(distances)
    elif mode == "min":
        return np.min(distances)
    else:
        # Default fallback
        return np.mean(distances)


def rank_drugs_by_proximity(
    B: nx.Graph,
    disease_genes: Set[str],
    mode: str = "mean",
) -> pd.DataFrame:
    """
    Itereaza prin toate medicamentele si le calculeaza scorul de proximitate.
    """
    drugs = get_drug_nodes(B)
    
    # Filtram genele bolii care exista efectiv in retea
    # (Nu putem calcula distanta catre un nod care nu exista in graf)
    valid_disease_genes = {g for g in disease_genes if g in B}
    
    print(f"[INFO] Din {len(disease_genes)} gene ale bolii, {len(valid_disease_genes)} se afla in retea.")
    
    if not valid_disease_genes:
        print("[WARN] Nicio gena a bolii nu e in graf! Scorurile vor fi maxime.")
    
    results = []
    for drug in drugs:
        score = compute_drug_disease_distance(B, drug, valid_disease_genes, mode=mode)
        results.append({
            "drug": drug, 
            "distance": score
        })
        
    df = pd.DataFrame(results)
    
    # Sortam crescator (distanta mica = proximitate mare = mai bun)
    df = df.sort_values("distance", ascending=True)
    return df


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # TODO 1: verificați input-urile
    ensure_exists(DRUG_GENE_CSV)
    ensure_exists(DISEASE_GENES_TXT)

    # TODO 2: încărcați / construiți graful bipartit
    G = load_bipartite_graph_or_build()
    print(f"[INFO] Graf incarcat: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii.")

    # TODO 3: încărcați setul de disease genes
    d_genes = load_disease_genes(DISEASE_GENES_TXT)

    # TODO 4: calculați ranking-ul medicamentelor după proximitate
    print("[INFO] Calculare ranking (Shortest Path)...")
    ranking_df = rank_drugs_by_proximity(G, d_genes, mode="mean")

    # TODO 5: salvați rezultatele
    ranking_df.to_csv(OUT_DRUG_PRIORITY, index=False)
    
    print(f"[INFO] Top 5 Medicamente prioritare:")
    print(ranking_df.head(5))
    print(f"[SUCCESS] Rezultatele au fost salvate in: {OUT_DRUG_PRIORITY}")