"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network
Completed for: RAZVAN24RR
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List

import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Config — adaptați pentru handle-ul vostru
# --------------------------
HANDLE = "RAZVAN24RR"

# Input: fișier cu coloane cel puțin: drug, gene
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Output directory & files
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"
OUT_GRAPH_IMG = OUT_DIR / f"network_drug_gene_{HANDLE}.png"


def ensure_exists(path: Path) -> None:
    """Verifică existența fișierului."""
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul la: {path.absolute()}")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    """Încarcă CSV-ul și validează coloanele."""
    print(f"[INFO] Încărcare date din {path}...")
    df = pd.read_csv(path)
    
    required_cols = {'drug', 'gene'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV-ul trebuie să conțină coloanele: {required_cols}")
        
    return df


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Construiește dicționarul {medicament: {gena1, gena2, ...}}."""
    return df.groupby("drug")["gene"].apply(set).to_dict()


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    """Construiește graful bipartit NetworkX."""
    G = nx.Graph()
    
    drugs = list(drug2genes.keys())
    # Adăugăm nodurile 'drug'
    G.add_nodes_from(drugs, bipartite="drug")
    
    # Adăugăm nodurile 'gene' și muchiile
    for drug, genes in drug2genes.items():
        for gene in genes:
            G.add_node(gene, bipartite="gene")
            G.add_edge(drug, gene)
            
    return G


def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    """Generează statistici despre numărul de ținte per medicament."""
    data = []
    for drug, genes in drug2genes.items():
        data.append({"drug": drug, "num_targets": len(genes)})
    
    df = pd.DataFrame(data).sort_values("num_targets", ascending=False)
    return df


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """Calculează similaritatea Jaccard între toate perechile de medicamente."""
    edges = []
    drugs = list(drug2genes.keys())
    
    # itertools.combinations generează perechi unice (A, B) fără (B, A) și fără (A, A)
    for d1, d2 in itertools.combinations(drugs, 2):
        s1 = drug2genes[d1]
        s2 = drug2genes[d2]
        
        sim = jaccard_similarity(s1, s2)
        if sim >= min_sim:
            edges.append((d1, d2, sim))
            
    return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    """Convertește lista de muchii în DataFrame."""
    df = pd.DataFrame(edges, columns=["drug1", "drug2", "similarity"])
    return df.sort_values("similarity", ascending=False)


def visualize_network(G: nx.Graph, out_path: Path):
    """
    Task 4: Vizualizare
    - Medicamente: Albastru
    - Gene: Roșu
    - Mărime nod: Proporțională cu gradul
    """
    print("[INFO] Generare vizualizare rețea...")
    plt.figure(figsize=(12, 12))
    
    # Separăm nodurile după tip
    drugs = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 'drug']
    genes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 'gene']
    
    # Layout (algoritmul spring aranjează nodurile conectate aproape unul de altul)
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # Dimensiunea nodurilor în funcție de numărul de conexiuni
    deg = dict(G.degree())
    
    # Desenăm medicamentele (Albastru)
    nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_color='skyblue', 
                           node_size=[deg[n] * 50 + 100 for n in drugs], label='Drugs')
    
    # Desenăm genele (Roșu)
    nx.draw_networkx_nodes(G, pos, nodelist=genes, node_color='salmon', 
                           node_size=[deg[n] * 50 + 50 for n in genes], label='Genes')
    
    # Desenăm muchiile
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Etichete (doar pentru nodurile importante/hub-uri ca să nu aglomerăm)
    labels = {n: n for n in G.nodes() if deg[n] > 2} 
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Drug-Gene Bipartite Network")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # TODO 1: verificați input-urile
    ensure_exists(DRUG_GENE_CSV)

    # TODO 2: încărcați tabelul drug-gene
    df = load_drug_gene_table(DRUG_GENE_CSV)
    
    # TODO 3: construiți mapping-ul drug -> set de gene
    drug2genes = build_drug2genes(df)
    
    # TODO 4: construiți graful bipartit
    G = build_bipartite_graph(drug2genes)
    print(f"[INFO] Graf construit: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii.")
    
    # (Optional) Export Gpickle pentru reutilizare rapida
    # nx.write_gpickle(G, OUT_GRAPH_DRUG_GENE)

    # Vizualizare (Task 4 din cerințe)
    visualize_network(G, OUT_GRAPH_IMG)
    print(f"[INFO] Imagine salvată în {OUT_GRAPH_IMG}")

    # TODO 5: generați și salvați sumarul pe medicamente
    summary_df = summarize_drugs(drug2genes)
    summary_df.to_csv(OUT_DRUG_SUMMARY, index=False)
    print(f"[INFO] Sumar medicamente salvat în {OUT_DRUG_SUMMARY}")

    # TODO 6: calculați similaritatea între medicamente
    print("[INFO] Calculare similaritate Jaccard...")
    edges = compute_drug_similarity_edges(drug2genes, min_sim=0.0)
    sim_df = edges_to_dataframe(edges)
    sim_df.to_csv(OUT_DRUG_SIMILARITY, index=False)
    print(f"[INFO] Similaritate salvată în {OUT_DRUG_SIMILARITY}")

    print("[SUCCESS] Exercițiul 9.1 finalizat.")