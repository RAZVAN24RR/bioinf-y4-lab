"""
Exercise 10.2 — Identify top SNP–Gene correlations (Cross-Omics Markers)
Creat pentru: RAZVAN24RR
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- CONFIGURARE ---
HANDLE = "RAZVAN24RR"

# Input: Matricea concatenata generata de ex01
INPUT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/multiomics_concat_{HANDLE}.csv")

# Output
OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PAIRS = OUT_DIR / f"snp_gene_pairs_{HANDLE}.csv"

def main():
    print(f"--- Cross-Omics Correlation Analysis: {HANDLE} ---")

    # 1. Incarcare date integrate
    if not INPUT_CSV.exists():
        print(f"EROARE: Nu gasesc {INPUT_CSV}. Ruleaza intai ex01!")
        return

    print("1. Incarcare matrice Multi-Omics...")
    df_joint = pd.read_csv(INPUT_CSV, index_col=0)
    print(f"   Dimensiune totala: {df_joint.shape} (Features x Samples)")

    # 2. Separare SNP vs Gene
    # Stim din setup ca SNP-urile incep cu "rs" si Genele cu "GENE"
    snp_rows = [idx for idx in df_joint.index if idx.startswith("rs")]
    gene_rows = [idx for idx in df_joint.index if idx.startswith("GENE")]

    if not snp_rows or not gene_rows:
        print("EROARE: Nu pot distinge SNP-urile de Gene (verificati numele randurilor).")
        return

    print(f"   Identificat: {len(snp_rows)} SNPs si {len(gene_rows)} Gene.")

    # 3. Calcul Corelatii (All-vs-All)
    print("2. Calcul matrice de corelatie (Pearson)...")
    
    # Transpunem (Samples x Features) pentru ca .corr() lucreaza pe coloane
    # Calculam corelatia completa (SNP+Gene vs SNP+Gene)
    corr_matrix_full = df_joint.T.corr()
    
    # Extragem doar dreptunghiul de interes: Randuri=SNPs, Coloane=Gene
    # Ne intereseaza relatia SNP -> Gena, nu SNP-SNP sau Gena-Gena
    corr_cross = corr_matrix_full.loc[snp_rows, gene_rows]
    
    print(f"   Matrice cross-omics calculata: {corr_cross.shape}")

    # 4. Filtrare si Clasament
    print("3. Filtrare perechi puternice (|r| > 0.5)...")
    
    # "Despaturim" matricea (stack) pentru a obtine o lista lunga de perechi
    pairs = corr_cross.unstack().reset_index()
    pairs.columns = ['Gene', 'SNP', 'Correlation']
    
    # Filtram pragul de relevanta
    strong_pairs = pairs[pairs['Correlation'].abs() > 0.5].copy()
    
    # Sortam descrescator dupa puterea corelatiei (valoare absoluta)
    strong_pairs['Abs_Corr'] = strong_pairs['Correlation'].abs()
    strong_pairs = strong_pairs.sort_values('Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])
    
    # 5. Export
    strong_pairs.to_csv(OUT_PAIRS, index=False)
    
    print(f"[SUCCESS] {len(strong_pairs)} perechi identificate.")
    print(f"          Salvat in: {OUT_PAIRS.name}")
    print("\nTop 5 Perechi Candidate (Biomarkeri):")
    print(strong_pairs.head(5))

if __name__ == "__main__":
    main()