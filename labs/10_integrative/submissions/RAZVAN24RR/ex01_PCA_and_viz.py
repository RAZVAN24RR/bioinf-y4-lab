"""
Exercise 10.1 — PCA Single-Omics vs Joint
Creat pentru: RAZVAN24RR
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURARE ---
HANDLE = "RAZVAN24RR"

# Input paths
DATA_DIR = Path(f"data/work/{HANDLE}/lab10")
SNP_CSV = DATA_DIR / f"snp_matrix_{HANDLE}.csv"
EXP_CSV = DATA_DIR / f"expression_matrix_{HANDLE}.csv"

# Output paths
OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONCAT = OUT_DIR / f"multiomics_concat_{HANDLE}.csv"
OUT_IMG_SNP = OUT_DIR / f"pca_snp_{HANDLE}.png"
OUT_IMG_EXP = OUT_DIR / f"pca_expr_{HANDLE}.png"
OUT_IMG_JOINT = OUT_DIR / f"pca_joint_{HANDLE}.png"

def run_pca_and_plot(df: pd.DataFrame, title: str, out_path: Path, color: str):
    """
    Primeste un DataFrame (Features x Samples), face PCA și salvează plot-ul.
    """
    # PCA in sklearn cere (Samples x Features), deci facem Transpose (.T)
    X = df.T 
    
    # Rulam PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    
    # Vizualizare
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=color, edgecolor='k', s=60, alpha=0.7)
    
    # Adnotam varianta explicata
    var_exp = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Salvat: {out_path.name}")

def main():
    print(f"--- Multi-Omics PCA Pipeline: {HANDLE} ---")
    
    # 1. Incarcare date
    if not SNP_CSV.exists() or not EXP_CSV.exists():
        print("EROARE: Nu gasesc fisierele de input. Rulati intai setup_lab10.py!")
        return

    print("1. Incarcare date...")
    df_snp = pd.read_csv(SNP_CSV, index_col=0)
    df_exp = pd.read_csv(EXP_CSV, index_col=0)
    
    # 2. Aliniere probe (Intersectia coloanelor)
    common_samples = df_snp.columns.intersection(df_exp.columns)
    df_snp = df_snp[common_samples]
    df_exp = df_exp[common_samples]
    print(f"   Probe comune identificate: {len(common_samples)}")

    # 3. Normalizare (Z-score pe randuri/features)
    # (Valoare - Medie_Gena) / Std_Gena
    print("2. Normalizare Z-score...")
    
    # Functie helper pentru normalizare pe randuri
    def normalize_rows(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0).fillna(0)

    df_snp_norm = normalize_rows(df_snp)
    df_exp_norm = normalize_rows(df_exp)

    # 4. Concatenare (Joint Matrix)
    # Concatenam pe verticala (axis=0) -> Features devin mai multe, Samples raman aceleasi
    df_joint = pd.concat([df_snp_norm, df_exp_norm], axis=0)
    
    # Salvam matricea concatenata pentru Exercițiul 2
    df_joint.to_csv(OUT_CONCAT)
    print(f"   Matrice integrata salvata: {df_joint.shape} in {OUT_CONCAT.name}")

    # 5. Rulare PCA si Generare Figuri
    print("3. Generare PCA Plots...")
    
    # A) PCA pe SNP
    run_pca_and_plot(df_snp_norm, f"PCA - SNP Layer ({HANDLE})", OUT_IMG_SNP, "skyblue")
    
    # B) PCA pe Expression
    run_pca_and_plot(df_exp_norm, f"PCA - Expression Layer ({HANDLE})", OUT_IMG_EXP, "salmon")
    
    # C) PCA Joint
    run_pca_and_plot(df_joint, f"PCA - Joint Multi-Omics ({HANDLE})", OUT_IMG_JOINT, "purple")

    print("\n[SUCCESS] Exercițiul 1 finalizat.")

if __name__ == "__main__":
    main()