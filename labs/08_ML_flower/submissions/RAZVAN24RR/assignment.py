"""
Laborator 8 - Tasks 4 & 5: Unsupervised & Semi-Supervised Learning
Creat pentru: RAZVAN24RR
Optimizat pentru memorie (Codespace safe) - Versiune Clean
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURARE ---
HANDLE = "RAZVAN24RR"
DATA_CSV = Path(f"data/work/{HANDLE}/lab06/expression_matrix.csv")
OUT_DIR = Path(f"labs/08_ml/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parametri de siguranta
TOP_GENES = 1000
RANDOM_STATE = 42

def load_safe_data(path: Path):
    """
    Incarca datele eficient: doar top 1000 gene + eliminare clase mici.
    """
    print("Status: Initializare incarcare date...")
    if not path.is_file():
        raise FileNotFoundError(f"Nu gasesc fisierul: {path}")

    # 1. Identificam coloanele (fara a incarca datele)
    header = pd.read_csv(path, nrows=0)
    cols = [header.columns[0]] + list(header.columns[1:TOP_GENES+1]) + [header.columns[-1]]
    
    # 2. Incarcam selectiv
    print(f"Status: Incarcam {len(cols)} coloane...")
    df = pd.read_csv(path, usecols=cols, index_col=0)
    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1]

    # 3. Filtram clasele cu prea putine probe (< 2)
    counts = y.value_counts()
    valid_classes = counts[counts > 1].index
    mask = y.isin(valid_classes)
    
    X = X[mask]
    y = y[mask]
    
    print(f"Status: Date pregatite: {X.shape} probe.")
    return X, y

def task_unsupervised(X, y, n_classes):
    """
    Task 4: PCA + KMeans
    """
    print("\n--- Start Task 4: Unsupervised (PCA + KMeans) ---")
    
    # 1. Scalare (Obligatorie pt PCA/KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA pentru vizualizare
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. KMeans
    print("Status: Rulare KMeans...")
    kmeans = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # 4. Vizualizare Scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"PCA Projection - KMeans Clusters ({HANDLE})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    out_png = OUT_DIR / f"sup_vs_unsup_scatter_{HANDLE}.png"
    plt.savefig(out_png)
    plt.close() # Eliberam memoria
    print(f"Imagine salvata: {out_png.name}")

    # 5. Crosstab (Validare)
    # Folosim etichetele originale (string) pentru lizibilitate
    ctab = pd.crosstab(y, clusters, rownames=['True Label'], colnames=['Cluster'])
    out_csv = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"
    ctab.to_csv(out_csv)
    print(f"Crosstab salvat: {out_csv.name}")

def task_semisupervised(X, y_enc):
    """
    Task 5: Semi-Supervised Learning (Simulation)
    """
    print("\n--- Start Task 5: Semi-Supervised Experiment ---")
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    
    results_txt = []
    
    # --- SCENARIUL 1: Baseline (50% date etichetate) ---
    # Simulam ca am pierdut jumatate din etichete
    rng = np.random.RandomState(RANDOM_STATE)
    mask_labeled = rng.rand(len(X_train)) < 0.5 
    
    X_labeled = X_train[mask_labeled]
    y_labeled = y_train[mask_labeled]
    
    print(f"1. Antrenare Baseline pe {len(X_labeled)} probe...")
    rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
    rf_base.fit(X_labeled, y_labeled)
    
    acc_base = accuracy_score(y_test, rf_base.predict(X_test))
    res1 = f"Baseline (50% labels): Accuracy = {acc_base:.4f}"
    print(res1)
    results_txt.append(res1)

    # --- SCENARIUL 2: Pseudo-Labeling ---
    # Folosim modelul slab sa ghiceasca etichetele lipsa
    X_unlabeled = X_train[~mask_labeled]
    
    if len(X_unlabeled) > 0:
        pseudo_labels = rf_base.predict(X_unlabeled)
        
        # Combinam: Date Reale + Date 'Ghicite'
        X_augmented = pd.concat([X_labeled, X_unlabeled])
        y_augmented = np.concatenate([y_labeled, pseudo_labels])
        
        print(f"2. Antrenare Semi-Supervised pe {len(X_augmented)} probe (Real+Pseudo)...")
        rf_semi = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
        rf_semi.fit(X_augmented, y_augmented)
        
        acc_semi = accuracy_score(y_test, rf_semi.predict(X_test))
        res2 = f"Semi-Supervised (Pseudo-Labeling): Accuracy = {acc_semi:.4f}"
        print(res2)
        results_txt.append(res2)
        
        diff = acc_semi - acc_base
        res3 = f"Improvement: {diff:+.4f}"
        results_txt.append(res3)
    else:
        results_txt.append("Nu au existat date unlabeled (split-ul a pastrat tot).")

    # Salvare raport text
    out_file = OUT_DIR / "semi_supervised_results.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(results_txt))
    print(f"Raport salvat: {out_file.name}")

def main():
    try:
        # 1. Incarcare
        X, y = load_safe_data(DATA_CSV)
        
        # 2. Encodare pentru modele
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # 3. Rulare Task-uri
        task_unsupervised(X, y, n_classes)
        task_semisupervised(X, y_enc)
        
        print(f"\nSucces! Toate fisierele sunt in {OUT_DIR}")
        
    except Exception as e:
        print(f"\nEroare critica: {e}")

if __name__ == "__main__":
    main()