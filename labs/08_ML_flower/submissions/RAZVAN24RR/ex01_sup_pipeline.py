"""
Exercise 8 — Supervised ML pipeline pentru expresie genică (Random Forest)
Creat pentru: RAZVAN24RR
Fix Final: Gestionarea claselor lipsă din setul de test (labels parameter)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURARE ---
HANDLE = "RAZVAN24RR"
DATA_CSV = Path(f"data/work/{HANDLE}/lab06/expression_matrix.csv")
OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parametri
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100 
TOP_GENES = 1000 # Limitare RAM

def main():
    print(f"--- Pipeline ML pentru {HANDLE} ---")
    
    # 1. Incarcare eficienta (Memory Safe)
    if not DATA_CSV.is_file():
        print(f"Eroare: Nu gasesc fisierul {DATA_CSV}"); return

    # Citim doar primele 1000 gene + Label
    header = pd.read_csv(DATA_CSV, nrows=0)
    cols = [header.columns[0]] + list(header.columns[1:TOP_GENES+1]) + [header.columns[-1]]
    
    df = pd.read_csv(DATA_CSV, usecols=cols, index_col=0)
    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1]
    
    # 2. Filtrare clase foarte mici (singleton)
    # Daca o clasa are < 2 probe, nu putem face stratificare
    counts = y.value_counts()
    valid_classes = counts[counts > 1].index
    
    # Pastram doar datele din clasele valide
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"Date incarcate si filtrate: {X.shape}")

    # 3. Encodare
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Convertim numele claselor in string pentru a evita erorile de tip TypeError
    class_names = [str(c) for c in le.classes_]
    all_labels_ids = range(len(class_names)) # ID-urile tuturor claselor (0, 1, ..., 17)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    # 5. Antrenare (n_jobs=1 pentru Codespace)
    print("Antrenare Random Forest...")
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)

    # 6. Raportare (AICI ESTE FIX-UL)
    print("Generare rapoarte...")
    y_pred = rf.predict(X_test)

    # Parametrul 'labels' forțează raportul să includă toate cele 18 clase,
    # chiar dacă una lipsește din y_test (va avea scor 0, dar nu va da eroare).
    report = classification_report(
        y_test, 
        y_pred, 
        labels=all_labels_ids, 
        target_names=class_names, 
        zero_division=0
    )
    (OUT_DIR / f"classification_report_{HANDLE}.txt").write_text(report)

    # Matricea de confuzie
    cm = confusion_matrix(y_test, y_pred, labels=all_labels_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {HANDLE}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_rf_{HANDLE}.png")
    plt.close()

    # 7. Feature Importance
    df_imp = pd.DataFrame({"Gene": X.columns, "Importance": rf.feature_importances_})
    df_imp.sort_values("Importance", ascending=False).to_csv(OUT_DIR / f"feature_importance_{HANDLE}.csv", index=False)

    # 8. Unsupervised (KMeans)
    print("Rulare KMeans...")
    kmeans = KMeans(n_clusters=len(class_names), random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X)
    pd.crosstab(y, clusters).to_csv(OUT_DIR / f"cluster_crosstab_{HANDLE}.csv")

    print(f"Gata! Rezultate salvate in {OUT_DIR}")

if __name__ == "__main__":
    main()