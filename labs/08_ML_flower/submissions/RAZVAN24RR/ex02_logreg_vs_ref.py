"""
Exercise 8b — Logistic Regression vs Random Forest
Creat pentru: RAZVAN24RR
Optimizat pentru Codespaces (RAM limited)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- CONFIGURARE ---
HANDLE = "RAZVAN24RR"
DATA_CSV = Path(f"data/work/{HANDLE}/lab06/expression_matrix.csv")
OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_COMPARE = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"

# Parametri siguranță
TOP_GENES = 1000  # Limitare RAM
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    print(f"--- Comparatie Modele ML: {HANDLE} ---")
    
    # 1. Incarcare Safe (doar top 1000 gene)
    if not DATA_CSV.is_file():
        print("Eroare: Fisier lipsa."); return

    header = pd.read_csv(DATA_CSV, nrows=0)
    cols = [header.columns[0]] + list(header.columns[1:TOP_GENES+1]) + [header.columns[-1]]
    
    print("Status: Incarcare date...")
    df = pd.read_csv(DATA_CSV, usecols=cols, index_col=0)
    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1]

    # 2. Filtrare clase mici (pentru a evita erori la split)
    counts = y.value_counts()
    valid_classes = counts[counts > 1].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    print(f"Status: Date filtrate: {X.shape}")

    # 3. Encodare
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_] # String conversion
    all_labels = range(len(class_names))

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    # 5. Scalare (CRITICA pentru Logistic Regression)
    # Random Forest nu are nevoie neaparat, dar LogReg nu merge fara ea
    print("Status: Scalare date (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model 1: Random Forest
    print("Status: Antrenare Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)
    
    # 7. Model 2: Logistic Regression
    print("Status: Antrenare Logistic Regression...")
    # n_jobs=1 pentru a nu bloca RAM-ul
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=1)
    lr.fit(X_train_scaled, y_train)

    # 8. Generare Rapoarte
    print("Status: Generare raport comparativ...")
    
    # Predictii
    y_pred_rf = rf.predict(X_test)
    y_pred_lr = lr.predict(X_test_scaled)

    # Configurare raport (safe mode)
    report_kwargs = {
        "target_names": class_names,
        "labels": all_labels,
        "zero_division": 0
    }

    report_rf = classification_report(y_test, y_pred_rf, **report_kwargs)
    report_lr = classification_report(y_test, y_pred_lr, **report_kwargs)

    # Salvare
    with open(OUT_COMPARE, "w") as f:
        f.write(f"=== REZULTATE COMPARATIVE - {HANDLE} ===\n")
        f.write(f"Data: {pd.Timestamp.now()}\n")
        f.write(f"Gene folosite: {TOP_GENES}\n\n")
        
        f.write("########################################\n")
        f.write("### 1. RANDOM FOREST (Non-Liniar)    ###\n")
        f.write("########################################\n")
        f.write(report_rf)
        f.write("\n\n")
        
        f.write("########################################\n")
        f.write("### 2. LOGISTIC REGRESSION (Liniar)  ###\n")
        f.write("########################################\n")
        f.write(report_lr)
    
    print(f"Gata! Raportul este salvat in: {OUT_COMPARE}")

if __name__ == "__main__":
    main()