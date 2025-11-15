#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rezolvarea Laboratorului 2 (Alignment) - Task 1 și Task 2

Acest script încarcă secvențe dintr-un fișier FASTA din Lab 1
și realizează:
1. Task 1: Calcularea unei matrici p-distance pentru 3 secvențe.
2. Task 2: Compararea aliniamentelor global (NW) vs. local (SW)
   folosind Biopython.

Autor: RAZVAN24RR - Pasaran Razvan Andrei
"""

from pathlib import Path
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import itertools 

# --- Configurare Globală ---
HANDLE = "RAZVAN24RR"
# Setăm calea la fișierul cu datele tale (CYTB de la 3 specii)
FASTA_FILE_PATH = Path(f"data/work/{HANDLE}/lab01/lab02_final.fa") 
MAX_LENGTH = 5000 # Lungimea maximă la care trunchiem secvențele

# --- Funcție Helper pentru Task 1: p-distance ---

def calculate_p_distance(seq1, seq2):
    """
    Calculează p-distance bazată pe o aliniere globală (globalxx).
    p-distance = (număr mismatch-uri) / (lungimea aliniamentului fără gap-uri în ambele)
    """
    
    try:
        # Folosim globalxx pentru a obține o aliniere simplă
        alignment = pairwise2.align.globalxx(seq1, seq2)[0]
    except IndexError:
        print(f"Eroare: Nu s-a putut alinia una din secvențe.")
        return 0.0

    align1, align2, score, _, _ = alignment
    
    mismatches = 0
    comparisons = 0 
    
    for a, b in zip(align1, align2):
        # Ignorăm gap-urile
        if a == '-' or b == '-':
            continue
            
        comparisons += 1
        if a != b:
            mismatches += 1
            
    if comparisons == 0:
        return 0.0
        
    return mismatches / comparisons

# --- Task 1: Matricea de Distanțe ---

def run_task_1(sequences):
    """
    Rulează Task 1: Calculează matricea p-distance pentru 3+ secvențe.
    """
    print("=" * 30)
    print("TASK 1: Matricea P-Distance")
    print("=" * 30)
    
    if len(sequences) < 3:
        print(f"Atenție: Task 1 necesită cel puțin 3 secvențe. Ai doar {len(sequences)}.")
        return

    indices_task_1 = [0, 1, 2]
    seqs_task_1 = [sequences[i] for i in indices_task_1]
    ids_task_1 = [seq.id for seq in seqs_task_1]
    
    print(f"Se analizează secvențele: {', '.join(ids_task_1)}\n")
    
    distance_matrix = {}
    
    for (seq_a, seq_b) in itertools.combinations(seqs_task_1, 2):
        # APLICĂM TRUNCHIEREA
        s1_str = str(seq_a.seq)[:MAX_LENGTH]
        s2_str = str(seq_b.seq)[:MAX_LENGTH]
        
        if len(s1_str) < 100 or len(s2_str) < 100:
             print(f"Atenție: Secvența {seq_a.id} sau {seq_b.id} este prea scurtă după trunchiere.")
             p_dist = 0.0
        else:
             p_dist = calculate_p_distance(s1_str, s2_str)
        
        if seq_a.id not in distance_matrix:
            distance_matrix[seq_a.id] = {}
        distance_matrix[seq_a.id][seq_b.id] = p_dist

    # Afișăm matricea (triunghiul superior)
    header = "\t" + "\t".join(ids_task_1[1:])
    print(header)
    
    for i in range(len(ids_task_1) - 1):
        row_id = ids_task_1[i]
        row_str = f"{row_id}"
        for j in range(i + 1, len(ids_task_1)):
            col_id = ids_task_1[j]
            dist = distance_matrix[row_id][col_id]
            row_str += f"\t{dist:.4f}"
        print(row_str)
        
    print(f"\n[ATENȚIE: Secvențele au fost trunchiate la {MAX_LENGTH} baze pentru a preveni 'Segmentation Fault']\n")

# --- Task 2: Global vs. Local ---

def run_task_2(sequences):
    """
    Rulează Task 2: Compară alinierea globală cu cea locală.
    """
    print("=" * 30)
    print("TASK 2: Global (NW) vs. Local (SW)")
    print("=" * 30)
    
    if len(sequences) < 2:
        print(f"Atenție: Task 2 necesită cel puțin 2 secvențe.")
        return
        
    s1_rec = sequences[0]
    s2_rec = sequences[1]
    
    # APLICĂM TRUNCHIEREA
    s1 = str(s1_rec.seq)[:MAX_LENGTH]
    s2 = str(s2_rec.seq)[:MAX_LENGTH]
    
    id1 = s1_rec.id
    id2 = s2_rec.id

    print(f"Se compară {id1} (Lungime: {len(s1)}) vs {id2} (Lungime: {len(s2)}) [Trunchiate la {MAX_LENGTH}]\n")
    
    # Setări de scorare customizate
    MATCH_SCORE = 2
    MISMATCH_PENALTY = -1
    GAP_OPEN_PENALTY = -0.5
    GAP_EXTEND_PENALTY = -0.1
    
    # 1. Aliniere GLOBALĂ (Needleman-Wunsch)
    print(f"--- [Global (globalms)] ---")
    global_alignments = pairwise2.align.globalms(
        s1, s2, MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    )
    
    if global_alignments:
        best_global = global_alignments[0]
        print(f"Cea mai bună aliniere GLOBALĂ (Scor: {best_global.score:.2f}):")
        print(format_alignment(*best_global)) 
    else:
        print("Nu s-a găsit nicio aliniere globală.")

    # 2. Aliniere LOCALĂ (Smith-Waterman)
    print(f"\n--- [Local (localms)] ---")
    local_alignments = pairwise2.align.localms(
        s1, s2, MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    )
    
    if local_alignments:
        best_local = local_alignments[0]
        print(f"Cea mai bună aliniere LOCALĂ (Scor: {best_local.score:.2f}):")
        print(format_alignment(*best_local))
    else:
        print("Nu s-a găsit nicio aliniere locală.")
        

# --- MAIN ---

def main():
    if not FASTA_FILE_PATH.exists():
        print(f"EROARE CRITICĂ: Fișierul FASTA nu a fost găsit la:")
        print(f"{FASTA_FILE_PATH.resolve()}")
        print("\nVerificați calea sau asigurați-vă că fișierul Lab 1 este numit 'lab02_final.fa'.")
        return

    try:
        all_sequences = list(SeqIO.parse(FASTA_FILE_PATH, "fasta"))
        if len(all_sequences) < 2:
             print(f"EROARE: Fișierul FASTA trebuie să conțină cel puțin 2 secvențe. S-au găsit {len(all_sequences)}.")
             return
    except Exception as e:
        print(f"EROARE la citirea fișierului FASTA: {e}")
        return
        
    print(f"S-au încărcat {len(all_sequences)} secvențe din {FASTA_FILE_PATH.name}")
    print(f"ATENȚIE: Secvențele lungi sunt trunchiate la {MAX_LENGTH} baze.\n")

    run_task_1(all_sequences)
    run_task_2(all_sequences)

if __name__ == "__main__":
    main()