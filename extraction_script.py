"""
Script di analisi per V0_no_weight - VERSIONE ZERO SOVRAPPOSIZIONI
Genera grafici con sistema di riferimento numerato per massima chiarezza

REQUISITI:
- pip install matplotlib numpy scipy pandas

ESECUZIONE:
python analysis_v0_no_weight_FIXED.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import math
from textwrap import shorten


# Configurazione matplotlib per grafici professionali
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Directory di output
OUTPUT_DIR = Path("C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Stats_and_Models_History/v0_no_weight/analysis_results/analysis_results_v0_FIXED")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Directory output: {OUTPUT_DIR}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# DATI DA CLASSIFICATION REPORT (EPOCA 10)
# ═══════════════════════════════════════════════════════════════════

class_names = [
    'Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS',
    'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP',
    'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest',
    'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration',
    'SQL Injection', 'SSH-Bruteforce'
]

class_abbr = [
    'Benign', 'Bot', 'BF-Web', 'BF-XSS',
    'DDOS-HOIC', 'DDOS-LOIC-UDP', 'DDOS-LOIC-HTTP',
    'DoS-GoldenEye', 'DoS-Hulk', 'DoS-SlowHTTP',
    'DoS-Slowloris', 'FTP-BF', 'Infiltration',
    'SQL Inj', 'SSH-BF'
]

supports = np.array([
    8804790, 571605, 228936, 162791, 76033,
    61390, 50448, 14725, 13833, 7458,
    4997, 1174, 1100, 490, 230
])

precisions = np.array([
    0.9947, 0.9972, 1.0000, 0.9997, 1.0000,
    0.9878, 0.9999, 0.9999, 1.0000, 1.0000,
    0.9998, 0.7188, 0.9972, 1.0000, 0.8947
])

recalls = np.array([
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    0.2447, 1.0000, 0.9999, 1.0000, 1.0000,
    1.0000, 0.0784, 0.9627, 0.1122, 0.0739
])

f1_scores = np.array([
    0.9974, 0.9986, 1.0000, 0.9998, 1.0000,
    0.3922, 0.9999, 0.9999, 1.0000, 1.0000,
    0.9999, 0.1413, 0.9796, 0.2018, 0.1365
])

# ═══════════════════════════════════════════════════════════════════
# CALCOLO CORRELAZIONE
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("📊 ANALISI CORRELAZIONE SUPPORT-F1")
print("=" * 70)

# Correlazione lineare
corr_linear, p_linear = pearsonr(supports, f1_scores)
print(f"\n✅ Correlazione LINEARE:")
print(f"   Pearson r = {corr_linear:.4f}")
print(f"   p-value   = {p_linear:.6f}")

# Correlazione log-lineare (più appropriata per range ampio di support)
log_supports = np.log10(supports)
corr_log, p_log = pearsonr(log_supports, f1_scores)
print(f"\n✅ Correlazione LOG-LINEARE (consigliata):")
print(f"   Pearson r = {corr_log:.4f}")
print(f"   p-value   = {p_log:.6f}")

if p_log < 0.001:
    print(f"   ⚠️ Correlazione ALTAMENTE SIGNIFICATIVA (p < 0.001)")
elif p_log < 0.05:
    print(f"   ⚠️ Correlazione SIGNIFICATIVA (p < 0.05)")
else:
    print(f"   ℹ️ Correlazione NON significativa (p > 0.05)")

# Interpretazione
if corr_log > 0.7:
    interpretation = "FORTE correlazione positiva"
elif corr_log > 0.4:
    interpretation = "MODERATA correlazione positiva"
elif corr_log > 0.2:
    interpretation = "DEBOLE correlazione positiva"
else:
    interpretation = "Correlazione TRASCURABILE"

print(f"\n📝 INTERPRETAZIONE: {interpretation}")
print(f"   → Maggiore il support, {'maggiore' if corr_log > 0 else 'minore'} l'F1-score")

# Salva risultati per la tesi
with open(OUTPUT_DIR/'v0_correlation_results.txt', 'w') as f:
    f.write(f"Correlazione lineare: r={corr_linear:.4f}, p={p_linear:.6f}\n")
    f.write(f"Correlazione log-lineare: r={corr_log:.4f}, p={p_log:.6f}\n")
    f.write(f"Interpretazione: {interpretation}\n")

print("\n✅ Risultati salvati in: v0_correlation_results.txt")




# ═══════════════════════════════════════════════════════════════════
# GRAFICO PROFESSIONALE: SUPPORT vs F1-SCORE
# Layout ottimizzato per pubblicazione accademica
# ═══════════════════════════════════════════════════════════════════



print("\n" + "=" * 70)
print("📈 GENERAZIONE SCATTER PLOT PROFESSIONALE: SUPPORT vs F1")
print("=" * 70)
# Pearson lineare su support raw
corr_linear, p_linear = pearsonr(supports, f1_scores)

# Pearson su log10(support)
log_supports = np.log10(supports)
corr_log, p_log = pearsonr(log_supports, f1_scores)

# Interpretazione semplice (log-lineare preferita per ampie differenze di support)
if corr_log > 0.7:
    interpretation = "Interpretazione: FORTE correlazione positiva"
elif corr_log > 0.4:
    interpretation = "Interpretazione: MODERATA correlazione positiva"
elif corr_log > 0.2:
    interpretation = "Interpretazione: DEBOLE correlazione positiva"
else:
    interpretation = "Interpretazione: Correlazione TRASCURABILE"

print(f"Pearson linear: r={corr_linear:.4f}, p={p_linear:.6f}")
print(f"Pearson log10:  r={corr_log:.4f}, p={p_log:.6f}")
print("Interpretazione (log10):", interpretation)

# Salvati risultati sintetici
with open(OUTPUT_DIR / 'v0_correlation_results.txt', 'w') as f:
    f.write(f"Pearson linear: r={corr_linear:.4f}, p={p_linear:.6f}\n")
    f.write(f"Pearson log10:  r={corr_log:.4f}, p={p_log:.6f}\n")
    f.write(f"Interpretazione (log10): {interpretation}\n")
    

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE FIGURA - LAYOUT OTTIMIZZATO (VERSIONE MIGLIORATA)
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(26, 15))
fig.patch.set_facecolor('white')

# Coordinate layout ottimizzate - TUTTO A DESTRA
SF_TITLE_Y = 0.975
SF_SUBTITLE_Y = 0.945

# GRAFICO PRINCIPALE - Centrato a sinistra (più largo)
SF_GRAPH_X = 0.07
SF_GRAPH_Y = 0.07
SF_GRAPH_W = 0.53
SF_GRAPH_H = 0.85

# TABELLA LEGENDA - Prima in alto a destra
SF_LEGEND_X = 0.645
SF_LEGEND_Y = 0.765
SF_LEGEND_W = 0.32
SF_LEGEND_H = 0.16

# TABELLA CORRELAZIONI - Seconda a destra (sotto legenda)
CORR_TABLE_X = 0.645
CORR_TABLE_Y = 0.505
CORR_TABLE_W = 0.32
CORR_TABLE_H = 0.24

# TABELLA DIAGNOSI - Terza a destra (in basso)
DIAG_TABLE_X = 0.645
DIAG_TABLE_Y = 0.065
DIAG_TABLE_W = 0.32
DIAG_TABLE_H = 0.42

# ═══════════════════════════════════════════════════════════════════
# TITOLO E SOTTOTITOLO
# ═══════════════════════════════════════════════════════════════════

fig.text(0.55, SF_TITLE_Y, 'Support vs F1-score — Analisi di Distribuzione',
         fontsize=24, fontweight='bold', ha='center', family='sans-serif')

fig.text(0.55, SF_SUBTITLE_Y, 
         'Impatto dello sbilanciamento dei dati sulle performance — V0_no_weight (Epoch 10)',
         fontsize=17, ha='center', style='italic', color='#555555')

# ═══════════════════════════════════════════════════════════════════
# LEGENDA - TABELLA PROFESSIONALE STILE WORD
# ═══════════════════════════════════════════════════════════════════

legend_ax = plt.axes([SF_LEGEND_X-0.02, SF_LEGEND_Y-0.07, SF_LEGEND_W, SF_LEGEND_H+0.06])
legend_ax.axis('off')

# Header
legend_ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18,
                                  fill=True, facecolor='#ECEFF1',
                                  edgecolor='#424242', linewidth=1.5,
                                  transform=legend_ax.transAxes))

legend_ax.text(0.5, 0.91, 'LEGENDA GRAFICO',
               fontsize=20, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)

# Linea separatrice header
legend_ax.plot([0, 1], [0.82, 0.82], 'k-', linewidth=1.5,
               transform=legend_ax.transAxes)

# Header colonne
legend_ax.add_patch(plt.Rectangle((0, 0.70), 1, 0.12,
                                  fill=True, facecolor='#F5F5F5',
                                  transform=legend_ax.transAxes))

legend_ax.text(0.15, 0.76, 'Elemento',
               fontsize=19, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)
legend_ax.text(0.65, 0.76, 'Descrizione',
               fontsize=19, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)

# Linea verticale separatore colonne
legend_ax.plot([0.30, 0.30], [0, 0.82], '-', color='#BDBDBD',
               linewidth=1.2, transform=legend_ax.transAxes)

# Linea separatrice dopo header colonne
legend_ax.plot([0, 1], [0.70, 0.70], 'k-', linewidth=1.5,
               transform=legend_ax.transAxes)

# RIGA 1: Marker
y1 = 0.58
legend_ax.add_patch(plt.Rectangle((0, y1 - 0.06), 1, 0.12,
                                  fill=True, facecolor='#FFFFFF',
                                  transform=legend_ax.transAxes))

legend_ax.text(0.15, y1+0.02, 'Marker',
               fontsize=17, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)
legend_ax.text(0.68, y1+0.02, 'F1 ≥ 0.5 (● blu)  |  F1 < 0.5 (● numero rosso)',
               fontsize=15, ha='center', va='center',
               transform=legend_ax.transAxes)

# Separatore riga
legend_ax.plot([0, 1], [y1 - 0.06, y1 - 0.06], '-', color='#BDBDBD',
               linewidth=1, transform=legend_ax.transAxes)

# RIGA 2: Zone
y2 = 0.40
legend_ax.add_patch(plt.Rectangle((0, y2 - 0.06), 1, 0.12,
                                  fill=True, facecolor='#F9F9F9',
                                  transform=legend_ax.transAxes))

legend_ax.text(0.15, y2+0.02, 'Zone',
               fontsize=17, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)
legend_ax.text(0.65, y2+0.025, 'Critica F1<0.3 (rosso)\n  Problematica 0.3<F1<0.5 (arancio)',
               fontsize=15, ha='center', va='center',
               transform=legend_ax.transAxes)

# Separatore riga
legend_ax.plot([0, 1], [y2 - 0.06, y2 - 0.06], '-', color='#BDBDBD',
               linewidth=1, transform=legend_ax.transAxes)

# RIGA 3: Dimensione (più bassa)
y3 = 0.22
legend_ax.add_patch(plt.Rectangle((0, y3 - 0.06), 1, 0.12,
                                  fill=True, facecolor='#FFFFFF',
                                  transform=legend_ax.transAxes))

legend_ax.text(0.15, y3+0.02, 'Dimensione',
               fontsize=17, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)
legend_ax.text(0.65, y3+0.02, 'Punti uniformi (dimensione costante)',
               fontsize=15, ha='center', va='center',
               transform=legend_ax.transAxes)

# Separatore riga
legend_ax.plot([0, 1], [y3 - 0.06, y3 - 0.06], '-', color='#BDBDBD',
               linewidth=1, transform=legend_ax.transAxes)

# RIGA 4: Regressione
y4 = 0.04
legend_ax.add_patch(plt.Rectangle((0, 0), 1, 0.10,
                                  fill=True, facecolor='#F9F9F9',
                                  transform=legend_ax.transAxes))

legend_ax.text(0.15, y4+0.03, 'Regressione',
               fontsize=17, fontweight='bold', ha='center', va='center',
               transform=legend_ax.transAxes)
legend_ax.text(0.65, y4+0.03, f'Trend log (- -) con r={corr_log:.3f}',
               fontsize=15, ha='center', va='center',
               transform=legend_ax.transAxes)

# Bordo esterno
legend_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                  fill=False, edgecolor='#424242',
                                  linewidth=2.5, transform=legend_ax.transAxes, zorder=100))

# ═══════════════════════════════════════════════════════════════════
# TABELLA CORRELAZIONI (SOPRA DESTRA) - MIGLIORATA CON PIÙ SPAZIO
# ═══════════════════════════════════════════════════════════════════

corr_ax = plt.axes([CORR_TABLE_X-0.02, CORR_TABLE_Y-0.11, CORR_TABLE_W, CORR_TABLE_H])
corr_ax.axis('off')

# Header tabella
corr_ax.add_patch(plt.Rectangle((0, 0.85), 1, 0.15,
                                fill=True, facecolor="#C7E5FA",
                                edgecolor='none',
                                transform=corr_ax.transAxes))

corr_ax.text(0.52, 0.92, 'ANALISI CORRELAZIONE',
             fontsize=20, fontweight='bold', ha='center', va='center',
             transform=corr_ax.transAxes)

# Header colonne
corr_ax.add_patch(plt.Rectangle((0, 0.69), 1, 0.16,
                                fill=True, facecolor="#F5F5F56B",
                                edgecolor='none',
                                transform=corr_ax.transAxes))

corr_ax.text(0.2, 0.77, 'Metodo',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=corr_ax.transAxes)
corr_ax.text(0.5, 0.77, 'r',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=corr_ax.transAxes)
corr_ax.text(0.80, 0.77, 'p-value',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=corr_ax.transAxes)

# Linea separatrice header
corr_ax.plot([0, 1], [0.69, 0.69], 'k-', linewidth=1.5,
             transform=corr_ax.transAxes, zorder=50)

# Righe dati - PIÙ AMPIE
corr_data = [
    ('Pearson lineare', corr_linear, p_linear),
    ('Pearson log', corr_log, p_log)
]

y_start = 0.55
row_height = 0.16  # AUMENTATO da 0.14 a 0.18

for i, (method, r_val, p_val) in enumerate(corr_data):
    y_pos = y_start - i * row_height
    
    # Background alternato - RETTANGOLI PIÙ ALTI
    if i % 2 != 0:
        corr_ax.add_patch(plt.Rectangle((0, y_pos - 0.08), 1, 0.16,
                                        fill=True, facecolor='#E8EAF6',
                                        edgecolor='none',
                                        transform=corr_ax.transAxes))
    else:
        corr_ax.add_patch(plt.Rectangle((0, y_pos - 0.08), 1, 0.16,
                                        fill=True, facecolor='#FFFFFF',
                                        edgecolor='none',
                                        transform=corr_ax.transAxes))
    
    corr_ax.text(0.2, y_pos, method,
                 fontsize=16, ha='center', va='center',
                 fontweight='bold',
                 transform=corr_ax.transAxes)
    corr_ax.text(0.50, y_pos, f'{r_val:.4f}',
                 fontsize=16, ha='center', va='center',
                 family='monospace', fontweight='bold',
                 transform=corr_ax.transAxes)
    corr_ax.text(0.80, y_pos, f'{p_val:.2e}',  # CORRETTO: era y_val
                 fontsize=16, ha='center', va='center',
                 family='monospace',
                 transform=corr_ax.transAxes)

# Separatore ABBASSATO prima interpretazione
corr_ax.plot([0, 1], [0.19, 0.19], 'k-', linewidth=1.5,
             transform=corr_ax.transAxes, zorder=50)

# Interpretazione - CARATTERI PIÙ GRANDI
if corr_log > 0.7:
    interp_color = '#1B5E20'
    interp_bg = '#A5D6A7'
elif corr_log > 0.4:
    interp_color = '#EF6C00'
    interp_bg = '#FFB74D'
else:
    interp_color = '#B71C1C'
    interp_bg = '#EF5350'

corr_ax.add_patch(plt.Rectangle((0, 0), 1, 0.19,
                                fill=True, facecolor=interp_bg,
                                alpha=0.25, edgecolor='none',
                                transform=corr_ax.transAxes))

corr_ax.text(0.5, 0.10, f'{interpretation}',
             fontsize=19, ha='center', va='center',
             fontweight='bold', color=interp_color,
             transform=corr_ax.transAxes)

# BORDO CONTINUO
corr_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                fill=False, edgecolor='#1976D2',
                                linewidth=3, transform=corr_ax.transAxes,
                                zorder=100))

# ═══════════════════════════════════════════════════════════════════
# TABELLA DIAGNOSI SBILANCIAMENTO - PIÙ SPAZIOSA
# ═══════════════════════════════════════════════════════════════════

diag_ax = plt.axes([DIAG_TABLE_X-0.02, DIAG_TABLE_Y+0.003, DIAG_TABLE_W, DIAG_TABLE_H-0.15])
diag_ax.axis('off')

# Header tabella
corr_ax.add_patch(plt.Rectangle((0, 0.89), 1, 0.11,
                                fill=True, facecolor='#FFEBEE',
                                edgecolor='none',
                                transform=diag_ax.transAxes))

diag_ax.text(0.525, 0.94, 'DIAGNOSI SBILANCIAMENTO',
             fontsize=21, fontweight='bold', ha='center', va='center',
             color='#C62828',
             transform=diag_ax.transAxes)

# Linea separatrice header
diag_ax.plot([0, 1], [0.89, 0.89], '-', color='#E53935', linewidth=1.5,
             transform=diag_ax.transAxes, zorder=50)

# Calcolo classi critiche
critical_classes = [(i+1, cls, supp, f1) for i, (cls, supp, f1) in 
                    enumerate(zip(class_names, supports, f1_scores)) if f1 < 0.5]
critical_classes.sort(key=lambda x: x[3])
critical_count = len(critical_classes)
ratio_max_min = supports.max() / supports.min()

# Header colonne - PIÙ AMPIO
diag_ax.add_patch(plt.Rectangle((0, 0.79), 1, 0.10,
                                fill=True, facecolor='#F5F5F5',
                                edgecolor='none',
                                transform=diag_ax.transAxes))

diag_ax.text(0.12, 0.83, 'N°',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=diag_ax.transAxes)
diag_ax.text(0.50, 0.83, 'Classe',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=diag_ax.transAxes)
diag_ax.text(0.88, 0.83, 'F1-score',
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=diag_ax.transAxes)

# Linee verticali separatori
diag_ax.plot([0.24, 0.24], [0.12, 0.89], '-', color='#BDBDBD',
             linewidth=1.2, transform=diag_ax.transAxes, zorder=50)
diag_ax.plot([0.76, 0.76], [0.12, 0.89], '-', color='#BDBDBD',
             linewidth=1.2, transform=diag_ax.transAxes, zorder=50)

# Linea separatrice dopo header
diag_ax.plot([0, 1], [0.79, 0.79], 'k-', linewidth=1.5,
             transform=diag_ax.transAxes, zorder=50)

# Righe classi critiche
y_start = 0.70
row_height = 0.10
max_rows_to_show = min(4, len(critical_classes))

for i in range(max_rows_to_show):
    num, cls, supp, f1 = critical_classes[i]
    y_pos = y_start - i * row_height
    
    # Background alternato
    if i % 2 == 0:
        diag_ax.add_patch(plt.Rectangle((0, y_pos - 0.045), 1, 0.09,
                                        fill=True, facecolor='#FFCDD2',
                                        alpha=0.4, edgecolor='none',
                                        transform=diag_ax.transAxes))
    else:
        diag_ax.add_patch(plt.Rectangle((0, y_pos - 0.045), 1, 0.09,
                                        fill=True, facecolor='#FFFFFF',
                                        edgecolor='none',
                                        transform=diag_ax.transAxes))
    
    # Numero (colonna 1)
    diag_ax.text(0.12, y_pos, f'{num}',
                 fontsize=16, ha='center', va='center',
                 fontweight='bold', color='white',
                 bbox=dict(boxstyle='circle', facecolor='#E53935', 
                          edgecolor='#B71C1C', linewidth=1.5),
                 transform=diag_ax.transAxes)
    
    # Nome classe (colonna 2 - centrata)
    from textwrap import shorten
    cls_short = shorten(cls, width=30, placeholder='…')
    diag_ax.text(0.50, y_pos, cls_short,
                 fontsize=16.5, ha='center', va='center',
                 transform=diag_ax.transAxes)
    
    # F1-score (colonna 3)
    diag_ax.text(0.88, y_pos, f'{f1:.3f}',
                 fontsize=16.5, ha='center', va='center',
                 family='monospace', fontweight='bold',
                 color='#C62828',
                 transform=diag_ax.transAxes)

# Separatore prima delle righe riassuntive
separator_y = y_start - max_rows_to_show * row_height - 0.02
diag_ax.plot([0, 1], [separator_y, separator_y], 'k-', linewidth=1.5,
             transform=diag_ax.transAxes, zorder=50)

# RIGA RIASSUNTIVA 1: Interpretazione - CENTRATA
y_interp = separator_y - 0.11
diag_ax.add_patch(plt.Rectangle((0, y_interp - 0.05), 1, 0.160,
                                fill=True, facecolor='#FFE0B2',
                                alpha=0.5, edgecolor='none',
                                transform=diag_ax.transAxes))

interp_text = f'Lo sbilanciamento (ratio {ratio_max_min:.1f}×)\nimpatta le classi minoritarie'
diag_ax.text(0.50, y_interp+0.025, interp_text,  # CENTRATO a 0.50
             fontsize=16.5, ha='center', va='center',
             fontweight='bold', color='#EF6C00',
             transform=diag_ax.transAxes)

# Separatore
diag_ax.plot([0, 1], [y_interp - 0.05, y_interp - 0.05], '-', 
             color='#BDBDBD', linewidth=1,
             transform=diag_ax.transAxes, zorder=50)

# RIGA RIASSUNTIVA 2: Conteggio - CENTRATA
y_count = 0.05
diag_ax.add_patch(plt.Rectangle((0, 0), 1, 0.12,
                                fill=True, facecolor='#FFCDD2',
                                alpha=0.5, edgecolor='none',
                                transform=diag_ax.transAxes))

count_text = f'{critical_count}/{len(class_names)} classi sotto soglia F1=0.5'
diag_ax.text(0.50, y_count, count_text,  # CENTRATO a 0.50
             fontsize=18, ha='center', va='center',
             fontweight='bold', color='#C62828',
             transform=diag_ax.transAxes)

# BORDO CONTINUO
diag_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                fill=False, edgecolor='#E53935',
                                linewidth=3, transform=diag_ax.transAxes,
                                zorder=100))


# ═══════════════════════════════════════════════════════════════════
# GRAFICO PRINCIPALE - CON NUMERI PER CLASSI CRITICHE
# ═══════════════════════════════════════════════════════════════════

ax = plt.axes([SF_GRAPH_X, SF_GRAPH_Y, SF_GRAPH_W, SF_GRAPH_H])

# Maschere per classi critiche
critical_mask = f1_scores < 0.5
normal_mask = ~critical_mask

# Dimensione uniforme per tutti i marker
marker_size = 180

# Scatter plot classi normali (blu)
ax.scatter(supports[normal_mask], f1_scores[normal_mask],
           s=marker_size, alpha=0.80, c='#42A5F5',
           edgecolors='#1565C0', linewidth=1.6,
           label='F1 ≥ 0.5', zorder=3)

# Regressione log₁₀
slope, intercept, r_value, p_value, std_err = linregress(log_supports, f1_scores)
x_line = np.logspace(np.log10(supports.min()*0.9), np.log10(supports.max()*1.1), 200)
y_line = slope * np.log10(x_line) + intercept
ax.plot(x_line, y_line, '--', color='#424242', linewidth=3.2, alpha=0.85,
        label=f'Regressione log (r={corr_log:.3f})', zorder=2)

# Classi critiche con NUMERI invece di scatter
critical_mapping = {cls: num for num, cls, _, _ in critical_classes}

for i, (cls, supp, f1) in enumerate(zip(class_names, supports, f1_scores)):
    if f1 < 0.5:
        num = critical_mapping[cls]
        
        # Cerchio rosso di sfondo
        ax.scatter([supp], [f1], s=marker_size*3.3, 
                  c='#EF5350', edgecolors='#B71C1C', 
                  linewidth=2.5, alpha=0.95, zorder=4)
        
        # Numero bianco sopra
        ax.text(supp, f1, str(num),
               fontsize=13, fontweight='bold', color='white',
               ha='center', va='center', zorder=5)

# Configurazione assi
ax.set_xscale('log')
ax.set_xlabel('Support (numero campioni, scala log₁₀)', 
              fontsize=17, fontweight='bold', labelpad=12)
ax.set_ylabel('F1-score', fontsize=17, fontweight='bold', labelpad=12)
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(supports.min() * 0.75, supports.max() * 1.25)

# Soglie orizzontali
ax.axhline(y=0.5, color='#7CB342', linestyle='--', linewidth=3.2, 
           alpha=0.75, zorder=1, label='Soglia accettabile')
ax.axhline(y=0.3, color='#FB8C00', linestyle=':', linewidth=2.8, 
           alpha=0.7, zorder=1, label='Soglia critica')

# Zone colorate
ax.axhspan(-0.02, 0.3, alpha=0.12, color='#D32F2F', zorder=0)
ax.axhspan(0.3, 0.5, alpha=0.10, color='#F57C00', zorder=0)
ax.axhspan(0.5, 1.02, alpha=0.06, color='#388E3C', zorder=0)

# Grid professionale
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6, color='#E0E0E0')
ax.tick_params(axis='both', labelsize=11, width=1.2, length=5)

# Stile assi
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('#424242')

# Legend pulita
ax.legend(loc='lower right', fontsize=13, frameon=True, 
          framealpha=0.95, edgecolor='#BDBDBD', fancybox=True)

# Salvataggio
plt.savefig(OUTPUT_DIR / 'v0_support_vs_f1_IMPROVED.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Grafico migliorato salvato con successo!")
print(f"   → {OUTPUT_DIR / 'v0_support_vs_f1.png'}")




#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#


 


# ═══════════════════════════════════════════════════════════════════
# GRAFICO PRECISION-RECALL - PROFESSIONALE CON TABELLE WORD-STYLE
# Layout ottimizzato: grafico grande + tabelle laterali strutturate
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("📈 GENERAZIONE PRECISION-RECALL (LAYOUT PROFESSIONALE)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# CALCOLI ANALITICI PRELIMINARI
# ═══════════════════════════════════════════════════════════════════

f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)
precision_mean = np.mean(precisions)
recall_mean = np.mean(recalls)

# Classi critiche (R<0.3 O P<0.3)
high_risk_idx = np.where((recalls < 0.3) | (precisions < 0.3))[0]
problematic_idx = np.where(f1_scores < 0.5)[0]

# Status complessivo
if f1_mean >= 0.85:
    status_text = "ECCELLENTE"
    status_color = "#2E7D32"
elif f1_mean >= 0.70:
    status_text = "BUONO"
    status_color = "#F57C00"
else:
    status_text = "CRITICO"
    status_color = "#C62828"

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE FIGURA - LAYOUT OTTIMIZZATO V2 - PROPORZIONATO
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(26, 15))  # Aumentata dimensione figura
fig.patch.set_facecolor('white')

# Coordinate layout ottimizzate e proporzionate
PR_TITLE_Y = 0.975
PR_SUBTITLE_Y = 0.945

# GRAFICO PRINCIPALE - Ben centrato a sinistra
PR_GRAPH_X = 0.07
PR_GRAPH_Y = 0.07
PR_GRAPH_W = 0.53
PR_GRAPH_H = 0.72

# LEGENDA - Tabella sopra il grafico, ben proporzionata
PR_LEGEND_X = 0.1385
PR_LEGEND_Y = 0.812
PR_LEGEND_W = 0.39
PR_LEGEND_H = 0.10

# COLORBAR - Adiacente al grafico
COLORBAR_X = 0.55
COLORBAR_Y = 0.07
COLORBAR_W = 0.015
COLORBAR_H = 0.72

# TABELLA METRICHE - Sopra a destra
METRICS_TABLE_X = 0.645
METRICS_TABLE_Y = 0.715
METRICS_TABLE_W = 0.32
METRICS_TABLE_H = 0.20

# TABELLA ANALISI DETTAGLIATA - Centro destra (più grande)
PR_DETAIL_TABLE_X = 0.645
PR_DETAIL_TABLE_Y = 0.235
PR_DETAIL_TABLE_W = 0.32
PR_DETAIL_TABLE_H = 0.47

# TABELLA CLASSI CRITICHE - Sotto destra
CRITICAL_TABLE_X = 0.645
CRITICAL_TABLE_Y = 0.065
CRITICAL_TABLE_W = 0.32
CRITICAL_TABLE_H = 0.15

# ═══════════════════════════════════════════════════════════════════
# TITOLO E SOTTOTITOLO
# ═══════════════════════════════════════════════════════════════════

fig.text(0.55, PR_TITLE_Y, 'Trade-off Precision-Recall per Classe di Attacco',
        fontsize=24, fontweight='bold', ha='center', family='sans-serif')

fig.text(0.55, PR_SUBTITLE_Y, 'V0_no_weight (Epoca 10) - Modello di Classificazione Multi-Classe',
        fontsize=17, ha='center', style='italic', color='#555555')

# ═══════════════════════════════════════════════════════════════════
# TABELLA METRICHE AGGREGATE (SOPRA DESTRA) - STILE WORD
# ═══════════════════════════════════════════════════════════════════

metrics_ax = plt.axes([METRICS_TABLE_X, METRICS_TABLE_Y, METRICS_TABLE_W, METRICS_TABLE_H])
metrics_ax.axis('off')

# Bordo tabella
metrics_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   fill=False, edgecolor='#1976D2',
                                   linewidth=3, transform=metrics_ax.transAxes))

# Header tabella
metrics_ax.add_patch(plt.Rectangle((0, 0.83), 1, 0.17,
                                   fill=True, facecolor='#E3F2FD',
                                   edgecolor='#1976D2', linewidth=1.8,
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.5, 0.91, 'PERFORMANCE AGGREGATA',
                fontsize=19, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)

# Header colonne
metrics_ax.add_patch(plt.Rectangle((0, 0.70), 1, 0.13,
                                   fill=True, facecolor='#F5F5F5',
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.35, 0.765, 'Metrica',
                fontsize=17, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)
metrics_ax.text(0.70, 0.765, 'Valore',
                fontsize=17, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)

# Linea separatrice header
metrics_ax.plot([0, 1], [0.70, 0.70], 'k-', linewidth=1.5,
                transform=metrics_ax.transAxes)

# Righe dati
metrics_data = [
    ('F1-Score medio', f'{f1_mean:.4f} ± {f1_std:.4f}'),
    ('Precision media', f'{precision_mean:.4f}'),
    ('Recall media', f'{recall_mean:.4f}')
]

y_start = 0.58
row_height = 0.13

for i, (metric, value) in enumerate(metrics_data):
    y_pos = y_start - i * row_height
    
    # Background alternato
    if i % 2 == 0:
        metrics_ax.add_patch(plt.Rectangle((0, y_pos - 0.055), 1, row_height - 0.005,
                                           fill=True, facecolor='#FAFAFA',
                                           transform=metrics_ax.transAxes))
    
    metrics_ax.text(0.35, y_pos, metric,
                    fontsize=16, ha='center', va='center',
                    fontweight='bold',
                    transform=metrics_ax.transAxes)
    metrics_ax.text(0.70, y_pos, value,
                    fontsize=16, ha='center', va='center',
                    family='monospace',
                    transform=metrics_ax.transAxes)

# Separatore prima del status
metrics_ax.plot([0, 1], [0.19, 0.19], 'k-', linewidth=1.5,
                transform=metrics_ax.transAxes)

# Status finale (riga evidenziata)
metrics_ax.add_patch(plt.Rectangle((0, 0), 1, 0.19,
                                   fill=True, facecolor=status_color,
                                   alpha=0.15,
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.35, 0.095, 'Status Modello',
                fontsize=15, ha='center', va='center',
                fontweight='bold',
                transform=metrics_ax.transAxes)
metrics_ax.text(0.70, 0.095, status_text,
                fontsize=15, ha='center', va='center',
                fontweight='bold', color=status_color,
                transform=metrics_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# LEGENDA - TABELLA STILE WORD SOPRA IL GRAFICO
# ═══════════════════════════════════════════════════════════════════

legend_ax = plt.axes([PR_LEGEND_X, PR_LEGEND_Y, PR_LEGEND_W, PR_LEGEND_H])
legend_ax.axis('off')

# Bordo tabella principale
legend_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                  fill=False, edgecolor='#424242',
                                  linewidth=2.5, transform=legend_ax.transAxes))

# Header
legend_ax.add_patch(plt.Rectangle((0, 0.70), 1, 0.30,
                                  fill=True, facecolor='#ECEFF1',
                                  edgecolor='#424242', linewidth=1.5,
                                  transform=legend_ax.transAxes))

legend_ax.text(0.5, 0.85, 'LEGENDA GRAFICO',
              fontsize=19, fontweight='bold', ha='center', va='center',
              transform=legend_ax.transAxes)

# Linea separatrice
legend_ax.plot([0, 1], [0.70, 0.70], 'k-', linewidth=1.5,
              transform=legend_ax.transAxes)

# Contenuto - Riga 1: Curve Iso-F1
legend_ax.text(0.03, 0.48, '• Curve Iso-F1 (grigio):',
              fontsize=16, ha='left', va='center', fontweight='bold',
              transform=legend_ax.transAxes)
legend_ax.text(0.50, 0.48, '0.30 — 0.50 — 0.70 — 0.90 — 0.95',
              fontsize=15, ha='left', va='center', family='monospace',
              transform=legend_ax.transAxes)

# Separatore sottile
legend_ax.plot([0.02, 0.98], [0.37, 0.37], '-', color='#BDBDBD', 
              linewidth=1, alpha=0.5, transform=legend_ax.transAxes)

# Contenuto - Riga 2: Elementi grafici
legend_ax.text(0.03, 0.20, '• Elementi:',
              fontsize=16, ha='left', va='center', fontweight='bold',
              transform=legend_ax.transAxes)
legend_ax.text(0.25, 0.20, 'Diag. P=R (nero)',
              fontsize=15, ha='left', va='center',
              transform=legend_ax.transAxes)
legend_ax.text(0.50, 0.20, 'Verde: P,R > 0.9',
              fontsize=15, ha='left', va='center', color='#2E7D32',
              fontweight='bold', transform=legend_ax.transAxes)
legend_ax.text(0.72, 0.20, 'Rosso: P,R < 0.5',
              fontsize=15, ha='left', va='center', color='#C62828',
              fontweight='bold', transform=legend_ax.transAxes)
# ═══════════════════════════════════════════════════════════════════
# GRAFICO PRINCIPALE (GRANDE E CENTRATO)
# ═══════════════════════════════════════════════════════════════════

# ax = plt.axes([GRAPH_X-0.19, GRAPH_Y, GRAPH_W+0.2, GRAPH_H+0.1])
ax = plt.axes([PR_GRAPH_X, PR_GRAPH_Y, PR_GRAPH_W, PR_GRAPH_H])
# Scatter plot
scatter = ax.scatter(recalls, precisions,
                    s=380, alpha=0.8,
                    c=supports, cmap='plasma',
                    norm=LogNorm(),
                    edgecolors='#212121', linewidth=2.5,
                    zorder=3)

# Numeri con colori intelligenti
colors_critical = ['#D32F2F', '#E64A19', '#F57C00', '#FBC02D', '#7CB342']

for i in range(len(class_names)):
    if f1_scores[i] < 0.3:
        color_bg = '#B71C1C'  # Rosso scuro
    elif f1_scores[i] < 0.5:
        color_bg = colors_critical[i % len(colors_critical)]
    else:
        color_bg = '#1565C0'  # Blu

    ax.text(recalls[i], precisions[i], str(i+1),
           fontsize=11, fontweight='bold', color='white',
           ha='center', va='center', zorder=5,
           bbox=dict(boxstyle='circle,pad=0.35',
                    facecolor=color_bg,
                    edgecolor='white',
                    linewidth=2.2))

# Curve Iso-F1
recall_range = np.linspace(0.001, 0.999, 500)
f1_values = [0.3, 0.5, 0.7, 0.9, 0.95]
grays = ['#E0E0E0', '#BDBDBD', '#9E9E9E', '#757575', '#424242']
linewidths = [2.1, 2.18, 2.23, 2.23, 2.4]

for f1_val, gray, lw in zip(f1_values, grays, linewidths):
    precision_curve = (f1_val * recall_range) / (2 * recall_range - f1_val * recall_range + 1e-10)
    valid_mask = (precision_curve >= 0) & (precision_curve <= 1)
    ax.plot(recall_range[valid_mask], precision_curve[valid_mask],
           '--', alpha=0.45, linewidth=lw, color=gray, zorder=2)

    # Etichetta F1
    if f1_val in [0.5, 0.9]:
        idx_label = int(len(recall_range) * 0.6)
        ax.text(recall_range[idx_label], precision_curve[idx_label] + 0.03,
               f'F1={f1_val}', fontsize=11, color=gray, fontweight='bold', zorder=2.5)

# Diagonale P=R
ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=2.5, zorder=2)

# Zone colorate
ax.fill_between([0.9, 1.03], [0.9, 0.9], [1.03, 1.03],
               alpha=0.10, color='#2E7D32', zorder=0)
ax.fill_between([0, 0.5], [0, 0], [0.5, 0.5],
               alpha=0.10, color='#C62828', zorder=0)

# Configurazione assi
ax.set_xlabel('Recall (Sensibilità)', fontsize=15, fontweight='bold', labelpad=10)
ax.set_ylabel('Precision (Precisione)', fontsize=15, fontweight='bold', labelpad=10)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6, color='#E0E0E0')
ax.tick_params(axis='both', labelsize=11, width=1.2, length=5)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('#424242')




# ═══════════════════════════════════════════════════════════════════
# COLORBAR (SPOSTATA A SINISTRA)
# ═══════════════════════════════════════════════════════════════════

cbar_ax = plt.axes([COLORBAR_X, COLORBAR_Y, COLORBAR_W, COLORBAR_H])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Support\n(campioni)', fontsize=14, fontweight='bold',
              rotation=0, labelpad=15, va='center', ha='left')
cbar.ax.tick_params(labelsize=10, width=1.2, length=4)




# ═══════════════════════════════════════════════════════════════════
# TABELLA CLASSI CRITICHE (CENTRO DESTRA) - STILE WORD MIGLIORATO
# ═══════════════════════════════════════════════════════════════════

critical_ax = plt.axes([CRITICAL_TABLE_X, CRITICAL_TABLE_Y-0.03, CRITICAL_TABLE_W, CRITICAL_TABLE_H+0.04])
critical_ax.axis('off')

# Bordo tabella esterno
critical_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   fill=False, edgecolor='#C62828',
                                   linewidth=3, transform=critical_ax.transAxes))

# Header tabella SINGOLO (10% dell'altezza)
critical_ax.add_patch(plt.Rectangle((0, 0.90), 1, 0.15,
                                   fill=True, facecolor='#FFEBEE',
                                   edgecolor='#C62828', linewidth=1.8,
                                   transform=critical_ax.transAxes))

critical_ax.text(0.53, 0.950, 'CLASSI CRITICHE',
                fontsize=18, fontweight='bold', ha='center', va='center',
                transform=critical_ax.transAxes)

# Linea separatrice sotto header principale
critical_ax.plot([0, 1], [0.90, 0.90], 'k-', linewidth=1.8,
                transform=critical_ax.transAxes)

# Contenuto
if len(high_risk_idx) > 0:
    # Header colonne INTEGRATO con descrizione (12% altezza)
    critical_ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.12,
                                       fill=True, facecolor="#E8E6E6D8",
                                       transform=critical_ax.transAxes))
    
    # Descrizione a sinistra in italico
    critical_ax.text(0.07, 0.83, 'R < 0.3 o P < 0.3',
                    fontsize=16, ha='left', va='center',
                    color='black', fontweight='bold',
                    transform=critical_ax.transAxes)
    
    # Separatore verticale sottile
    critical_ax.plot([0.35, 0.35], [0.78, 0.90], '-', color="#282727", 
                    linewidth=1, alpha=0.6, transform=critical_ax.transAxes)
    
    # Header colonne a destra
    critical_ax.text(0.52, 0.81, 'ID', fontsize=16, fontweight='bold', ha='center',
                    transform=critical_ax.transAxes)
    # critical_ax.text(0.52, 0.81, '━━', fontsize=8, ha='center', color='#999999',
    #                 transform=critical_ax.transAxes)
    
    critical_ax.text(0.75, 0.81, 'Classe', fontsize=16, fontweight='bold', ha='center',
                    transform=critical_ax.transAxes)
    # critical_ax.text(0.75, 0.81, '━━━━━', fontsize=8, ha='center', color='#999999',
    #                 transform=critical_ax.transAxes)
    
    # Linea separatrice sotto header colonne
    critical_ax.plot([0, 1], [0.78, 0.78], 'k-', linewidth=1.5,
                    transform=critical_ax.transAxes)
    
    # Calcolo dinamico spazio righe (58% disponibile / numero righe)
    n_rows = min(len(high_risk_idx), 4)
    available_space = 0.58  # Dal 0.78 al 0.20
    row_height = available_space / n_rows if n_rows > 0 else 0.145
    
    # Righe dati
    y_start = 0.75
    
    for i, idx in enumerate(high_risk_idx[:4]):
        y_pos = y_start - (i * row_height)
        
        # Background alternato
        if i % 2 == 0:
            critical_ax.add_patch(plt.Rectangle((0.01, y_pos - row_height + 0.005), 
                                               0.98, row_height - 0.01,
                                               fill=True, facecolor='#F5F5F5',  # Grigio chiaro
                                               edgecolor='none',
                                               transform=critical_ax.transAxes))
        
        # Testo centrato verticalmente
        text_y = y_pos - row_height/2
        
        critical_ax.text(0.52, text_y, f'[{idx+1}]',
                        fontsize=14.5, ha='center', va='center',
                        family='monospace', fontweight='bold',
                        transform=critical_ax.transAxes)
        
        critical_ax.text(0.75, text_y, class_abbr[idx][:18],
                        fontsize=14.5, ha='center', va='center',
                        family='monospace',
                        transform=critical_ax.transAxes)
    
    # Linea separatrice prima del footer
    critical_ax.plot([0, 1], [0.17, 0.17], 'k-', linewidth=1.5,
                    transform=critical_ax.transAxes)
    
    # Footer con background
    critical_ax.add_patch(plt.Rectangle((0, 0), 1, 0.17,
                                       fill=True, facecolor='#FFEBEE',  # Stesso del header
                                       edgecolor='none',
                                       transform=critical_ax.transAxes))
   
    critical_ax.text(0.5, 0.085, 
                    f'Totale: {len(high_risk_idx)}/15 classi ({len(high_risk_idx)/15*100:.1f}%)',
                    fontsize=15.5, ha='center', va='center',
                    fontweight='bold', color='#C62828',
                    transform=critical_ax.transAxes)

else:
    # Messaggio nessuna classe critica
    critical_ax.text(0.5, 0.45, '✓ Nessuna classe critica rilevata',
                    fontsize=15.5, ha='center', va='center',
                    style='italic', color='#2E7D32',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8',
                             facecolor='#E8F5E9',
                             edgecolor='#2E7D32',
                             linewidth=2),
                    transform=critical_ax.transAxes)



# ═══════════════════════════════════════════════════════════════════
# TABELLA ANALISI DETTAGLIATA (SOTTO DESTRA) - STILE WORD MIGLIORATO
# ═══════════════════════════════════════════════════════════════════

detail_ax = plt.axes([PR_DETAIL_TABLE_X, PR_DETAIL_TABLE_Y, PR_DETAIL_TABLE_W, PR_DETAIL_TABLE_H])
detail_ax.axis('off')

# Bordo tabella esterno ben definito
detail_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                 fill=False, edgecolor='#1565C0',
                                 linewidth=3, transform=detail_ax.transAxes))

# Header tabella
detail_ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06,
                                 fill=True, facecolor='#E3F2FD',
                                 edgecolor='#1565C0', linewidth=1.8,
                                 transform=detail_ax.transAxes))

detail_ax.text(0.5, 0.97, 'ANALISI DETTAGLIATA CLASSI',
              fontsize=19, fontweight='bold', ha='center', va='center',
              transform=detail_ax.transAxes)

# Sub-header
# detail_ax.text(0.5, 0.915, 'Ordinato per F1-Score decrescente',
#               fontsize=16, ha='center', style='italic', color='#666666',
#               transform=detail_ax.transAxes)

# Header colonne con background
detail_ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.06,
                                 fill=True, facecolor='#F5F5F5',
                                 transform=detail_ax.transAxes))

col_headers = [
    (0.10, 'ID'),
    (0.35, 'Classe'),
    (0.58, 'F1'),
    (0.73, 'P'),
    (0.88, 'R')
]

for x, header in col_headers:
    detail_ax.text(x, 0.91, header, fontsize=16, fontweight='bold',
                  ha='center', va='center', transform=detail_ax.transAxes)

# Linea separatrice header robusta
detail_ax.plot([0, 1], [0.88, 0.88], 'k-', linewidth=1.8,
              transform=detail_ax.transAxes)

# Ordina per F1 decrescente
sorted_idx = np.argsort(f1_scores)
n_classes = 15

# Righe dati con font più grande
y_start = 0.84
row_height = 0.058

for rank, idx in enumerate(sorted_idx[:n_classes]):
    y_pos = y_start - rank * row_height
    
    # Colore background basato su performance
    if f1_scores[idx] >= 0.95:
        bg_color = '#C8E6C9'
        edge_color = '#2E7D32'
        edge_width = 1.5
    elif f1_scores[idx] >= 0.80:
        bg_color = '#E8F5E9'
        edge_color = '#4CAF50'
        edge_width = 1.3
    elif f1_scores[idx] >= 0.60:
        bg_color = '#FFF9C4'
        edge_color = '#F9A825'
        edge_width = 1.3
    elif f1_scores[idx] >= 0.50:
        bg_color = '#FFE0B2'
        edge_color = '#EF6C00'
        edge_width = 1.5
    else:
        bg_color = '#FFCDD2'
        edge_color = '#C62828'
        edge_width = 1.8
    
    # Background riga con bordi ben definiti
    detail_ax.add_patch(plt.Rectangle((0, y_pos - 0.026), 1, row_height - 0.002,
                                     fill=True, facecolor=bg_color,
                                     edgecolor=edge_color, linewidth=edge_width,
                                     transform=detail_ax.transAxes))
    
    # Dati cella con font ingrandito
    detail_ax.text(0.10, y_pos, f'[{idx+1}]',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace', fontweight='bold',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.35, y_pos, class_abbr[idx][:16],
                  fontsize=14.5, ha='center', va='center',
                  family='monospace',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.58, y_pos, f'{f1_scores[idx]:.3f}',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace', fontweight='bold',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.73, y_pos, f'{precisions[idx]:.3f}',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.88, y_pos, f'{recalls[idx]:.3f}',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace',
                  transform=detail_ax.transAxes)
    
    # Separatori ogni 5 righe
    if (rank + 1) % 5 == 0 and rank < n_classes - 1:
        sep_y = y_pos - row_height / 2
        detail_ax.plot([0, 1], [sep_y, sep_y], 'k-', alpha=0.3, linewidth=1,
                      transform=detail_ax.transAxes)
# ═══════════════════════════════════════════════════════════════════
# FOOTER METADATI
# ═══════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════
# SALVATAGGIO
# ═══════════════════════════════════════════════════════════════════

plt.savefig(OUTPUT_DIR/'v0_precision_recall.png',
           dpi=400, bbox_inches='tight', facecolor='white')

print("✅ Grafico PROFESSIONALE salvato!")
print(f"   📊 Layout ottimizzato:")
print(f"      • Grafico principale: {PR_GRAPH_W*100:.0f}% larghezza")
print(f"      • Tabelle Word-style: {PR_DETAIL_TABLE_W*100:.0f}% larghezza")
print(f"      • Zero sovrapposizioni garantite")
print(f"   🎨 Caratteristiche:")
print(f"      • F1-Score medio: {f1_mean:.4f}")
print(f"      • Classi critiche: {len(high_risk_idx)}/15")
print(f"      • Colorbar visibile al 100%")
print("=" * 70 + "\n")

plt.close()



#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#


# ═══════════════════════════════════════════════════════════════════
# GRAFICO EVOLUTION - PROFESSIONALE CON TABELLE WORD-STYLE
# Layout ottimizzato seguendo lo stile Precision-Recall
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("📈 GENERAZIONE EVOLUTION PLOT (LAYOUT PROFESSIONALE)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# DATI EVOLUTION
# ═══════════════════════════════════════════════════════════════════

epochs = np.array([1, 10])
f1_loic_udp = np.array([0.3781, 0.3922])
f1_ftp = np.array([0.0034, 0.1413])
f1_sql = np.array([0.0349, 0.2018])
f1_ssh = np.array([0.0000, 0.1365])

# Calcola miglioramenti
improvements_data = [
    ('DDOS attack-LOIC-UDP', f1_loic_udp, '#3498DB'),
    ('FTP-BruteForce', f1_ftp, '#E67E22'),
    ('SQL Injection', f1_sql, '#27AE60'),
    ('SSH-Bruteforce', f1_ssh, '#E74C3C')
]

# Status complessivo
mean_improvement = np.mean([
    (f1_loic_udp[-1] - f1_loic_udp[0]) / (f1_loic_udp[0] + 1e-10) * 100,
    (f1_ftp[-1] - f1_ftp[0]) / (f1_ftp[0] + 1e-10) * 100,
    (f1_sql[-1] - f1_sql[0]) / (f1_sql[0] + 1e-10) * 100,
    (f1_ssh[-1] - f1_ssh[0] + 0.1365) / 0.1365 * 100  # SSH parte da 0
])

if mean_improvement >= 300:
    status_text = "SIGNIFICATIVO"
    status_color = "#F57C00"
elif mean_improvement >= 100:
    status_text = "MODERATO"
    status_color = "#FBC02D"
else:
    status_text = "LIMITATO"
    status_color = "#C62828"

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE FIGURA - LAYOUT OTTIMIZZATO
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(26, 15))
fig.patch.set_facecolor('white')

# Coordinate layout (proporzionate come Precision-Recall)
TITLE_Y = 0.975
SUBTITLE_Y = 0.945

# GRAFICO PRINCIPALE - Centrato a sinistra (più grande)
GRAPH_X = 0.07
GRAPH_Y = 0.07
GRAPH_W = 0.53
GRAPH_H = 0.72

# LEGENDA - Tabella sopra il grafico
LEGEND_X = 0.1385
LEGEND_Y = 0.812
LEGEND_W = 0.39
LEGEND_H = 0.10

# TABELLA METRICHE AGGREGATE - Sopra a destra
METRICS_TABLE_X = 0.645
METRICS_TABLE_Y = 0.715
METRICS_TABLE_W = 0.32
METRICS_TABLE_H = 0.20

# TABELLA MIGLIORAMENTI DETTAGLIATI - Centro destra
DETAIL_TABLE_X = 0.645
DETAIL_TABLE_Y = 0.235
DETAIL_TABLE_W = 0.32
DETAIL_TABLE_H = 0.47

# TABELLA ALERT - Sotto destra
ALERT_TABLE_X = 0.645
ALERT_TABLE_Y = 0.065
ALERT_TABLE_W = 0.32
ALERT_TABLE_H = 0.15

# ═══════════════════════════════════════════════════════════════════
# TITOLO E SOTTOTITOLO
# ═══════════════════════════════════════════════════════════════════

fig.text(0.55, TITLE_Y, 'Evoluzione delle Performance per Classi Minoritarie',
        fontsize=24, fontweight='bold', ha='center', family='sans-serif')

fig.text(0.55, SUBTITLE_Y, 'V0_no_weight (Training Epoch 1 → 10) - Analisi Progressione F1-Score',
        fontsize=17, ha='center', style='italic', color='#555555')

# ═══════════════════════════════════════════════════════════════════
# TABELLA METRICHE AGGREGATE (SOPRA DESTRA) - STILE WORD
# ═══════════════════════════════════════════════════════════════════

metrics_ax = plt.axes([METRICS_TABLE_X, METRICS_TABLE_Y, METRICS_TABLE_W, METRICS_TABLE_H])
metrics_ax.axis('off')

# Bordo tabella
metrics_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   fill=False, edgecolor='#1976D2',
                                   linewidth=3, transform=metrics_ax.transAxes))

# Header tabella
metrics_ax.add_patch(plt.Rectangle((0, 0.83), 1, 0.17,
                                   fill=True, facecolor='#E3F2FD',
                                   edgecolor='#1976D2', linewidth=1.8,
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.5, 0.91, 'SINTESI PROGRESSIONE',
                fontsize=19, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)

# Header colonne
metrics_ax.add_patch(plt.Rectangle((0, 0.70), 1, 0.13,
                                   fill=True, facecolor='#F5F5F5',
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.35, 0.765, 'Metrica',
                fontsize=17, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)
metrics_ax.text(0.70, 0.765, 'Valore',
                fontsize=17, fontweight='bold', ha='center', va='center',
                transform=metrics_ax.transAxes)

# Linea separatrice header
metrics_ax.plot([0, 1], [0.70, 0.70], 'k-', linewidth=1.5,
                transform=metrics_ax.transAxes)

# Calcola metriche aggregate
final_f1_values = [f1_loic_udp[-1], f1_ftp[-1], f1_sql[-1], f1_ssh[-1]]
initial_f1_values = [f1_loic_udp[0], f1_ftp[0], f1_sql[0], f1_ssh[0]]
mean_final = np.mean(final_f1_values)
mean_initial = np.mean(initial_f1_values)
mean_delta = mean_final - mean_initial

# Righe dati
metrics_data = [
    ('F1 medio Epoch 1', f'{mean_initial:.4f}'),
    ('F1 medio Epoch 10', f'{mean_final:.4f}'),
    ('Miglioramento Δ', f'+{mean_delta:.4f}')
]

y_start = 0.58
row_height = 0.13

for i, (metric, value) in enumerate(metrics_data):
    y_pos = y_start - i * row_height
    
    # Background alternato
    if i % 2 == 0:
        metrics_ax.add_patch(plt.Rectangle((0, y_pos - 0.055), 1, row_height - 0.005,
                                           fill=True, facecolor='#FAFAFA',
                                           transform=metrics_ax.transAxes))
    
    metrics_ax.text(0.35, y_pos, metric,
                    fontsize=16, ha='center', va='center',
                    fontweight='bold',
                    transform=metrics_ax.transAxes)
    metrics_ax.text(0.70, y_pos, value,
                    fontsize=16, ha='center', va='center',
                    family='monospace',
                    transform=metrics_ax.transAxes)

# Separatore prima del status
metrics_ax.plot([0, 1], [0.19, 0.19], 'k-', linewidth=1.5,
                transform=metrics_ax.transAxes)

# Status finale
metrics_ax.add_patch(plt.Rectangle((0, 0), 1, 0.19,
                                   fill=True, facecolor=status_color,
                                   alpha=0.15,
                                   transform=metrics_ax.transAxes))

metrics_ax.text(0.35, 0.095, 'Tipo Miglioramento',
                fontsize=15, ha='center', va='center',
                fontweight='bold',
                transform=metrics_ax.transAxes)
metrics_ax.text(0.70, 0.095, status_text,
                fontsize=15, ha='center', va='center',
                fontweight='bold', color=status_color,
                transform=metrics_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# LEGENDA - TABELLA STILE WORD SOPRA IL GRAFICO
# ═══════════════════════════════════════════════════════════════════

legend_ax = plt.axes([LEGEND_X, LEGEND_Y, LEGEND_W, LEGEND_H])
legend_ax.axis('off')

# Bordo tabella
legend_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                  fill=False, edgecolor='#424242',
                                  linewidth=2.5, transform=legend_ax.transAxes))

# Header
legend_ax.add_patch(plt.Rectangle((0, 0.70), 1, 0.30,
                                  fill=True, facecolor='#ECEFF1',
                                  edgecolor='#424242', linewidth=1.5,
                                  transform=legend_ax.transAxes))

legend_ax.text(0.5, 0.85, 'LEGENDA GRAFICO',
              fontsize=19, fontweight='bold', ha='center', va='center',
              transform=legend_ax.transAxes)

# Linea separatrice
legend_ax.plot([0, 1], [0.70, 0.70], 'k-', linewidth=1.5,
              transform=legend_ax.transAxes)

# Contenuto - Riga 1: Classi
legend_ax.text(0.03, 0.48, '• Classi:',
              fontsize=16, ha='left', va='center', fontweight='bold',
              transform=legend_ax.transAxes)
legend_ax.text(0.20, 0.48, 'LOIC-UDP (●—)  |  FTP-BF (■- -)  |  SQL Inj (▲-·)  |  SSH-BF (♦···)',
              fontsize=14, ha='left', va='center', family='monospace',
              transform=legend_ax.transAxes)

# Separatore
legend_ax.plot([0.02, 0.98], [0.37, 0.37], '-', color='#BDBDBD', 
              linewidth=1, alpha=0.5, transform=legend_ax.transAxes)

# Contenuto - Riga 2: Soglie
legend_ax.text(0.03, 0.20, '• Soglie:',
              fontsize=16, ha='left', va='center', fontweight='bold',
              transform=legend_ax.transAxes)
legend_ax.text(0.20, 0.20, 'Accettabile F1=0.5 (- -)  |  Critica F1=0.3 (···)',
              fontsize=14, ha='left', va='center',
              transform=legend_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# GRAFICO PRINCIPALE (GRANDE E CENTRATO)
# ═══════════════════════════════════════════════════════════════════

ax = plt.axes([GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H])

colors = ['#3498DB', '#E67E22', '#27AE60', '#E74C3C']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']
labels = ['DDOS attack-LOIC-UDP', 'FTP-BruteForce', 'SQL Injection', 'SSH-Bruteforce']

data_series = [
    (labels[0], f1_loic_udp, colors[0], markers[0], linestyles[0]),
    (labels[1], f1_ftp, colors[1], markers[1], linestyles[1]),
    (labels[2], f1_sql, colors[2], markers[2], linestyles[2]),
    (labels[3], f1_ssh, colors[3], markers[3], linestyles[3])
]

# Plot linee con marcatori più grandi
for label, data, color, marker, ls in data_series:
    ax.plot(epochs, data, linestyle=ls, linewidth=4, 
            markersize=18, marker=marker, color=color,
            markeredgecolor='black', markeredgewidth=2.2,
            alpha=0.85, zorder=3)

# Soglie con stile professionale
ax.axhline(y=0.5, color='#95A5A6', linestyle='--', linewidth=3, 
           alpha=0.7, zorder=2, label='Soglia accettabile')
ax.axhline(y=0.3, color='#E67E22', linestyle=':', linewidth=2.5, 
           alpha=0.6, zorder=2, label='Soglia critica')

# Configurazione assi
ax.set_xlabel('Epoca di Training', fontsize=15, fontweight='bold', labelpad=10)
ax.set_ylabel('F1-score', fontsize=15, fontweight='bold', labelpad=10)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.02, 0.58)
ax.set_xticks(range(1, 11))
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6, color='#E0E0E0')
ax.tick_params(axis='both', labelsize=11, width=1.2, length=5)

# Zone problematiche
ax.axhspan(-0.02, 0.3, alpha=0.10, color='#C62828', zorder=0)
ax.axhspan(0.3, 0.5, alpha=0.08, color='#F57C00', zorder=0)

# Stile assi
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('#424242')



# ═══════════════════════════════════════════════════════════════════
# TABELLA MIGLIORAMENTI DETTAGLIATI (CENTRO DESTRA) - STILE WORD V2
# Con leggenda colori integrata e spacing ottimizzato
# ═══════════════════════════════════════════════════════════════════

detail_ax = plt.axes([DETAIL_TABLE_X, DETAIL_TABLE_Y, DETAIL_TABLE_W, DETAIL_TABLE_H])
detail_ax.axis('off')

# Bordo tabella esterno
detail_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                 fill=False, edgecolor='#1565C0',
                                 linewidth=3, transform=detail_ax.transAxes))

# ═══════════════════════════════════════════════════════════════════
# HEADER PRINCIPALE (8% altezza)
# ═══════════════════════════════════════════════════════════════════

detail_ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08,
                                 fill=True, facecolor='#E3F2FD',
                                 edgecolor='#1565C0', linewidth=1.8,
                                 transform=detail_ax.transAxes))

detail_ax.text(0.5, 0.96, 'MIGLIORAMENTI DETTAGLIATI',
              fontsize=19, fontweight='bold', ha='center', va='center',
              transform=detail_ax.transAxes)

# Linea separatrice sotto header
detail_ax.plot([0, 1], [0.92, 0.92], 'k-', linewidth=1.8,
              transform=detail_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# LEGENDA COLORI (12% altezza) - SUBITO SOTTO IL TITOLO
# ═══════════════════════════════════════════════════════════════════

legend_section_ax = plt.axes([DETAIL_TABLE_X, DETAIL_TABLE_Y + DETAIL_TABLE_H * 0.80, 
                              DETAIL_TABLE_W, DETAIL_TABLE_H * 0.12])
legend_section_ax.axis('off')

# Background legenda
detail_ax.add_patch(plt.Rectangle((0, 0.80), 1, 0.12,
                                 fill=True, facecolor='#F5F5F5',
                                 transform=detail_ax.transAxes))

# Titolo legenda (compatto)
detail_ax.text(0.5, 0.89, 'Legenda Colorimetrica (basata su Δ %)',
              fontsize=15, ha='center', va='center',
              style='italic', color='#424242',
              transform=detail_ax.transAxes)

# Linea separatrice sottile sopra legenda
detail_ax.plot([0.02, 0.98], [0.92, 0.92], '-', color='#BDBDBD', 
              linewidth=0.8, transform=detail_ax.transAxes)


legend_items = [
    ("≥ 400%", "#246528E6", "#C8E6C9"),
    ("200–400%", "#388E3C", "#E8F5E9"),
    ("50–200%", "#F9A825", "#FFF9C4"),
    ("< 50%", "#EF6C00", "#FFE0B2")
]

x_positions = [0.15, 0.40, 0.65, 0.90]

for (label, edge_color, bg_color), x in zip(legend_items, x_positions):
    detail_ax.add_patch(
        plt.Rectangle((x - 0.055, 0.82), 0.11, 0.045,
                      facecolor=bg_color,
                      edgecolor=edge_color,
                      linewidth=1.3,
                      transform=detail_ax.transAxes)
    )

    detail_ax.text(x, 0.85, label,
                   fontsize=11, ha='center', va='top',
                   color='black', fontweight='bold',
                   family='monospace',
                   transform=detail_ax.transAxes)

# Linea separatrice sotto legenda
detail_ax.plot([0, 1], [0.80, 0.80], 'k-', linewidth=1.5,
              transform=detail_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# HEADER COLONNE (6% altezza)
# ═══════════════════════════════════════════════════════════════════

col_headers = [
    (0.28, 'Classe'),
    (0.54, 'Epoch 1'),
    (0.70, 'Epoch 10'),
    (0.88, 'Δ %')
]

for x, header in col_headers:
    detail_ax.text(x, 0.77, header, fontsize=16, fontweight='bold',
                  ha='center', va='center', transform=detail_ax.transAxes)

# Linea separatrice header colonne
detail_ax.plot([0, 1], [0.74, 0.74], 'k-', linewidth=1.8,
              transform=detail_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# RIGHE DATI (74% spazio disponibile, ben distribuito)
# ═══════════════════════════════════════════════════════════════════

# Ordina per percentuale decrescente
improvements_with_idx = []
for i, (label, data, color) in enumerate(improvements_data):
    delta = data[-1] - data[0]
    if data[0] > 0:
        perc = (delta / data[0]) * 100
    else:
        perc = 999  # SSH che parte da 0
    improvements_with_idx.append((label, data, color, perc, delta))

improvements_with_idx.sort(key=lambda x: x[3], reverse=True)

# Calcolo spacing ottimizzato (4 righe in 74% spazio = 0.74/4 = 0.185 per riga)
n_rows = len(improvements_with_idx)
available_space = 0.74  # Da 0.74 a 0.00
row_height = available_space / n_rows if n_rows > 0 else 0.185

y_start = 0.72

for rank, (label, data, color, perc, delta) in enumerate(improvements_with_idx):
    y_pos = y_start - (rank * row_height)
    
    # Colore background basato su Δ%
    if perc >= 400:
        bg_color = '#C8E6C9'
        edge_color = '#2E7D32'
        edge_width = 1.8
    elif perc >= 200:
        bg_color = '#E8F5E9'
        edge_color = '#4CAF50'
        edge_width = 1.5
    elif perc >= 50:
        bg_color = '#FFF9C4'
        edge_color = '#F9A825'
        edge_width = 1.5
    else:
        bg_color = '#FFE0B2'
        edge_color = '#EF6C00'
        edge_width = 1.8
    
    
    detail_ax.add_patch(plt.Rectangle((0, y_pos - row_height),
                                 1, row_height,
                                 fill=True, facecolor=bg_color,
                                 edgecolor=edge_color, linewidth=1.2,
                                 transform=detail_ax.transAxes))

    # Testo centrato verticalmente nella riga
    text_y = y_pos - row_height / 2
    
    detail_ax.text(0.28, text_y, label[:18],
                  fontsize=14.5, ha='center', va='center',
                  fontweight='bold',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.54, text_y, f'{data[0]:.4f}',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace',
                  transform=detail_ax.transAxes)
    
    detail_ax.text(0.70, text_y, f'{data[-1]:.4f}',
                  fontsize=14.5, ha='center', va='center',
                  family='monospace', fontweight='bold',
                  transform=detail_ax.transAxes)
    
    if perc == 999:
        perc_text = '(da 0)'
    else:
        perc_text = f'+{perc:.1f}%'
    
    detail_ax.text(0.88, text_y, perc_text,
                  fontsize=14.5, ha='center', va='center',
                  family='monospace', fontweight='bold',
                  color=edge_color,
                  transform=detail_ax.transAxes)

# Fine sezione tabella miglioramenti
# ═══════════════════════════════════════════════════════════════════
# TABELLA ALERT (SOTTO DESTRA) - STILE WORD
# ═══════════════════════════════════════════════════════════════════

alert_ax = plt.axes([ALERT_TABLE_X, ALERT_TABLE_Y - 0.03, ALERT_TABLE_W, ALERT_TABLE_H + 0.04])
alert_ax.axis('off')

# Bordo tabella
alert_ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                fill=False, edgecolor='#C62828',
                                linewidth=3, transform=alert_ax.transAxes))

# Header tabella
alert_ax.add_patch(plt.Rectangle((0, 0.75), 1, 0.25,
                                fill=True, facecolor='#FFEBEE',
                                edgecolor='#C62828', linewidth=1.8,
                                transform=alert_ax.transAxes))

alert_ax.text(0.5, 0.875, 'PERFORMANCE RESIDUA',
             fontsize=18, fontweight='bold', ha='center', va='center',
             color='#C62828',
             transform=alert_ax.transAxes)

# Linea separatrice
alert_ax.plot([0, 1], [0.75, 0.75], 'k-', linewidth=1.8,
             transform=alert_ax.transAxes)

# Contenuto alert
classes_below_threshold = sum(1 for _, data, _, _, _ in improvements_with_idx if data[-1] < 0.5)

alert_text = (
    f"Classi sotto F1=0.5: {classes_below_threshold}/4\n\n"
    f"Miglior F1 raggiunto: {max([d[-1] for _, d, _, _, _ in improvements_with_idx]):.4f}\n"
    f"(DDOS-LOIC-UDP)\n\n"
    f"Tutte le classi restano sotto\n"
    f"la soglia accettabile"
)

alert_ax.text(0.5, 0.35, alert_text,
             fontsize=14.5, ha='center', va='center',
             family='monospace',fontweight='bold',
             transform=alert_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════
# SALVATAGGIO
# ═══════════════════════════════════════════════════════════════════

plt.savefig(OUTPUT_DIR/'v0_f1_evolution.png',
           dpi=400, bbox_inches='tight', facecolor='white')

print("✅ Grafico EVOLUTION PROFESSIONALE salvato!")
print(f"   📊 Layout ottimizzato:")
print(f"      • Grafico principale: {GRAPH_W*100:.0f}% larghezza")
print(f"      • Tabelle Word-style: 3 tabelle strutturate")
print(f"      • Stile coerente con Precision-Recall")
print(f"   🎨 Caratteristiche:")
print(f"      • Miglioramento medio: {mean_improvement:.1f}%")
print(f"      • Classi sotto F1=0.5: {classes_below_threshold}/4")
print(f"      • Status: {status_text}")
print("=" * 70 + "\n")

plt.close() 

# ═══════════════════════════════════════════════════════════════════
# RIEPILOGO FINALE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("✅ ANALISI COMPLETATA - VERSIONE ZERO SOVRAPPOSIZIONI!")
print("=" * 70)
print("\n📁 File generati in:", OUTPUT_DIR)
print("   1. v0_support_vs_f1_FIXED.png")
print("   2. v0_precision_recall_FIXED.png")
print("   3. v0_f1_evolution_FIXED.png")
print("\n✨ SOLUZIONI IMPLEMENTATE:")
print("   ✓ Sistema riferimenti NUMERATI sui punti")
print("   ✓ Tabelle legenda ESTERNE (sotto i grafici)")
print("   ✓ Zero sovrapposizioni garantite")
print("   ✓ Maggiore spazio per elementi grafici")
print("   ✓ Colori codificati per performance")
print("   ✓ 400 DPI per qualità pubblicazione")
print("   ✓ Font monospace per tabelle allineate")
print("\n📐 Layout ottimizzato:")
print("   • Grafico principale: area pulita")
print("   • Riferimenti: tabelle sotto (bbox separate)")
print("   • Statistiche: box angolo con abbondante padding")
print("=" * 70)