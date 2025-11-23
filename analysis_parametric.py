"""
Script di analisi PARAMETRICO - Versione Generica
Legge dati da file di testo e genera grafici automaticamente

REQUISITI:
- pip install matplotlib numpy scipy pandas

ESECUZIONE:
python analysis_parametric.py --classification_report report.txt --evolution_data evolution.txt --output ./results
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
from matplotlib.colors import LogNorm
import argparse
import re
import json

# Directory di output
# output_dir = Path("C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Stats_and_Models_History/v0_no_weight/analysis_results/analisi_risultati_v0")
output_dir = Path("./analysis_results")
output_dir.mkdir(parents=True, exist_ok=True)

# print(f"📁 Directory output: {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════════════════
# FUNZIONI DI PARSING - ESTRAZIONE DATI DA FILE
# ═══════════════════════════════════════════════════════════════════

def parse_classification_report(filepath, extract_epoch=True):
    """
    Estrae metriche dal classification report sklearn.
    
    FORMATO ATTESO (file di testo):
    
    Classification Report - Epoch 1
    ================================================================================
                          precision    recall  f1-score   support
    
    Benign                  0.9947    1.0000    0.9974   8804790
    Bot                     0.9972    1.0000    0.9986    571605
    ...
    
    OPPURE formato JSON:
    {
        "Benign": {"precision": 0.9947, "recall": 1.0000, "f1-score": 0.9974, "support": 8804790},
        ...
    }
    
    Args:
        filepath: percorso del file
        extract_epoch: se True, cerca di estrarre il numero epoca dal file
    
    Returns:
        dict con chiavi: class_names, supports, precisions, recalls, f1_scores, epoch (opzionale)
    """
    filepath = Path(filepath)
    
    # Estrai numero epoca dal filename o dal contenuto
    epoch_num = None
    if extract_epoch:
        # Cerca nel filename (es: "report_epoch_5.txt")
        epoch_match = re.search(r'epoch[_\s]*(\d+)', filepath.name, re.IGNORECASE)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
    
    # Prova a leggere come JSON
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        class_names = []
        supports = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, metrics in data.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            class_names.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])
        
        return {
            'class_names': class_names,
            'supports': np.array(supports),
            'precisions': np.array(precisions),
            'recalls': np.array(recalls),
            'f1_scores': np.array(f1_scores)
        }
    
    # Parsing da file di testo formato sklearn
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Cerca epoca nel contenuto se non trovata nel filename
    if epoch_num is None and extract_epoch:
        epoch_match = re.search(r'Epoch\s+(\d+)', content, re.IGNORECASE)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
    
    class_names = []
    supports = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Pattern regex per linee dati
    # Esempio: "    Benign       0.9947    1.0000    0.9974   8804790"
    pattern = r'\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
    
    for line in lines:
        # Salta header e separatori
        if 'precision' in line.lower() or 'accuracy' in line.lower():
            continue
        if line.strip().startswith('---') or not line.strip():
            continue
        
        match = re.match(pattern, line)
        if match:
            class_name = match.group(1).strip()
            
            # Salta righe aggregate (macro avg, weighted avg)
            if 'avg' in class_name.lower() or class_name.lower() == 'accuracy':
                continue
            
            class_names.append(class_name)
            precisions.append(float(match.group(2)))
            recalls.append(float(match.group(3)))
            f1_scores.append(float(match.group(4)))
            supports.append(int(match.group(5)))
    
    if not class_names:
        raise ValueError(f"Nessuna classe trovata nel file {filepath}. Verifica il formato.")
    
    result = {
        'class_names': class_names,
        'supports': np.array(supports),
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'f1_scores': np.array(f1_scores)
    }
    
    if epoch_num is not None:
        result['epoch'] = epoch_num
    
    return result


def parse_evolution_data(filepath):
    """
    Estrae dati di evoluzione F1 per epoche.
    
    FORMATO ATTESO (CSV o TXT):
    
    epoch,class_name,f1_score
    1,DDOS attack-LOIC-UDP,0.3781
    10,DDOS attack-LOIC-UDP,0.3922
    1,FTP-BruteForce,0.0034
    10,FTP-BruteForce,0.1413
    ...
    
    OPPURE formato JSON:
    {
        "DDOS attack-LOIC-UDP": [0.3781, 0.3922],
        "FTP-BruteForce": [0.0034, 0.1413],
        ...
    }
    
    Returns:
        dict con chiavi: epochs, class_evolution (dict {class_name: [f1_values]})
    """
    filepath = Path(filepath)
    
    # JSON format
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Assumi che tutte le classi abbiano lo stesso numero di epoche
        first_class = list(data.values())[0]
        epochs = np.arange(1, len(first_class) + 1)
        
        return {
            'epochs': epochs,
            'class_evolution': {k: np.array(v) for k, v in data.items()}
        }
    
    # CSV/TXT format
    import pandas as pd
    
    # Prova delimitatori comuni
    for sep in [',', '\t', ';']:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) >= 3:
                break
        except:
            continue
    
    if 'epoch' not in df.columns or 'class_name' not in df.columns or 'f1_score' not in df.columns:
        raise ValueError(f"File {filepath} deve avere colonne: epoch, class_name, f1_score")
    
    # Estrai epoche uniche
    epochs = np.sort(df['epoch'].unique())
    
    # Costruisci dizionario per classe
    class_evolution = {}
    for class_name in df['class_name'].unique():
        class_data = df[df['class_name'] == class_name].sort_values('epoch')
        class_evolution[class_name] = class_data['f1_score'].values
    
    return {
        'epochs': epochs,
        'class_evolution': class_evolution
    }


def generate_class_abbreviations(class_names):
    """
    Genera abbreviazioni intelligenti per nomi di classi lunghi.
    
    Args:
        class_names: lista di nomi completi
    
    Returns:
        lista di abbreviazioni (max 15 caratteri)
    """
    abbreviations = []
    
    for name in class_names:
        # Rimuovi "attack" e "attacks"
        name_clean = name.replace(' attack', '').replace(' attacks', '')
        
        # Se già corto, mantienilo
        if len(name_clean) <= 15:
            abbreviations.append(name_clean)
            continue
        
        # Strategia: prendi iniziali delle parole + parte finale
        words = name_clean.split()
        if len(words) > 2:
            # Es: "DoS SlowHTTPTest" -> "DoS-SlowHTTP"
            abbr = '-'.join([words[0]] + [w[:7] for w in words[1:]])
        else:
            # Tronca
            abbr = name_clean[:15]
        
        abbreviations.append(abbr)
    
    return abbreviations


# ═══════════════════════════════════════════════════════════════════
# FUNZIONE AGGREGAZIONE MULTI-EPOCH - TRACKING EVOLUZIONE METRICHE
# ═══════════════════════════════════════════════════════════════════

def aggregate_multi_epoch_reports(folder_path, pattern='*.txt'):
    """
    Legge tutti i classification report in una cartella e traccia l'evoluzione delle metriche.
    
    Args:
        folder_path: percorso cartella contenente i file
        pattern: pattern per matching file (default: '*.txt')
    
    Returns:
        dict con:
            - epochs: array epoche ordinate
            - classes: lista nomi classi
            - metrics_evolution: dict {metric_name: DataFrame con evoluzione}
            - aggregate_metrics: metriche accuracy, macro avg, weighted avg per epoca
    """
    import pandas as pd
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")
    
    # Trova tutti i file matching
    files = sorted(folder.glob(pattern))
    if not files:
        raise ValueError(f"Nessun file trovato in {folder} con pattern '{pattern}'")
    
    print(f"\n📂 Trovati {len(files)} file in {folder}")
    
    # Parse tutti i report
    all_reports = []
    for filepath in files:
        try:
            report = parse_classification_report(filepath, extract_epoch=True)
            
            # Se epoca non trovata, prova a estrarla dal nome file
            if 'epoch' not in report:
                # Cerca pattern tipo "report_1.txt", "epoch_5.txt", etc.
                match = re.search(r'(\d+)', filepath.stem)
                if match:
                    report['epoch'] = int(match.group(1))
                else:
                    print(f"⚠️ Impossibile determinare epoca per {filepath.name} - saltato")
                    continue
            
            all_reports.append(report)
            print(f"   ✓ {filepath.name} → Epoch {report['epoch']}")
        except Exception as e:
            print(f"   ✗ Errore su {filepath.name}: {e}")
            continue
    
    if not all_reports:
        raise ValueError("Nessun report valido trovato!")
    
    # Ordina per epoca
    all_reports.sort(key=lambda x: x['epoch'])
    epochs = np.array([r['epoch'] for r in all_reports])
    
    # Verifica coerenza classi
    reference_classes = all_reports[0]['class_names']
    for i, report in enumerate(all_reports[1:], 1):
        if report['class_names'] != reference_classes:
            print(f"⚠️ ATTENZIONE: Epoch {report['epoch']} ha classi diverse!")
    
    # Costruisci DataFrame per ogni metrica
    metrics = ['precision', 'recall', 'f1_score', 'support']
    metrics_evolution = {}
    
    for metric in metrics:
        # Matrice: righe=classi, colonne=epoche
        data = []
        for report in all_reports:
            if metric == 'f1_score':
                data.append(report['f1_scores'])
            elif metric == 'precision':
                data.append(report['precisions'])
            elif metric == 'recall':
                data.append(report['recalls'])
            elif metric == 'support':
                data.append(report['supports'])
        
        # Trasponi: vogliamo classi come righe, epoche come colonne
        df = pd.DataFrame(
            np.array(data).T,
            index=reference_classes,
            columns=[f'Epoch {e}' for e in epochs]
        )
        metrics_evolution[metric] = df
    
    # Estrai metriche aggregate (accuracy, macro avg, weighted avg) se presenti
    # Per ora non implementato nel parser base, ma possiamo aggiungere
    
    print(f"\n✅ Aggregazione completata: {len(epochs)} epoche, {len(reference_classes)} classi")
    
    return {
        'epochs': epochs,
        'classes': reference_classes,
        'metrics_evolution': metrics_evolution,
        'num_epochs': len(epochs),
        'num_classes': len(reference_classes)
    }


def print_evolution_summary(aggregated_data, metric='f1_score', top_k=5):
    """
    Stampa un riepilogo testuale dell'evoluzione di una metrica.
    
    Args:
        aggregated_data: output di aggregate_multi_epoch_reports()
        metric: metrica da analizzare ('f1_score', 'precision', 'recall')
        top_k: numero di classi da mostrare (migliori/peggiori performance)
    """
    df = aggregated_data['metrics_evolution'][metric]
    epochs = aggregated_data['epochs']
    
    # Header
    header = "=" * 100
    title = f"📊 EVOLUZIONE {metric.upper().replace('_', '-')} - RIEPILOGO"
    
    print("\n" + header)
    print(title)
    print(header)
    
    # Performance iniziale vs finale
    initial_col = df.columns[0]
    final_col = df.columns[-1]
    
    df_summary = pd.DataFrame({
        'Initial': df[initial_col],
        'Final': df[final_col],
        'Delta': df[final_col] - df[initial_col],
        'Delta%': ((df[final_col] - df[initial_col]) / (df[initial_col] + 1e-10) * 100)
    })
    
    # Migliori miglioramenti
    top_improvements = df_summary.nlargest(top_k, 'Delta')[['Initial', 'Final', 'Delta']]
    print(f"\n🚀 TOP {top_k} MIGLIORAMENTI (Δ assoluto):")
    print(top_improvements)
    
    # Peggiori performance finali
    worst_performers = df_summary.nsmallest(top_k, 'Final')[['Initial', 'Final', 'Delta']]
    print(f"\n⚠️ TOP {top_k} CLASSI CON PERFORMANCE PIÙ BASSE (valore finale):")
    print(worst_performers)
    
    # Statistiche globali
    mean_initial = df[initial_col].mean()
    mean_final = df[final_col].mean()
    improvement = mean_final - mean_initial
    classes_below_threshold = (df[final_col] < 0.5).sum()
    
    print("\n📈 STATISTICHE GLOBALI:")
    print(f"   Media iniziale:  {mean_initial:.4f}")
    print(f"   Media finale:    {mean_final:.4f}")
    print(f"   Miglioramento:   {improvement:.4f}")
    print(f"   Classi < 0.5:    {classes_below_threshold} / {len(df)}")
    print(header)
    
    # ═══════════════════════════════════════════════════════════════
    # ✅ GENERAZIONE MARKDOWN
    # ═══════════════════════════════════════════════════════════════
    
    markdown_text = f"""# 📊 Evoluzione {metric.upper().replace('_', ' ')} - Riepilogo Dettagliato

    ## 🚀 Top {top_k} Miglioramenti (Δ Assoluto)

    Le classi che hanno mostrato i maggiori miglioramenti in termini assoluti:

    {top_improvements.to_markdown(floatfmt=".4f")}

    ---

    ## ⚠️ Top {top_k} Classi con Performance più Basse (Valore Finale)

    Le classi che richiedono maggiore attenzione al termine del training:

    {worst_performers.to_markdown(floatfmt=".4f")}

    ---

    ## 📈 Statistiche Globali

    | Metrica | Valore |
    |---------|--------|
    | **Media Iniziale** | {mean_initial:.4f} |
    | **Media Finale** | {mean_final:.4f} |
    | **Miglioramento Assoluto** | {improvement:.4f} |
    | **Miglioramento Percentuale** | {(improvement/mean_initial*100):.2f}% |
    | **Classi sotto soglia (F1 < 0.5)** | {classes_below_threshold} / {len(df)} |

    ---

    ## 📋 Interpretazione

    """
    
    # Aggiungi interpretazione intelligente
    if improvement > 0.1:
        markdown_text += "✅ **Miglioramento Significativo**: Il modello ha mostrato progressi sostanziali durante il training.\n\n"
    elif improvement > 0.05:
        markdown_text += "⚠️ **Miglioramento Moderato**: Il modello ha mostrato progressi, ma c'è spazio per ulteriori ottimizzazioni.\n\n"
    else:
        markdown_text += "🔴 **Miglioramento Limitato**: Il modello ha mostrato progressi minimi. Considerare strategie alternative.\n\n"
    
    if classes_below_threshold > 0:
        markdown_text += f"⚠️ **Attenzione**: {classes_below_threshold} classi hanno ancora performance sotto la soglia accettabile (0.5).\n\n"
    else:
        markdown_text += "✅ **Tutte le classi** hanno superato la soglia accettabile (0.5).\n\n"
    
    markdown_text += f"\n---\n\n*Report generato automaticamente - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    return markdown_text



def export_evolution_tables(aggregated_data, subdirs, formats=['csv', 'excel', 'markdown'], summaries=None):
    """
    Esporta le tabelle di evoluzione nelle sottocartelle dedicate.
    
    Args:
        aggregated_data: output di aggregate_multi_epoch_reports()
        subdirs: dict con i percorsi delle sottocartelle
        formats: lista formati ('csv', 'excel', 'markdown')
        summaries: dict {metric: summary_text} con i riepiloghi
    """
    metrics_evolution = aggregated_data['metrics_evolution']
    
    print(f"\n💾 Esportazione tabelle organizzate...")
    
    for metric, df in metrics_evolution.items():
        base_name = f"evolution_{metric}"
        
        if 'csv' in formats:
            csv_path = subdirs['csv'] / f"{base_name}.csv"
            df.to_csv(csv_path)
            print(f"   ✓ CSV:      {csv_path.name}")
        
        if 'excel' in formats:
            excel_path = subdirs['excel'] / f"{base_name}.xlsx"
            df.to_excel(excel_path, sheet_name=metric)
            print(f"   ✓ Excel:    {excel_path.name}")
        
        if 'markdown' in formats:
            md_path = subdirs['markdown'] / f"{base_name}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                # Header principale
                f.write(f"# Evoluzione {metric.replace('_', ' ').title()}\n\n")
                
                # Tabella completa
                f.write("## 📊 Tabella Completa Valori per Epoca\n\n")
                f.write(df.to_markdown(floatfmt=".4f"))
                f.write("\n\n---\n\n")
                
                # Riepilogo statistico
                if summaries and metric in summaries:
                    f.write(summaries[metric])
                # ✅ AGGIUNGI ANALISI CORRELAZIONE SOLO PER F1-SCORE
                if metric == 'f1_score':
                    correlation_file = subdirs['markdown'].parent / 'v0_correlation_analysis.md'
                    if correlation_file.exists():
                        with open(correlation_file, 'r', encoding='utf-8') as corr_f:
                            correlation_content = corr_f.read()
                        f.write("\n\n")
                        f.write(correlation_content)
                        print(f"   ✓ Correlazione integrata in {md_path.name}")
            
            print(f"   ✓ Markdown: {md_path.name}")
    
    print("✅ Esportazione completata!\n")


def plot_metrics_evolution_heatmap(aggregated_data, metric='f1_score', output_path=None):
    """
    Genera heatmap dell'evoluzione di una metrica usando SOLO matplotlib.
    Versione PROFESSIONALE con colormap attenuata e contrasto intelligente.
    
    Args:
        aggregated_data: output di aggregate_multi_epoch_reports()
        metric: metrica da visualizzare
        output_path: percorso file output (opzionale)
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    df = aggregated_data['metrics_evolution'][metric]
    
    # Converti DataFrame in numpy array
    data = df.values
    class_names = df.index.tolist()
    epoch_labels = df.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))
    
    # ═══════════════════════════════════════════════════════════════
    # ✅ COLORMAP PERSONALIZZATA - Tonalità Attenuate
    # ═══════════════════════════════════════════════════════════════
    
    colors_custom = [
        '#C62828',  # Rosso scuro (0.0)
        '#E57373',  # Rosso chiaro (0.25)
        '#FFB74D',  # Arancione (0.4)
        '#FFF59D',  # Giallo chiaro (0.5)
        '#AED581',  # Verde chiaro (0.7)
        '#66BB6A',  # Verde medio (0.85)
        '#388E3C'   # Verde scuro (1.0)
    ]
    
    cmap_professional = LinearSegmentedColormap.from_list('professional', colors_custom, N=256)
    
    # Applica heatmap
    vmax = 1 if metric in ['f1_score', 'precision', 'recall'] else data.max()
    im = ax.imshow(data, cmap=cmap_professional, aspect='auto', vmin=0, vmax=vmax)
    
    # Imposta tick labels
    ax.set_xticks(np.arange(len(epoch_labels)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(epoch_labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Ruota le etichette delle epoche
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # ═══════════════════════════════════════════════════════════════
    # ✅ ANNOTAZIONI CON CONTRASTO INTELLIGENTE
    # ═══════════════════════════════════════════════════════════════
    
    for i in range(len(class_names)):
        for j in range(len(epoch_labels)):
            value = data[i, j]
            
            # Logica contrasto ottimizzata
            if value < 0.30:
                text_color = 'black' #white
                font_weight = 'bold'
            elif value < 0.50:
                text_color = 'black'
                font_weight = 'bold'
            elif value < 0.75:
                text_color = 'black'  # Grigio scuro: #212121
                font_weight = 'bold'
            else:
                text_color = 'black' #white
                font_weight = 'bold'
            
            ax.text(j, i, f'{value:.3f}',
                    ha="center", va="center", 
                    color=text_color,
                    fontsize=9.5, 
                    fontweight=font_weight)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Titoli e labels
    ax.set_title(f'Evoluzione {metric.replace("_", " ").title()} per Epoca',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoca', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Classe', fontsize=13, fontweight='bold', labelpad=10)
    
    # Grid sottile
    ax.set_xticks(np.arange(len(epoch_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap salvata: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_metrics_evolution_lines(aggregated_data, metric='f1_score', 
                                 classes_to_plot=None, output_path=None):
    """
    Genera grafico a linee dell'evoluzione per classi selezionate.
    
    Args:
        aggregated_data: output di aggregate_multi_epoch_reports()
        metric: metrica da visualizzare
        classes_to_plot: lista classi da plottare (None = tutte)
        output_path: percorso file output (opzionale)
    """
    df = aggregated_data['metrics_evolution'][metric]
    epochs = aggregated_data['epochs']
    
    if classes_to_plot is None:
        # Mostra solo classi con F1 finale < 0.7 (più interessanti)
        final_col = df.columns[-1]
        classes_to_plot = df[df[final_col] < 0.7].index.tolist()
        if not classes_to_plot:
            classes_to_plot = df.index.tolist()[:5]  # Prime 5 se tutte buone
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for cls in classes_to_plot:
        values = df.loc[cls].values
        ax.plot(epochs, values, marker='o', linewidth=2.5, 
                markersize=8, label=cls, alpha=0.8)
    
    ax.set_xlabel('Epoca', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_title(f'Evoluzione {metric.replace("_", " ").title()} - Classi Selezionate',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Line plot salvato: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_evolution_analysis(folder_path, output_dir, pattern='*.txt'):
    """
    Pipeline completa: aggrega dati + esporta tabelle + genera grafici.
    """
    print("\n" + "=" * 100)
    print("🔄 ANALISI EVOLUZIONE MULTI-EPOCH")
    print("=" * 100)
    
    # Aggrega dati
    agg_data = aggregate_multi_epoch_reports(folder_path, pattern)
    
    # ✅ Genera e salva riepiloghi (ritornano anche il testo markdown)
    summaries = {}
    for metric in ['f1_score', 'precision', 'recall']:
        summary_text = print_evolution_summary(agg_data, metric=metric, top_k=5)
        summaries[metric] = summary_text
    
    # ═══════════════════════════════════════════════════════════════
    # ✅ CREA STRUTTURA SOTTOCARTELLE ORGANIZZATA
    # ═══════════════════════════════════════════════════════════════
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sottocartelle tematiche
    subdirs = {
        'csv': output_dir / 'evolution_csv',
        'excel': output_dir / 'evolution_excel',
        'markdown': output_dir / 'evolution_md',
        'heatmap': output_dir / 'heatmap_png',
        'lineplot': output_dir / 'lineplot_png'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Struttura cartelle creata:")
    for name, path in subdirs.items():
        print(f"   📁 {name:12s} → {path.name}")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 📊 GENERA GRAFICI NELLE SOTTOCARTELLE DEDICATE
    # ═══════════════════════════════════════════════════════════════
    for metric in ['f1_score', 'precision', 'recall']:
        # Heatmap
        heatmap_path = subdirs['heatmap'] / f'heatmap_{metric}.png'
        print(f"📊 Generando heatmap: {heatmap_path.name}...")
        plot_metrics_evolution_heatmap(
            agg_data, 
            metric=metric,
            output_path=heatmap_path
        )
        
        # Line plot
        lineplot_path = subdirs['lineplot'] / f'lineplot_{metric}.png'
        print(f"📈 Generando lineplot: {lineplot_path.name}...")
        plot_metrics_evolution_lines(
            agg_data,
            metric=metric,
            output_path=lineplot_path
        )
    
    print("\n" + "=" * 100)
    print("✅ ANALISI EVOLUZIONE COMPLETATA!")
    print(f"📁 Risultati salvati in: {output_dir.resolve()}")
    print("=" * 100)
    
    return agg_data, summaries, subdirs  # ✅ Ritorna anche i percorsi
# ═══════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE - GENERA TUTTI I GRAFICI
# ═══════════════════════════════════════════════════════════════════

def generate_analysis(classification_data, evolution_data, output_dir):
    """
    Genera tutti i grafici di analisi.
    
    Args:
        classification_data: dict da parse_classification_report()
        evolution_data: dict da parse_evolution_data() (opzionale)
        output_dir: Path di output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Estrai dati
    class_names = classification_data['class_names']
    supports = classification_data['supports']
    precisions = classification_data['precisions']
    recalls = classification_data['recalls']
    f1_scores = classification_data['f1_scores']
    
    # Genera abbreviazioni
    class_abbr = generate_class_abbreviations(class_names)
    
    print("=" * 70)
    print("📊 DATI CARICATI:")
    print(f"   • Numero classi: {len(class_names)}")
    print(f"   • Range support: {supports.min():,} - {supports.max():,}")
    print(f"   • F1-score medio: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════
    # GRAFICO 1: SUPPORT vs F1-SCORE
    # ═══════════════════════════════════════════════════════════════
    
    print("\n📈 Generazione grafico 1: Support vs F1-Score...")
    
    # Calcola correlazioni
    # corr_linear, p_linear = pearsonr(supports, f1_scores)
    # log_supports = np.log10(supports)
    # corr_log, p_log = pearsonr(log_supports, f1_scores)
    
    # # Interpretazione semplice (log-lineare preferita per ampie differenze di support)
    # if corr_log > 0.7:
    #     interpretation = "Interpretazione: FORTE correlazione positiva"
    # elif corr_log > 0.4:
    #     interpretation = "Interpretazione: MODERATA correlazione positiva"
    # elif corr_log > 0.2:
    #     interpretation = "Interpretazione: DEBOLE correlazione positiva"
    # else:
    #     interpretation = "Interpretazione: Correlazione TRASCURABILE"
    
    # # Salva risultati correlazione
    # with open(output_dir / 'correlation_results.txt', 'w') as f:
    #     f.write(f"Correlazione lineare: r={corr_linear:.4f}, p={p_linear:.6f}\n")
    #     f.write(f"Correlazione log-lineare: r={corr_log:.4f}, p={p_log:.6f}\n")
    #     f.write(f"Interpretazione (log10): {interpretation}\n")
    # Calcola correlazioni
    corr_linear, p_linear = pearsonr(supports, f1_scores)
    log_supports = np.log10(supports)
    corr_log, p_log = pearsonr(log_supports, f1_scores)

    # Interpretazione semplice
    if corr_log > 0.7:
        interpretation = "FORTE correlazione positiva"
        icon = "✅"
    elif corr_log > 0.4:
        interpretation = "MODERATA correlazione positiva"
        icon = "⚠️"
    elif corr_log > 0.2:
        interpretation = "DEBOLE correlazione positiva"
        icon = "⚠️"
    else:
        interpretation = "Correlazione TRASCURABILE"
        icon = "🔴"

    # ═══════════════════════════════════════════════════════════════
    # ✅ ESPORTA CORRELAZIONE IN TXT E MARKDOWN
    # ═══════════════════════════════════════════════════════════════

    # File TXT (formato classico)
    with open(output_dir / 'v0_correlation_results.txt', 'w') as f:
        f.write(f"Pearson linear: r={corr_linear:.4f}, p={p_linear:.6f}\n")
        f.write(f"Pearson log10: r={corr_log:.4f}, p={p_log:.6f}\n")
        f.write(f"Interpretazione (log10): {interpretation}\n")

    # File MARKDOWN (per integrare con evolution_f1)
    correlation_md = f"""
    ---

    ## 🔗 Analisi Correlazione Support vs F1-Score

    ### Risultati Statistici

    | Metodo | Coefficiente (r) | p-value | Significatività |
    |--------|------------------|---------|-----------------|
    | **Pearson Lineare** | {corr_linear:.4f} | {p_linear:.6f} | {'✅ Significativa' if p_linear < 0.05 else '❌ Non significativa'} |
    | **Pearson Log₁₀** | {corr_log:.4f} | {p_log:.6f} | {'✅ Significativa' if p_log < 0.05 else '❌ Non significativa'} |

    ### {icon} Interpretazione

    **{interpretation}**

    La correlazione log₁₀ è più appropriata per dataset con ampia variazione di support (ratio {supports.max() / supports.min():.1f}×).

    {'⚠️ **Nota**: Una correlazione debole/moderata suggerisce che lo sbilanciamento delle classi impatta negativamente le performance.' if corr_log < 0.7 else '✅ **Nota**: Il modello gestisce bene lo sbilanciamento delle classi.'}

    ---

    *Correlazione calcolata su {len(class_names)} classi - Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """

    with open(output_dir / 'v0_correlation_analysis.md', 'w', encoding='utf-8') as f:
        f.write(correlation_md)

    print(f"✅ Correlazione salvata:")
    print(f"   📄 v0_correlation_results.txt")
    print(f"   📄 v0_correlation_analysis.md")
    
    # Identifica classi critiche
    critical_mask = f1_scores < 0.5
    critical_classes = [(i+1, cls, supp, f1) for i, (cls, supp, f1) in 
                        enumerate(zip(class_names, supports, f1_scores)) if f1 < 0.5]
    critical_classes.sort(key=lambda x: x[3])

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
    plt.savefig(output_dir / 'v0_support_vs_f1_IMPROVED.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("✅ Grafico migliorato salvato con successo!")
    print(f"   → {output_dir / 'v0_support_vs_f1.png'}")
    
    print("✅ Grafico 1 salvato: support_vs_f1.png")
    
    # ═══════════════════════════════════════════════════════════════
    # GRAFICO 2: PRECISION-RECALL
    # ═══════════════════════════════════════════════════════════════
    
    print("\n📈 Generazione grafico 2: Precision-Recall...")
    
    # [QUI INSERISCI IL CODICE DEL GRAFICO PRECISION-RECALL]
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
    # SALVATAGGIO
    # ═══════════════════════════════════════════════════════════════════

    plt.savefig(output_dir/'v0_precision_recall.png',
            dpi=400, bbox_inches='tight', facecolor='white')

    
    print("✅ Grafico 2 salvato: precision_recall.png")
    
    # ═══════════════════════════════════════════════════════════════
    # GRAFICO 3: EVOLUTION (se disponibile)
    # ═══════════════════════════════════════════════════════════════
    
    if evolution_data:
        print("\n📈 Generazione grafico 3: F1 Evolution...")
        
        epochs = evolution_data['epochs']
        class_evolution = evolution_data['class_evolution']
        
        # [QUI INSERISCI IL CODICE DEL GRAFICO EVOLUTION]
        # Usa epochs e class_evolution invece dei valori hard-coded
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

        plt.savefig(output_dir/'v0_f1_evolution.png',
                dpi=400, bbox_inches='tight', facecolor='white')


        
        print("✅ Grafico 3 salvato: f1_evolution.png")
    else:
        print("\n⚠️ Dati evolution non disponibili - grafico 3 saltato")
    
    print("\n" + "=" * 70)
    print("✅ ANALISI COMPLETATA!")
    print(f"📁 File salvati in: {output_dir}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# MAIN - ENTRY POINT CON ARGPARSE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Analisi parametrica performance modello ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:

  # Analisi singolo report
  python analysis_parametric.py --classification report.json --output ./results

  # Analisi evoluzione multi-epoch
  python analysis_parametric.py --evolution-folder ./reports --output ./results

  # Analisi completa (singolo + evoluzione)
  python analysis_parametric.py --classification report.txt --evolution evolution.csv --evolution-folder ./reports --output ./results
        """
    )
    
    parser.add_argument('--classification', '-c', default=None,
                        help='File classification report singolo (JSON o TXT formato sklearn)')
    
    parser.add_argument('--evolution', '-e', default=None,
                        help='File dati evolution (CSV, TXT o JSON) - opzionale')
    
    parser.add_argument('--evolution-folder', '-ef', default=None,
                        help='Cartella con multipli classification report per tracking evoluzione')
    
    parser.add_argument('--pattern', '-p', default='*.txt',
                        help='Pattern matching file nella evolution-folder (default: *.txt)')
    
    # parser.add_argument('--output', '-o', default='./analysis_results',
    #                     help='Directory output (default: ./analysis_results)')
    parser.add_argument('--output', '-o', 
                    default='./analysis_results',
                    help='Directory output (default: ./analysis_results)')
    
    parser.add_argument('--export-formats', nargs='+', 
                        default=['csv', 'excel', 'markdown'],
                        choices=['csv', 'excel', 'latex', 'markdown'],
                        help='Formati export tabelle evoluzione (default: csv excel markdown)')
    
    args = parser.parse_args()
    
    # Verifica che almeno un input sia fornito
    if not args.classification and not args.evolution_folder:
        parser.error("Devi fornire almeno uno tra --classification o --evolution-folder")
    
    # ✅ ✅ ✅ AGGIUNTA FONDAMENTALE: CONVERSIONE PERCORSO RELATIVO → ASSOLUTO ✅ ✅ ✅
    output_path = Path(args.output).resolve()
    print("\n" + "=" * 80)
    print("📁 CONFIGURAZIONE PERCORSI:")
    print(f"   Directory di lavoro corrente: {Path.cwd()}")
    print(f"   Output richiesto (raw):       {args.output}")
    print(f"   Output assoluto (REALE):      {output_path}")
    print("=" * 80 + "\n")
    
    # Sovrascrive args.output con il percorso assoluto
    args.output = str(output_path)
    
    # Configurazione matplotlib
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    
    try:
        # ═══════════════════════════════════════════════════════════
        # ANALISI EVOLUZIONE MULTI-EPOCH (se richiesta)
        # ═══════════════════════════════════════════════════════════
        
        if args.evolution_folder:
            print("\n🔄 MODALITÀ: Analisi Evoluzione Multi-Epoch")
            
            evolution_output = Path(args.output) / 'evolution_analysis'
            agg_data, summaries, subdirs = generate_evolution_analysis(
                args.evolution_folder,
                evolution_output,
                pattern=args.pattern
            )
            
            export_evolution_tables(agg_data, subdirs,
                                formats=args.export_formats, 
                                summaries=summaries)
            
            # ═══════════════════════════════════════════════════════════════
            # ✅ GENERA ANCHE I 3 GRAFICI CLASSICI (usando l'ultima epoca)
            # ═══════════════════════════════════════════════════════════════
            
            print("\n📊 MODALITÀ: Analisi Singolo Report (Ultima Epoca)")
            
            # Estrai dati dell'ultima epoca
            final_epoch = agg_data['epochs'][-1]
            final_classification_data = {
                'class_names': agg_data['classes'],
                'f1_scores': agg_data['metrics_evolution']['f1_score'].iloc[:, -1].values,
                'precisions': agg_data['metrics_evolution']['precision'].iloc[:, -1].values,
                'recalls': agg_data['metrics_evolution']['recall'].iloc[:, -1].values,
                'supports': agg_data['metrics_evolution']['support'].iloc[:, -1].values.astype(int),
                'epoch': final_epoch
            }
            
            # Crea sottocartella per grafici singoli
            single_report_output = evolution_output / 'single_epoch_analysis'
            single_report_output.mkdir(parents=True, exist_ok=True)
            
            # Genera i 3 grafici classici
            print(f"📊 Generando grafici per Epoch {final_epoch}...")
            generate_analysis(final_classification_data, None, single_report_output)
            
            print(f"✅ Grafici singola epoca salvati in: {single_report_output}")

        
        # ═══════════════════════════════════════════════════════════
        # ANALISI SINGOLO REPORT (se richiesta)
        # ═══════════════════════════════════════════════════════════
        
        if args.classification:
            print("\n📊 MODALITÀ: Analisi Singolo Report")
            
            # Parse classification report
            print("📄 Caricamento classification report...")
            classification_data = parse_classification_report(args.classification)
            
            # Parse evolution data (opzionale)
            evolution_data = None
            if args.evolution:
                print("📄 Caricamento dati evolution...")
                evolution_data = parse_evolution_data(args.evolution)
            
            # Genera analisi
            single_output = Path(args.output) / 'single_report_analysis'
            generate_analysis(classification_data, evolution_data, single_output)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())