import torch
import json
import os
import sys
import glob
from typing import Type, TypeVar, Any, Dict, List

# --- DEFINIZIONE DEI TIPI ---
T = TypeVar('T')

class CheckpointInspector:
    """
    Classe responsabile del caricamento e dell'ispezione di un singolo checkpoint PyTorch.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = self._load_data()
        
    def _load_data(self) -> Any:
        # print(f"[*] Analisi file: {self.filename} ...")
        if not os.path.exists(self.filepath):
            print(f"[!] Errore: Il file {self.filepath} non esiste.")
            return None
            
        try:
            # --- MODIFICA FONDAMENTALE QUI SOTTO ---
            # map_location='cpu': per caricare su CPU se non hai GPU
            # weights_only=False: DISABILITA la protezione di PyTorch 2.6+ per permettere 
            # il caricamento di oggetti Numpy (loss, accuracy) salvati nel checkpoint.
            # USALO SOLO CON FILE DI CUI TI FIDI (come i tuoi).
            payload = torch.load(self.filepath, map_location=torch.device('cpu'), weights_only=False)
            return payload
        except Exception as e:
            # Se fallisce, ritorniamo None e gestiamo l'errore dopo
            print(f"[!] Errore critico caricando {self.filename}: {e}")
            return None

    def get(self, key: str, typ: Type[T]) -> T:
        """
        Recupera un valore dal dizionario, assicurandosi che sia del tipo richiesto.
        """
        if self.data is None:
            return None
            
        if not isinstance(self.data, dict):
            return None

        value = self.data.get(key)
        
        if value is None:
            return None
            
        if not isinstance(value, typ):
            return None # type: ignore
            
        return value

    def analyze_optimizer(self, optimizer_state: Dict) -> Dict:
        """Estrae LR e parametri dall'optimizer state dict."""
        info = {}
        if 'param_groups' in optimizer_state:
            for idx, group in enumerate(optimizer_state['param_groups']):
                info[f'lr_group_{idx}'] = group.get('lr', 'N/A')
                info[f'weight_decay_group_{idx}'] = group.get('weight_decay', 'N/A')
                info[f'momentum_group_{idx}'] = group.get('momentum', 'N/A')
        return info

    def extract_info(self) -> Dict:
        """
        Metodo principale che orchestra l'estrazione delle informazioni.
        """
        # Se il caricamento è fallito (self.data è None), ritorniamo un dizionario di errore
        if self.data is None:
            return {"filename": self.filename, "error": "Caricamento fallito"}

        report = {
            "filename": self.filename,
            "file_size_mb": round(os.path.getsize(self.filepath) / (1024 * 1024), 2),
            "content_type": str(type(self.data)),
            "hyperparameters": {},
            "model_structure": "Unknown"
        }

        # 1. Tentativo di estrazione pulita tramite .get() per scalari noti
        potential_keys = {
            "round": int, "round_num": int, "current_round": int,
            "epoch": int, "epochs": int,
            "loss": float, "train_loss": float, "best_loss": float,
            "accuracy": float, "acc": float, "best_acc": float
        }

        for key_name, key_type in potential_keys.items():
            val = self.get(key_name, key_type)
            if val is not None:
                report["hyperparameters"][key_name] = val

        # 2. Analisi profonda se è un dizionario
        if isinstance(self.data, dict):
            # Cerca Optimizer
            opt = self.data.get('optimizer_state_dict') or self.data.get('optimizer')
            if opt:
                report["hyperparameters"]["optimizer_analysis"] = self.analyze_optimizer(opt)
            
            # Cerca Architettura Modello
            model = self.data.get('model_state_dict') or self.data.get('model')
            if model and isinstance(model, dict):
                layer_names = list(model.keys())
                report["model_structure"] = {
                    "total_layers": len(layer_names),
                    "input_layer_hint": layer_names[0] if layer_names else "None",
                    "output_layer_hint": layer_names[-1] if layer_names else "None",
                    "contains_classifier_head": any("classifier" in k for k in layer_names),
                    "contains_feature_extractor": any("feature_extractor" in k for k in layer_names)
                }
        
        # A volte la loss o l'accuracy sono salvate come numpy scalar fuori dal dict principale?
        # Non possiamo saperlo con certezza senza vedere il codice di salvataggio, 
        # ma weights_only=False risolve il caricamento.

        return report

def scan_folder(folder_path: str):
    """
    Scansiona una cartella alla ricerca di file .pt/.pth e genera un report JSON unico.
    """
    print(f"--- START: Scansione cartella '{folder_path}' ---\n")
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] La cartella '{folder_path}' non esiste.")
        return

    extensions = ['*.pt', '*.pth']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not files:
        print("[WARNING] Nessun file .pt o .pth trovato nella cartella.")
        return

    # Ordina i file
    files.sort()
    
    full_report = []

    print(f"Trovati {len(files)} checkpoint. Inizio analisi...\n")

    for f_path in files:
        inspector = CheckpointInspector(f_path)
        file_info = inspector.extract_info()
        full_report.append(file_info)
        
        fname = os.path.basename(f_path)
        
        # --- FIX KEYERROR: Controllo se c'è stato un errore nel caricamento ---
        if "error" in file_info:
            print(f" -> {fname} | [!] ERRORE: {file_info['error']}")
            continue # Salta al prossimo file
            
        lr_info = "N/A"
        # Accesso sicuro ai dati
        if "optimizer_analysis" in file_info.get("hyperparameters", {}):
             opt_an = file_info["hyperparameters"]["optimizer_analysis"]
             lr_info = opt_an.get('lr_group_0', 'N/A')

        keys_found = list(file_info.get('hyperparameters', {}).keys())
        print(f" -> {fname} | LR: {lr_info} | Keys trovate: {keys_found}")

    output_json = "analysis_report.json"
    with open(output_json, 'w') as f:
        json.dump(full_report, f, indent=4)

    print(f"\n--- DONE. Report completo salvato in '{output_json}' ---")

# --- CONFIGURAZIONE ---
if __name__ == "__main__":
    # Percorso che hai indicato nell'errore
    # TARGET_FOLDER = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs_federated/Log_to_final/log_pseudo-final5_attacks/ddos_hoic/sixth_try_CB"
    # TARGET_FOLDER = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs_federated/Classic_FedAvg/log_classic/ddos_loic_udp"
    TARGET_FOLDER = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs_federated/Log_to_final/log6/ddos_hoic/smartCL/4_smartCL_100_LR0005"
    
    scan_folder(TARGET_FOLDER)