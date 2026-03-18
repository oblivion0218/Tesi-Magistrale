import papermill as pm
import os
from bayes_opt import BayesianOptimization

NOTEBOOK_IN = 'fixed.ipynb'
RESULT_FILE = 'result_fixed.txt'

# Rilevamento automatico per l'header del file di testo
is_first_run = not (os.path.exists(RESULT_FILE) and os.path.getsize(RESULT_FILE) > 0)

def preload_data(optimizer, filename):
    """
    Legge lo storico simulazioni e inietta i punti campionati nel Processo Gaussiano.
    """
    if not os.path.exists(filename):
        return
    
    print(f"=== Caricamento dati pregressi da {filename} ===")
    count = 0
    with open(filename, "r") as f:
        for l in f:
            parts = l.split()
            # Identifica le righe numeriche valide
            if len(parts) >= 3 and parts[0].replace('.', '', 1).isdigit():
                try:
                    molt = float(parts[0])
                    mix = float(parts[1])
                    fiss = float(parts[2])
                    
                    # Registra il punto nell'ottimizzatore
                    optimizer.register(
                        params={"moltiplicatore": molt, "mix_acqua": mix},
                        target=fiss
                    )
                    count += 1
                except KeyError:
                    # Ignora duplicati esatti sollevati dalla libreria
                    pass
    print(f"Caricati {count} punti pregressi con successo.\n")

def get_last_fission(filename):
    """
    Legge l'ultima riga valida di result_fixed.txt e restituisce fission_per_source.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        data_lines = [l for l in lines if len(l.split()) >= 3 and l.split()[0].replace('.', '', 1).isdigit()]
        if not data_lines:
            return None
        return float(data_lines[-1].split()[2])
    except Exception as e:
        print(f"Errore parsing: {e}")
        return None

def objective_function(moltiplicatore, mix_acqua):
    global is_first_run
    
    print(f"\n>>> Iterazione ML: Pitch_Mult={moltiplicatore:.4f}, Mix_Acqua={mix_acqua:.3f}")
    
    try:
        pm.execute_notebook(
            NOTEBOOK_IN,
            os.devnull, 
            parameters=dict(
                moltiplicatore=float(moltiplicatore), 
                mix_acqua=float(mix_acqua),
                is_first_run=bool(is_first_run)
            )
        )
        if is_first_run:
            is_first_run = False
            
    except Exception as e:
        print(f"Errore Papermill durante l'esecuzione: {e}")
        return -1.0 

    fission_per_source = get_last_fission(RESULT_FILE)
    
    if fission_per_source is None:
        return -1.0

    return fission_per_source

if __name__ == "__main__":
    pbounds = {
        'moltiplicatore': (0.8, 1),
        'mix_acqua': (0.6, 0.8)      
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    # Inietta i dati storici prima di iniziare
    preload_data(optimizer, RESULT_FILE)

    print("=== Avvio Ottimizzazione Bayesiana per Massimizzazione Potenza ADS ===")
    
    # Se hai già molti punti pregressi caricati, puoi azzerare init_points 
    # per far partire subito il modello matematico al posto dell'esplorazione randomica pura.
    punti_randomici_aggiuntivi = 0 if len(optimizer.space) > 5 else 5

    optimizer.maximize(
        init_points=punti_randomici_aggiuntivi,  
        n_iter=10
    )

    print("\n======================================")
    print("OTTIMIZZAZIONE COMPLETATA")
    print(f"Miglior configurazione trovata:")
    print(optimizer.max)
    print("======================================")