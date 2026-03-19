import papermill as pm
import os
import numpy as np
from bayes_opt import BayesianOptimization

### Configurazione Variabili Globali
NOTEBOOK_IN = 'auto.ipynb'
RESULT_FILE = 'result.txt'

# Variabile di stato persistente tra le chiamate della funzione obiettivo
is_first_run_global = True

def get_last_k(filename):
    """
    Legge l'ultima riga valida di result.txt e restituisce k_eff.
    """
    try:
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            lines = f.readlines()
        
        # Filtra righe numeriche (evita header)
        data_lines = [l for l in lines if len(l.split()) >= 5 and l.split()[0].replace('.', '', 1).replace('-', '', 1).isdigit()]
        
        if not data_lines:
            return None
        # Restituisce il k_eff dell'ultima riga (indice 4)
        return float(data_lines[-1].split()[4])
    except Exception as e:
        print(f"Errore parsing: {e}")
        return None

def objective_function(moltiplicatore, pressione, water_perc):
    global is_first_run_global
    
    print(f"\n>>> Iterazione ML: Pitch_Mult={moltiplicatore:.4f}, Pres_Mult={pressione:.2f}, perc_water = {water_perc:.2f}")
    print(f">>> is_first_run = {is_first_run_global}")
    
    # Esecuzione Notebook
    pm.execute_notebook(
        NOTEBOOK_IN,
        os.devnull,
        parameters=dict(
            moltiplicatore=float(moltiplicatore), 
            pressione=float(pressione), 
            water_perc=float(water_perc), 
            is_first_run=is_first_run_global
        )
    )

    # Dopo la prima esecuzione, setta a False permanentemente per questa sessione
    is_first_run_global = False
    
    k_czp = get_last_k(RESULT_FILE)
    
    if k_czp is None:
        return -10.0

    # Logica di massimizzazione k_eff < 1.0 (sottocriticità)
    if k_czp >= 1.0:
        return -15.0 * k_czp  # Penalità per superamento criticità
    else:
        return k_czp # Reward positivo: più è vicino a 1, meglio è


### Inizializzazione Ottimizzatore

pbounds = {
    'moltiplicatore': (0.5, 1.1),
    'pressione': (1.0, 1.0),
    'water_perc': (0.05 , 0.85)
}

optimizer = BayesianOptimization(
    f=objective_function, 
    pbounds=pbounds, 
    random_state=42,
    verbose=2
)

# Avvio ottimizzazione
optimizer.maximize(init_points=5, n_iter=50)

print(f"\nConfigurazione Ottimale: {optimizer.max}")