import papermill as pm
import os
import numpy as np
from bayes_opt import BayesianOptimization

NOTEBOOK_IN = 'auto.ipynb'
RESULT_FILE = 'result.txt'

def get_last_k_czp(filename):
    """
    Legge l'ultima riga valida di result.txt e restituisce k_eff.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        # Filtra righe numeriche (evita header)
        data_lines = [l for l in lines if len(l.split()) >= 5 and l.split()[0].replace('.', '', 1).isdigit()]
        if not data_lines:
            return None
        # Restituisce il k_eff dell'ultima riga (colonna 4)
        return float(data_lines[-1].split()[4])
    except Exception as e:
        print(f"Errore parsing: {e}")
        return None

def objective_function(moltiplicatore, pressione):
    print(f"\n>>> Iterazione ML: Pitch_Mult={moltiplicatore:.4f}, Pres_Mult={pressione:.2f}")
    
    pm.execute_notebook(
        NOTEBOOK_IN,
        os.devnull,
        parameters=dict(moltiplicatore=float(moltiplicatore), pressione=float(pressione), is_first_run=False)
    )

    k_czp = get_last_k_czp(RESULT_FILE)
    
    if k_czp is None:
        return -10.0

    # massimizzare k_czp mantenendolo < 1.0
    if k_czp >= 1.0:
        return -15.0 * k_czp  # Penalità per superamento criticità
    else:
        return k_czp # Reward positivo più è vicino a 1


pbounds = {
    'moltiplicatore': (0.98, 1.12), # Range ristretto attorno alla zona di interesse
    'pressione': (1.0, 150.0)
}

optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)

optimizer.maximize(init_points=5, n_iter=25)

print(f"\nConfigurazione Ottimale: {optimizer.max}")