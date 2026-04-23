import pandas as pd
import papermill as pm
import os
import glob

def parse_result_final(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Salta intestazioni e righe vuote
            if not line or line.startswith('-') or line.startswith('=') or 'Pres [Atm]' in line:
                continue
            cols = line.split()
            if len(cols) == 12:
                try:
                    data.append(list(map(float, cols)))
                except ValueError:
                    continue
                    
    columns = ['pressione_atm', 'moltiplicatore', 'perc_water', 'arricch_INT', 'arricch_EXT', 't_water', 't_fuel', 'k_max', 'std_k_max', 'k_auto', 'std_k_auto', 'compatibilita']
    return pd.DataFrame(data, columns=columns)

# 1. Lettura dei risultati
df_results = parse_result_final('result_final.txt')
notebook_in = 'fixed/fixed.ipynb'
output_dir = 'fixed/statepoints'

# Crea la directory se non esiste
os.makedirs(output_dir, exist_ok=True)

# 2. Esecuzione ciclica a partire dal FONDO del DataFrame
# Invertiamo l'ordine delle righe con lo slicing [::-1]
for idx, row in df_results[::-1].iterrows():
    
    # Condizione di skip se arricch_INT è 0
    if row['arricch_INT'] == 0.0:
        print(f"Skipping simulazione alla riga {idx}: arricch_INT è 0.")
        continue

    # ID univoco per il salvataggio dei file
    run_id = f"arr_ext_{row['arricch_EXT']:.1f}_pitch_{row['moltiplicatore']:.1f}"
    
    # 3. Controllo file esistenti tramite matching parziale del nome
    file_esistenti = glob.glob(os.path.join(output_dir, f"*{run_id}*"))
    if file_esistenti:
        print(f"Skipping {run_id}: file già presenti.")
        continue
        
    print(f"Avvio simulazione fixed: {run_id} (riga {idx})...")

    pm.execute_notebook(
        notebook_in,
        os.devnull,
        cwd=output_dir,
        parameters=dict(
            pressione_atm=row['pressione_atm'],
            moltiplicatore=row['moltiplicatore'],
            perc_water=row['perc_water'],
            arricch_max=row['arricch_INT'], 
            arricch_min=row['arricch_EXT'], 
            t_water=row['t_water'],
            t_fuel=row['t_fuel'],
            run_id=run_id
        )
    )

print("Tutte le simulazioni completate.")