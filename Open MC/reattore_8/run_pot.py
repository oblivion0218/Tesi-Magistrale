import pandas as pd
import papermill as pm
import os

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

# Definizione della riga di partenza (0-based: 3 corrisponde alla quarta riga dei dati validi)
riga_partenza = 2

# 2. Esecuzione ciclica con slicing del DataFrame
for idx, row in df_results.iloc[riga_partenza:].iterrows():
    
    # Condizione di skip se arricch_INT è 0
    if row['arricch_INT'] == 0.0:
        print(f"Skipping simulazione alla riga {idx}: arricch_INT è 0.")
        continue

    # ID univoco per il salvataggio dei file
    run_id = f"arr_ext_{row['arricch_EXT']:.0f}_pitch_{row['moltiplicatore']:.0f}"
    
    print(f"Avvio simulazione fixed: {run_id} ...")

    pm.execute_notebook(
        notebook_in,
        os.devnull,
        cwd='fixed/statepoints',
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