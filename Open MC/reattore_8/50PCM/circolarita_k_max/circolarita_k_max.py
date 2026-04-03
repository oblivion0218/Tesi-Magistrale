import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import papermill as pm
import os

def analizza(file_path):
    """
    Legge i dati di simulazione, esegue regressione parabolica ESCLUDENDO i punti con y=0,
    e genera il plot differenziando i marcatori per y=0.
    """
    # Definizione manuale delle colonne
    nomi_colonne = [
        'Pres', 'moltiplicatore', 'PERC_water', 'Arricch_INT',
        'Arricch_EXT', 'T_WATER', 'T_FUEL', 'K_max', 'std_k_max',
        'K_auto', 'std_k_auto', 'Compatibilità'
    ]
    
    # Lettura del file
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=3, names=nomi_colonne, engine='python')
    
    # Estrazione array completi per la visualizzazione
    x_tutti = df['Arricch_EXT'].values
    y_tutti = df['Arricch_INT'].values

    maschera_validi = x_tutti < 10

    x_validi = x_tutti[maschera_validi]
    y_validi = y_tutti[maschera_validi]

    return x_validi, y_validi
    

def parse_kmax_file(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
            
        cols = [col.strip() for col in line.split('|')]
        
        if len(cols) == 4:
            pressione = float(cols[0])
            pitch = float(cols[1])
            
            k_max, sigma_k = map(float, cols[2].split('+/-'))
            delta_rho, err_rho = map(float, cols[3].split('+/-'))
            
            data.append([pressione, pitch, k_max, sigma_k, delta_rho, err_rho])
            
    columns = ['Pressione_atm', 'Pitch', 'k_max', 'sigma_k', 'Delta_rho_pcm', 'err_rho_pcm']
    return pd.DataFrame(data, columns=columns)


# -------- CODE -----------

arr_ext , arr_int = analizza ('../result_final.txt')

df_kmax = parse_kmax_file('../k_max.txt')

i = 6

k = df_kmax['k_max'].iloc[i]
std_k_max = df_kmax['sigma_k'].iloc[i]
m = df_kmax['Pitch'].iloc[i] 
p = df_kmax['Pressione_atm'].iloc[i]
w = 0.15    # percentuale acqua

notebook_in = 'auto.ipynb'
is_first = True

for a_ext in arr_ext:
    
    a_int = arr_int[arr_ext == a_ext][0]  # Trova il corrispondente arricchimento interno
    print(f"Simulazione con arricchimento esterno: {a_ext}, arricchimento interno: {a_int}")
    
    # Casting esplicito dei tipi NumPy in tipi nativi Python per la serializzazione JSON
    pm.execute_notebook(
            notebook_in,
            os.devnull,
            parameters=dict(
                moltiplicatore=float(m), 
                pressione=float(p), 
                water_perc=float(w), 
                enrich_min=float(a_ext/100), 
                enrich_max=float(a_int/100), 
                is_first_run=bool(is_first), 
                iter=int(i)
            )
    )
    is_first = False