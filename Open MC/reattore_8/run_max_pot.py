import pandas as pd
import papermill as pm
import os
import numpy as np
from datetime import datetime

# Lettura di file 

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
    df = pd.DataFrame(data, columns=columns)
    return df

# --------------------------------------------------------------

def parse_result_auto(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if not line or line.startswith('-') or line.startswith('=') or \
               'intensità' in line or 'Arricchimento' in line or \
               'Target' in line or 'Settings' in line or 'Pres [Pa]' in line:
                continue
            
            cols = line.split()
            if len(cols) == 12:
                try:
                    row_data = list(map(float, cols))
                    data.append(row_data)
                except ValueError:
                    continue
                    
    columns = ['Pres_Pa', 'Pitch_cm', 'Perc_water','arricchimento_max', 'arricchimento_min', 'T_water_K', 'T_fuel_K', 'k_eff', 'std_dev', 'k_max', 'k_max_std','compatibility']
    df = pd.DataFrame(data, columns=columns)
    return df

now = datetime.now()
output_file = "result_final.txt"  

timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")

# Header Result Final
with open(output_file, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI DI SIMULAZIONE ===\n")
    f_out.write(f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. MAX':<12} {'Arricch.min':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n")

df_kmax = parse_kmax_file('../reattore_5/auto/k_max.txt')

is_first = True
i = 0

now = datetime.now()
output_file = "result_final.txt"  

timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")

# Header Result Final
with open(output_file, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI DI SIMULAZIONE ===\n")
    f_out.write(f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. MAX':<12} {'Arricch.min':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n")

df_kmax = parse_kmax_file('../reattore_5/auto/k_max.txt')

is_first = True

# Inizializzazione globale dello stato per WARM START
a_M_opt = 0.01  # primo anello
a_m_opt = 0.01  # ultimo anello

for i, k in enumerate(df_kmax['k_max']):

    std_k_max = df_kmax['sigma_k'].iloc[i]
    m = df_kmax['Pitch'].iloc[i] 
    p = df_kmax['Pressione_atm'].iloc[i]
    w = 0.15    # percentuale acqua
    
    # 1. WARM START: Partenza ESATTA dalla soluzione ottimale precedente 
    a_M = a_M_opt
    a_m = a_m_opt
    
    notebook_in = 'auto/auto.ipynb'
    
    best_delta_k = np.inf
    best_line = ""
    is_compatible = False

    while not is_compatible and a_m <= 0.20:

        print(f"\nAvvio simulazione: moltiplicatore={m:.3f}, pressione={p:.0f}, perc_water={w*100:.0f}%, arric. MAX={a_M*100:.0f}%, arric. MIN={a_m*100:.0f}% ...")
        
        # Chiamata unica a Papermill
        pm.execute_notebook(
            notebook_in,
            os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_M, is_first_run=is_first, iter=i)
        )

        df_results = parse_result_auto('result_auto.txt')
        idx = len(df_results) - 1  
        
        k_auto = df_results['k_eff'].iloc[idx] 
        std_k_auto = df_results['std_dev'].iloc[idx]
        T_water = df_results['T_water_K'].iloc[idx]
        T_fuel = df_results['T_fuel_K'].iloc[idx]

        # Calcolo metriche
        delta_k_abs = np.abs(k - k_auto)
        
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M:<12.2f} {a_m:<12.2f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"

        # Tracking configurazione ottimale (basato sul residuo minimo e assicurando k_auto < 1)
        if delta_k_abs < best_delta_k and k_auto < 1.0:
            best_delta_k = delta_k_abs
            best_line = current_line

        # 2. Logica di aggiustamento
        if delta_k_abs < 0.001: 
            # SUCCESS: Salva e chiudi
            is_compatible = True
            a_M_opt, a_m_opt = a_M, a_m
            break
            
# C'è QUALCOSA CHE NON MI CONVINCE IN QUESTI DUE ELIF

        elif k_auto > k:
            # OVERSHOOT: Siamo troppo alti
            a_M = max(0.01, round(a_M - 0.02, 2)) 
            a_m = min(0.20, round(a_m + 0.01, 2))
                
        else:
            # UNDERSHOOT: Siamo troppo bassi, sali
            delta = k - k_auto
            step = 0.04 if delta >= 0.1 else 0.01
            
            if a_M < 0.20:
                a_M = min(0.20, round(a_M + step, 2))
            else: # a_M = 0.20
                a_m = round(a_m + 0.01, 2)
                a_M = 0.15 # Riparti da un gradiente logico

        is_first = False 

    # 5. Uscita limite per saturazione spazio di ricerca (a_m > 0.20)
    if not is_compatible and best_line:
        with open(output_file, "a") as f_out:
            f_out.write(best_line)

os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")
