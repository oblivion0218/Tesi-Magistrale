# DEVO FARE BISEZIONE PER OGNI VALORE INTERO DI a_m su un SOLO dato
# CERCANDO LA COMPATIBILITÀ CON K_MAX

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
    return pd.DataFrame(data, columns=columns)

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
                    data.append(list(map(float, cols)))
                except ValueError:
                    continue
                    
    columns = ['Pres_Pa', 'Pitch_cm', 'Perc_water','arricchimento_max', 'arricchimento_min', 'T_water_K', 'T_fuel_K', 'k_eff', 'std_dev', 'k_max', 'k_max_std','compatibility']
    return pd.DataFrame(data, columns=columns)


is_first = True
i = 6 # sesto elemento pitch = 0.1

now = datetime.now()
output_file = "result_final.txt"  

timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")

# Header Result Final
with open(output_file, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI DI SIMULAZIONE ===\n")
    f_out.write(f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. INT':<12} {'Arricch. EXT':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n")


df_kmax = parse_kmax_file('k_max.txt')

k = df_kmax['k_max'].iloc[i]
std_k_max = df_kmax['sigma_k'].iloc[i]
m = df_kmax['Pitch'].iloc[i] 
p = df_kmax['Pressione_atm'].iloc[i]
w = 0.15    # percentuale acqua

notebook_in = 'auto/auto.ipynb'

a_m = 0.11          # a_m rappresenta l'arricimento più esterno
ARR_MAX = 0.2

while a_m <= ARR_MAX:
    
    ## --- 1. TEST PRELIMINARE CON a_M = 0.20 ---
    #a_M_test = ARR_MAX
    #print(f"\n[TEST PRELIMINARE] m={m:.3f}, p={p:.0f}, w={w*100:.0f}%, a_M={a_M_test*100:.2f}%, a_m={a_m*100:.0f}% ...")
    #
    #pm.execute_notebook(
    #    notebook_in,
    #    os.devnull,
    #    parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_M_test, is_first_run=is_first, iter=i)
    #)
    #is_first = False  
#
    #df_results = parse_result_auto('result_auto.txt')
    #idx = len(df_results) - 1  
    #
    #k_auto = df_results['k_eff'].iloc[idx] 
    #std_k_auto = df_results['std_dev'].iloc[idx]
    #T_water = df_results['T_water_K'].iloc[idx]
    #T_fuel = df_results['T_fuel_K'].iloc[idx]
    #sigma = df_results['compatibility'].iloc[idx] 
#
    #delta_k_abs = np.abs(k - k_auto)
    #current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M_test*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
#
    ## Se k_auto è minore del target o già in tolleranza, salto la bisezione
    #if k_auto < k or delta_k_abs <= 0.0015:
    #    if k_auto < k:
    #        print(f"-> SKIP BISEZIONE: k_auto ({k_auto:.5f}) < k_max ({k:.5f}) con arricchimento massimo.")
    #    else:
    #        print(f"-> CONVERGENZA IMMEDIATA al test preliminare. Scarto = {delta_k_abs:.5f}")
    #        
    #    with open(output_file, "a") as f_out:
    #        f_out.write(current_line)
    #    
    #    a_m = round(a_m + 0.01, 2)
    #    continue

    # --- 2. INIZIALIZZAZIONE E CICLO DI BISEZIONE ---
    a_M_min = 0
    a_M_max = ((0.101 * (a_m*100)**2 - 2.791*100*a_m + 18.043)*2)/100   # da interpolazione parabolica 

    if a_M_max <= 0:
        a_M_max = 0
        max_bisection_iters = 1
        print(f"-> ATTENZIONE: a_M_max <= 0 per a_m = {a_m*100:.0f}%. Imposto a_M_max = 0 e salto bisezione.")
    
    else:
        max_bisection_iters = 10

    a_M = (a_M_min + a_M_max) / 2.0
    
    best_delta_k = np.inf
    best_line = ""
    converged = False
    
    iter_count = 0
    
    while iter_count < max_bisection_iters:
        print(f"\n[BISEZIONE] iter={iter_count+1}/{max_bisection_iters}: m={m:.3f}, a_M={a_M*100:.2f}%, a_m={a_m*100:.0f}% ...")
        
        pm.execute_notebook(
            notebook_in,
            os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_M, is_first_run=is_first, iter=i)
        )

        is_first = False  

        df_results = parse_result_auto('result_auto.txt')
        idx = len(df_results) - 1  
        
        k_auto = df_results['k_eff'].iloc[idx] 
        std_k_auto = df_results['std_dev'].iloc[idx]
        T_water = df_results['T_water_K'].iloc[idx]
        T_fuel = df_results['T_fuel_K'].iloc[idx]
        sigma = df_results['compatibility'].iloc[idx]

        delta_k_abs = np.abs(k - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        # Tracking della migliore configurazione sub-critica
        if delta_k_abs < best_delta_k and k_auto < 1.0:
            best_delta_k = delta_k_abs
            best_line = current_line

        # Controllo convergenza
        if delta_k_abs <= 0.0015: 
            converged = True
            with open(output_file, "a") as f_out:
                f_out.write(best_line if best_line else current_line)
            break
            
        # Aggiornamento limiti bisezione
        if k_auto < k:
            a_M_min = a_M
        else:
            a_M_max = a_M
            
        a_M = (a_M_min + a_M_max) / 2.0
        iter_count += 1
        
    if not converged:
        print(f"-> ATTENZIONE: Bisezione non ha raggiunto la tolleranza per a_m = {a_m*100:.0f}%. Ultimo scarto = {delta_k_abs:.5f}")
        with open(output_file, "a") as f_out:
            f_out.write(best_line if best_line else current_line)

    a_m = round(a_m + 0.01, 2)

os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")