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


now = datetime.now()
output_file = "result_final.txt"  
timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")

# Header Result Final
with open(output_file, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI DI SIMULAZIONE ===\n")
    f_out.write(f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. MAX':<12} {'Arricch.min':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n")

df_kmax = parse_kmax_file('k_max.txt')
notebook_in = 'auto/auto.ipynb'

is_first = True

for i, k in enumerate(df_kmax['k_max']):
    std_k_max = df_kmax['sigma_k'].iloc[i]
    m = df_kmax['Pitch'].iloc[i] 
    p = df_kmax['Pressione_atm'].iloc[i]
    w = 0.15    # percentuale acqua fissa
    a_m = 0.01  # arricchimento esterno fisso
    
    # --- 1. TEST PRELIMINARE (MAX ARRICCHIMENTO INTERNO) ---
    a_M_test = 0.20
    print(f"\n[TEST] Iter {i} - p={p:.0f}, m={m:.3f} | a_M={a_M_test*100:.0f}%, a_m={a_m*100:.0f}% ...")
    
    pm.execute_notebook(
        notebook_in,
        os.devnull,
        parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_M_test, is_first_run=is_first, iter=i)
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
    current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M_test*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"

    # Controllo fattibilità
    if k_auto < k:
        print(f"-> FALLIMENTO: k_auto ({k_auto:.5f}) < k_max ({k:.5f}). Nessuna convergenza possibile.")
        with open(output_file, "a") as f_out:
            f_out.write(f"{p:<12.0f} {m:<15.3f} --- CONFIGURAZIONE NON COMPATIBILE (k_auto < k_max con a_M=20%, a_m=1%) ---\n")
        continue
    elif delta_k_abs <= 0.0015:
        # Se siamo già entro la tolleranza al primo colpo
        print(f"-> CONVERGENZA IMMEDIATA. Scarto = {delta_k_abs:.5f}")
        with open(output_file, "a") as f_out:
            f_out.write(current_line)
        continue
    else : 
        print ("Delta k : ", delta_k_abs, "\n")


    # --- 2. ALGORITMO DI BISEZIONE ---
    a_M_min = 0.01
    a_M_max = 0.20
    
    best_delta_k = np.inf
    best_line = ""
    converged = False
    max_bisection_iters = 20
    
    for iter_count in range(max_bisection_iters):
        a_M = (a_M_min + a_M_max) / 2.0
        
        print(f"[BISEZIONE] iter={iter_count+1}/{max_bisection_iters}: a_M={a_M*100:.2f}% ...")
        
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
        sigma = df_results['compatibility'].iloc[idx]

        delta_k_abs = np.abs(k - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        # Tracking della migliore configurazione sub-critica (o in generale con il minor delta)
        if delta_k_abs < best_delta_k:
            best_delta_k = delta_k_abs
            best_line = current_line

        # Controllo convergenza
        if delta_k_abs <= 0.0015: 
            converged = True
            break
            
        # Aggiornamento limiti bisezione
        if k_auto < k:
            a_M_min = a_M
        else:
            a_M_max = a_M
            
    if not converged:
        print(f"-> ATTENZIONE: Tolleranza non raggiunta per iter {i}. Ultimo scarto = {delta_k_abs:.5f}")
        
    # Scrittura del risultato migliore trovato
    with open(output_file, "a") as f_out:
        f_out.write(best_line)

os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")









''' IMPLEMENTAZIONE A SIGMA 

# Inizializzazione globale dello stato per WARM START
a_M_opt = 0.01  # primo anello
a_m_opt = 0.01  # ultimo anello

for i, k in enumerate(df_kmax['k_max']):

    std_k_max = df_kmax['sigma_k'].iloc[i]
    m = df_kmax['Pitch'].iloc[i] 
    p = df_kmax['Pressione_atm'].iloc[i]
    w = 0.15    # percentuale acqua
    
    # 1. WARM START: Partenza dalla soluzione ottimale precedente (con back-off)
    a_M = max(0.01, round(a_M_opt - 0.02, 2))
    a_m = max(0.01, round(a_m_opt , 2)) if a_M_opt >= 0.20 else 0.01
    
    notebook_in = 'auto/auto.ipynb'
    
    best_sigma = np.inf
    best_line = ""
    is_compatible = False

    while not is_compatible and a_m <= 0.20:
    
        if a_m == 0.01 and is_first == True :
            a_M = 0.1

        print(f"\nAvvio simulazione: moltiplicatore={m:.3f}, pressione={p:.0f}, perc_water={w*100:.0f}%, arric. MAX={a_M*100:.0f}%, arric. MIN={a_m*100:.0f}% ...")
        pm.execute_notebook(
            notebook_in,
            os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_M, is_first_run=is_first, iter = i)
        )

        df_results = parse_result_auto('result_auto.txt')
        idx = len(df_results) - 1  
        
        k_auto = df_results['k_eff'].iloc[idx] 
        std_k_auto = df_results['std_dev'].iloc[idx]
        T_water = df_results['T_water_K'].iloc[idx]
        T_fuel = df_results['T_fuel_K'].iloc[idx]

        # Calcolo della deviazione statistica
        sigma = np.abs(k_auto - k) / np.sqrt(std_k_auto**2 + std_k_max**2) 
        
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_M:<12.2f} {a_m:<12.2f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"

        # Tracking configurazione
        if sigma < best_sigma and k_auto < 1:
            best_sigma = sigma
            best_line = current_line

        # 2. Uscita Rapida: Target entro intervallo fiduciario
        if sigma < 10: 
            with open(output_file, "a") as f_out:
                f_out.write(best_line)
            is_compatible = True
            i+=1

            
            # Salvataggio stato ottimale per il prossimo step (Warm Start)
            a_M_opt = a_M
            a_m_opt = a_m
            break
            
        # 3. Cut-off: Gestione overshoot rispetto a k_max
        elif k_auto > k:
            a_m = round(a_m + 0.01, 2)
            a_M = 0.01  

        # 4. Esplorazione Adattiva (Pseudo-Gradiente)
        else:
            delta_k = k - k_auto
            
            # Modulazione dinamica del passo in base al residuo differenziale
            if delta_k >= 0.1:
                step = 0.04
            elif delta_k < 0.1:
                step = 0.01

            if a_M < 0.20:
                a_M = min(0.20, round(a_M + step, 2))
            else:
                a_m = min(0.20, round(a_m + step, 2))
                a_M = 0.01

        is_first = False 

    # 5. Uscita limite per saturazione spazio di ricerca
    if not is_compatible and best_line:
        with open(output_file, "a") as f_out:
            f_out.write(best_line)
        i += 1

os.system("rm *.h5")
os.system("rm *.xml")
os.system("rm *.out")
os.system("rm *.png")

'''