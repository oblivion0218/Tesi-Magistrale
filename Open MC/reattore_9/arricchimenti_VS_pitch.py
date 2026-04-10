import pandas as pd
import papermill as pm
import os
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="papermill")

# ==========================================
# FUNZIONI DI PARSING
# ==========================================

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

# ==========================================
# FASE 1: RICERCA ARRICCHIMENTO (REGULA FALSI)
# ==========================================

list_i = [1.0, 1.5, 2.0, 2.5, 0.5, 0.75, 0.375, 0.25]  
a_m = 0.05        # Arricchimento esterno FISSO al 5%
w = 0.15          # Percentuale acqua

notebook_auto = 'auto/auto.ipynb'
output_final = "result_final.txt"
output_scarti = "iterazioni_scartate.txt"

now = datetime.now()
timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")
header = f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. INT':<12} {'Arricch. EXT':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n"

# Inizializzazione file
for file_out in [output_final, output_scarti]:
    with open(file_out, "a") as f_out:
        f_out.write(timestamp)
        f_out.write(f"=== {'PARAMETRI CONVERGENTI / MIGLIORI' if file_out == output_final else 'ITERAZIONI SCARTATE'} ===\n")
        f_out.write(header)

df_kmax = parse_kmax_file('k_max.txt')
is_first = True

print(">>> INIZIO FASE 1: RICERCA ARRICCHIMENTO INTERNO <<<")

for m in list_i:
    # Trova l'indice riga esatto in k_max.txt associato al pitch 'm' per auto.ipynb
    row_idx_array = df_kmax.index[np.isclose(df_kmax['Pitch'], m, atol=1e-4)].tolist()
    if not row_idx_array:
        print(f"-> ATTENZIONE: Pitch {m} non trovato nel file k_max.txt. Salto.")
        continue
    
    iter_kmax = row_idx_array[0]
    k = df_kmax['k_max'].iloc[iter_kmax]
    std_k_max = df_kmax['sigma_k'].iloc[iter_kmax]
    p = df_kmax['Pressione_atm'].iloc[iter_kmax]
    
    a_1 = 0.10
    a_2 = 0.20

    k_vals = []
    a_vals = [a_1, a_2]
    converged = False
    
    # Inizializzazione tracker miglior riga
    best_delta_k = np.inf
    best_line = ""

    print(f"\n--- Analisi Pitch: {m} | k_max target: {k:.5f} ---")

    # --- FASE INIT CON CONTROLLO CONVERGENZA E SALVATAGGIO SCARTI ---
    for a_test in a_vals:
        print(f"[SECANTE INIT] Pitch={m:.3f}, a_int={a_test*100:.2f}% ...")
        pm.execute_notebook(
            notebook_auto, os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_test, is_first_run=is_first, iter=iter_kmax)
        )
        is_first = False
        
        df_res = parse_result_auto('result_auto.txt')
        idx = len(df_res) - 1
        k_auto = df_res['k_eff'].iloc[idx]
        std_k_auto = df_res['std_dev'].iloc[idx]
        T_water = df_res['T_water_K'].iloc[idx]
        T_fuel = df_res['T_fuel_K'].iloc[idx]
        sigma = df_res['compatibility'].iloc[idx]
        
        k_vals.append(k_auto)
        
        delta_k_abs = np.abs(k - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_test*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        # Aggiornamento miglior configurazione
        if delta_k_abs < best_delta_k:
            best_delta_k = delta_k_abs
            best_line = current_line

        if delta_k_abs <= 0.0005:
            converged = True
            with open(output_final, "a") as f_out:
                f_out.write(current_line)
            print(f"-> CONVERGENZA RAGGIUNTA IN FASE INIT per Pitch {m}: a_int = {a_test*100:.3f}%")
            break
        else:
            with open(output_scarti, "a") as f_out:
                f_out.write(current_line)
                
    if converged:
        continue
        
    k_1, k_2 = k_vals[0], k_vals[1]
    
    max_iters = 10  
    iter_count = 0

    # --- FASE SECANTE (REGULA FALSI) ---
    while iter_count < max_iters:
        
        if abs(k_2 - k_1) < 1e-5:
            print("-> ERRORE: Derivata nulla o troppo piccola. Impossibile procedere con secante.")
            break
            
        a_new = a_1 + (k - k_1) * (a_2 - a_1) / (k_2 - k_1)
        
        # Limiti fisici
        a_new = max(0.01, min(0.30, a_new))
        
        # Controllo stallo (se la variazione è < 0.05%)
        if abs(a_new - a_1) < 0.0005 or abs(a_new - a_2) < 0.0005:
            print(f"-> ATTENZIONE: Il metodo è in stallo su a_int={a_new*100:.2f}%. Limite o tolleranza raggiunti. Interrompo.")
            break
        
        print(f"[SECANTE] Pitch={m:.3f}, a_int={a_new*100:.2f}%, (iter={iter_count+1}/{max_iters})")
        
        pm.execute_notebook(
            notebook_auto, os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_new, is_first_run=False, iter=iter_kmax)
        )

        df_results = parse_result_auto('result_auto.txt')
        idx = len(df_results) - 1  
        
        k_auto = df_results['k_eff'].iloc[idx] 
        std_k_auto = df_results['std_dev'].iloc[idx]
        T_water = df_results['T_water_K'].iloc[idx]
        T_fuel = df_results['T_fuel_K'].iloc[idx]
        sigma = df_results['compatibility'].iloc[idx]

        delta_k_abs = np.abs(k - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_new*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        # Aggiornamento miglior configurazione
        if delta_k_abs < best_delta_k:
            best_delta_k = delta_k_abs
            best_line = current_line

        if delta_k_abs <= 0.0005:   
            converged = True
            with open(output_final, "a") as f_out:
                f_out.write(current_line)
            print(f"-> CONVERGENZA RAGGIUNTA per Pitch {m}: a_int = {a_new*100:.3f}%")
            break
        else:
            with open(output_scarti, "a") as f_out:
                f_out.write(current_line)
            
        # Logica Regula Falsi
        if k_auto < k:
            a_1, k_1 = a_new, k_auto
        else:
            a_2, k_2 = a_new, k_auto
            
        iter_count += 1
        
    if not converged:
        print(f"-> ATTENZIONE: Tolleranza non raggiunta o stallo per Pitch {m}.")
        print(f"-> SALVATAGGIO IN RESULT_FINAL: Scritto il best setup con scarto {best_delta_k:.5f}.")
        with open(output_final, "a") as f_out:
            f_out.write(best_line)
    
os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")

print("\n>>> TUTTE LE SIMULAZIONI COMPLETATE CON SUCCESSO <<<")