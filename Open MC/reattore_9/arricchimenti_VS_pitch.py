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

from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore", category=UserWarning, module="papermill")

# ==========================================
# FASE 1: RICERCA ARRICCHIMENTO (SURROGATE MODELING - PCHIP)
# ==========================================

list_i = [2.5]  
a_m = 0.05       
w = 0.15          

notebook_auto = 'auto/auto.ipynb'
output_final = "result_final.txt"

now = datetime.now()
timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")
header = f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. INT':<12} {'Arricch. EXT':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n"

with open(output_final, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI CONVERGENTI / MIGLIORI ===\n")
    f_out.write(header)

df_kmax = parse_kmax_file('k_max.txt')
is_first = True

print(">>> INIZIO FASE 1: RICERCA ARRICCHIMENTO INTERNO <<<")

for m in list_i:
    row_idx_array = df_kmax.index[np.isclose(df_kmax['Pitch'], m, atol=1e-4)].tolist()
    if not row_idx_array:
        print(f"-> ATTENZIONE: Pitch {m} non trovato nel file k_max.txt. Salto.")
        continue
    
    iter_kmax = row_idx_array[0]
    k_target = df_kmax['k_max'].iloc[iter_kmax]
    std_k_max = df_kmax['sigma_k'].iloc[iter_kmax]
    p = df_kmax['Pressione_atm'].iloc[iter_kmax]
    
    # Punti di partenza per inizializzare il modello
    a_vals = [0.0638, 0.0630]
    k_vals_dict = {}
    converged = False
    best_delta_k = np.inf
    best_line = ""

    print(f"\n--- Analisi Pitch: {m} | k_max target: {k_target:.5f} ---")

    # --- FASE DI CAMPIONAMENTO ---
    iter_count = 0
    max_iters = 10
    
    # Costruiamo la lista dinamica dei punti da testare (parte con i due iniziali)
    a_queue = list(a_vals)
    
    while iter_count < max_iters and a_queue:
        a_test = a_queue.pop(0)
        
        print(f"[TEST] Pitch={m:.3f} | Valuto a_int={a_test*100:.2f}% (iter {iter_count})")
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
        
        # Salva risultato nel dizionario della storia
        k_vals_dict[a_test] = k_auto
        
        delta_k_abs = np.abs(k_target - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_test*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k_target:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        if delta_k_abs < best_delta_k:
            best_delta_k = delta_k_abs
            best_line = current_line

        # Condizione di convergenza
        if delta_k_abs <= 0.0005:
            converged = True
            with open(output_final, "a") as f_out:
                f_out.write(current_line)
            print(f"-> CONVERGENZA RAGGIUNTA: a_int = {a_test*100:.3f}% con scarto {delta_k_abs:.5f}")
            break
            
        iter_count += 1
        
        # --- AGGIORNAMENTO DEL MODELLO SURROGATO ---
        # Se la coda è vuota e non abbiamo convergenza, generiamo il prossimo punto
        if not a_queue and not converged:
            k_arr = np.array(list(k_vals_dict.values()))
            a_arr = np.array(list(k_vals_dict.keys()))
            
            # Ordiniamo i dati per permettere l'interpolazione
            sort_idx = np.argsort(k_arr)
            k_sorted = k_arr[sort_idx]
            a_sorted = a_arr[sort_idx]
            
            # Se abbiamo solo 2 punti, la PCHIP fa essenzialmente una retta (Secante equivalente)
            # Con >= 3 punti, costruisce la spline monotona
            try:
                # Interpolazione dell'inversa: a(k)
                interp_func = PchipInterpolator(k_sorted, a_sorted)
                a_new = float(interp_func(k_target))
                
                # Vincolo fisico sull'arricchimento
                a_new = max(0.00, min(1.00, a_new))
                
                # Prevenzione di stallo: se l'algoritmo ripropone un punto già calcolato (tolleranza numerica)
                if any(abs(a_new - a_old) < 0.00025 for a_old in a_arr):
                    print(f"-> STALLO NUMERICO: Convergenza ai limiti del fit su a_int={a_new*100:.2f}%.")
                    break
                    
                a_queue.append(a_new)
                print(f"[SURROGATO] Interpolazione su {len(k_sorted)} punti -> Suggerisce a_int={a_new*100:.2f}%")
                
            except ValueError:
                print("-> ERRORE: Dati non strettamente monotoni. Impossibile interpolare l'inversa.")
                break

    if not converged:
        print(f"-> Tolleranza non raggiunta per Pitch {m} entro max_iters. Salvato miglior setup (scarto: {best_delta_k:.5f}).")
        with open(output_final, "a") as f_out:
            f_out.write(best_line)
            
os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")
print("\n>>> FASE 1 COMPLETATA <<<")