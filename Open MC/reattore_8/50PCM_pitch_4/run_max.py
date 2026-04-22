import pandas as pd
import papermill as pm
import os
import numpy as np
from datetime import datetime
from scipy.interpolate import PchipInterpolator

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

def get_last_a_int(filename, default=0.5):
    """Estrae l'ultimo valore di Arricch. INT dal file dei risultati."""
    if not os.path.exists(filename):
        return default
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith(('=', '-', 'Pres')):
                cols = line.split()
                if len(cols) > 4:
                    try:
                        return float(cols[3]) / 100.0  # Converte la percentuale in frazione
                    except ValueError:
                        continue
    return default

is_first = False
i = 25 # elemento pitch = 4

now = datetime.now()
output_file = "result_final.txt"  

# Lettura del valore iniziale dal file storico (prima di appendere nuovi header)
last_a_int = get_last_a_int(output_file, default=0.5)

timestamp = now.strftime("\n\n------ SIMULAZIONE DEL %d/%m/%Y ALLE ORE %H:%M ------\n")

# Header Result Final
with open(output_file, "a") as f_out:
    f_out.write(timestamp)
    f_out.write("=== PARAMETRI DI SIMULAZIONE ===\n")
    f_out.write(f"{'Pres [Atm]':<12} {'moltiplicatore':<15} {'PERC_water':<12} {'Arricch. INT':<12} {'Arricch. EXT':<12}{'T_WATER [K]':<12} {'T_FUEL [K]':<12} {'K_max':<12} {'std_k_max':<12} {'K_auto':<12} {'std_k_auto':<12} {'Compatibilità':<12}\n")

df_kmax = parse_kmax_file('k_max.txt')

k_target = df_kmax['k_max'].iloc[i]
std_k_max = df_kmax['sigma_k'].iloc[i]
m = df_kmax['Pitch'].iloc[i] 
p = df_kmax['Pressione_atm'].iloc[i]
w = 0.15    # percentuale acqua

notebook_in = 'auto/auto.ipynb'

a_m = 0.35        # a_m rappresenta l'arricchimento più esterno
ARR_MAX = 0.40

while a_m <= ARR_MAX:
    max_iters = 20
    
    # Inizializzazione utilizzando l'ultimo valore noto
    a_vals = [last_a_int, 0.00]
    k_vals_dict = {}
    
    best_delta_k = np.inf
    best_line = ""
    converged = False
    iter_count = 0
    
    a_queue = list(a_vals)
    
    while iter_count < max_iters and a_queue:
        a_test = a_queue.pop(0)
        
        print(f"\n[SURROGATO] iter={iter_count+1}/{max_iters}: m={m:.3f}, a_int={a_test*100:.2f}%, a_ext={a_m*100:.0f}% ...")
        
        pm.execute_notebook(
            notebook_in,
            os.devnull,
            parameters=dict(moltiplicatore=m, pressione=p, water_perc=w, enrich_min=a_m, enrich_max=a_test, is_first_run=is_first, iter=i)
        )

        is_first = False  

        df_results = parse_result_auto('result_auto.txt')
        idx = len(df_results) - 1  
        
        k_auto = df_results['k_eff'].iloc[idx] 
        std_k_auto = df_results['std_dev'].iloc[idx]
        T_water = df_results['T_water_K'].iloc[idx]
        T_fuel = df_results['T_fuel_K'].iloc[idx]
        sigma = df_results['compatibility'].iloc[idx]
        
        # Salvataggio risultato
        k_vals_dict[a_test] = k_auto

        delta_k_abs = np.abs(k_target - k_auto)
        current_line = f"{p:<12.0f} {m:<15.3f} {w:<12.2f} {a_test*100:<12.3f} {a_m*100:<12.0f} {T_water:<12.0f} {T_fuel:<12.0f} {k_target:<12.5f} {std_k_max:<12.5f} {k_auto:<12.5f} {std_k_auto:<12.5f} {sigma:<12.5f}\n"
        
        # Tracking della migliore configurazione sub-critica
        if delta_k_abs < best_delta_k and k_auto < 1.0:
            best_delta_k = delta_k_abs
            best_line = current_line

        # Controllo convergenza
        if delta_k_abs <= 0.0005:   # 50 PCM di tolleranza
            converged = True
            with open(output_file, "a") as f_out:
                f_out.write(current_line)
            print(f"-> CONVERGENZA RAGGIUNTA: a_int = {a_test*100:.3f}% con scarto {delta_k_abs:.5f}")
            break
            
        # --- CONTROLLO FATTIBILITÀ (Dopo aver calcolato i due punti iniziali) ---
        if len(k_vals_dict) == 2 and not converged:
            k_100 = k_vals_dict.get(last_a_int) # Adattato per supportare il limite dinamico
            k_0 = k_vals_dict.get(0.0)
            
            if k_100 is not None and k_0 is not None:
                if k_target > k_100 or k_target < k_0:
                    print(f"-> ERRORE FATTIBILITÀ: k_target ({k_target:.5f}) non è compreso tra k_0 ({k_0:.5f}) e il limite superiore ({k_100:.5f}).")
                    print(f"-> Impossibile raggiungere la convergenza per a_ext = {a_m*100:.0f}%. Salto iterazione.")
                    break
        
        iter_count += 1
        
        # --- AGGIORNAMENTO DEL MODELLO SURROGATO ---
        if not a_queue and not converged:
            k_arr = np.array(list(k_vals_dict.values()))
            a_arr = np.array(list(k_vals_dict.keys()))
            
            sort_idx = np.argsort(k_arr)
            k_sorted = k_arr[sort_idx]
            a_sorted = a_arr[sort_idx]
            
            try:
                # Interpolazione dell'inversa: a(k)
                interp_func = PchipInterpolator(k_sorted, a_sorted)
                a_new = float(interp_func(k_target))
                
                # Vincolo fisico (arricchimento tra 0 e 1)
                a_new = max(0.00, min(1.00, a_new))
                
                # Prevenzione di stallo numerico
                if any(abs(a_new - a_old) < 0.000005 for a_old in a_arr):
                    print(f"-> STALLO NUMERICO: Convergenza ai limiti del fit su a_int={a_new*100:.2f}%.")
                    break
                    
                a_queue.append(a_new)
                print(f"[SURROGATO] Interpolazione su {len(k_sorted)} punti -> Suggerisce a_int={a_new*100:.2f}%")
                
            except ValueError:
                print("-> ERRORE: Dati non strettamente monotoni. Impossibile interpolare l'inversa.")
                break
        
    if not converged and iter_count > 0:
        print(f"-> ATTENZIONE: Tolleranza non raggiunta o limiti violati per a_ext = {a_m*100:.0f}%. Ultimo scarto = {best_delta_k:.5f}")
        with open(output_file, "a") as f_out:
            f_out.write(best_line if best_line else current_line)

    # Aggiorna il valore per l'iterazione di a_m successiva
    if converged:
        last_a_int = a_test
    elif best_line:
        last_a_int = float(best_line.split()[3]) / 100.0

    a_m = round(a_m + 0.03, 2)

os.system("rm *.h5 *.xml *.out *.png 2>/dev/null")
