import numpy as np
import matplotlib.pyplot as plt
import os
import glob

try:
    import openmc
    import openmc.deplete
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False

# =====================================================================
# COSTANTI FISICHE
# =====================================================================
S_rate = 2.02e17 * 5.87905e-03   
eV_to_Joule = 1.60218e-19
E_fission_approx_J = 200e6 * eV_to_Joule 
nu_medio_approx = 2.44 

def load_k_ideale_dict(filepath):
    k_dict = {}
    if not os.path.exists(filepath):
        print("Attenzione: File {} non trovato. Verrà usato un valore di default.".format(filepath))
        return k_dict
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('-') or line.startswith('='): 
                continue
            parts = line.split()
            if len(parts) >= 11:
                try:
                    pitch = float(parts[1])
                    k_auto = float(parts[9])
                    k_dict[pitch] = k_auto
                except ValueError:
                    pass
    return k_dict

def format_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', color='#2F4F4F')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold', color='#2F4F4F')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=20, labelcolor='#2F4F4F')
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.grid(which='major', linestyle='--', alpha=0.6, color='#778899')
    ax.grid(which='minor', linestyle=':', alpha=0.3, color='#778899')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=20, facecolor='#F0F8FF')

def salva_plot(fig, folder, filename):
    plt.tight_layout()
    fig.savefig(os.path.join(folder, filename), dpi=300)
    plt.close(fig)

def analizza_pitch(lista_pitch, path_result_final="result_final.txt"):
    spegnimento = 400
    limite_picco = 410  
    k_dict = load_k_ideale_dict(path_result_final)
    
    summary_path = "riepilogo_reattivita.txt"
    with open(summary_path, "w") as f_out:
        f_out.write("Pitch\tk_ideale\tk_shut\terr_shut\tk_max\terr_max\tdelta_rho_pcm\terr_delta_rho_pcm\n")
        
        for n_nnn in lista_pitch:
            pitch_float = float(n_nnn.replace('_', '.'))
            k_ideale = k_dict.get(pitch_float, 100)
            
            folder = "depletion_pitch_{}".format(n_nnn)
            file_txt = os.path.join(folder, "ALTA_STAT_k_eff_transitorio_pitch_{}.txt".format(n_nnn))
            
            if not os.path.exists(file_txt):
                print("Skipping pitch {}: File testuale mancante ({})".format(n_nnn, file_txt))
                continue
                
            # --- 1. Analisi K_eff e Reattività ---
            try:
                dati = np.loadtxt(file_txt, skiprows=1, usecols=(1, 2, 3))
                if dati.ndim == 1:
                    dati = dati.reshape(1, -1)
                
                sort_idx = np.argsort(dati[:, 0])
                dati = dati[sort_idx]
                
                tempi = dati[:, 0]
                k_eff = dati[:, 1]
                k_err = dati[:, 2]
                
                idx_on = np.where(tempi <= spegnimento)[0]
                idx_off = np.where((tempi > spegnimento) & (tempi < limite_picco))[0]
                
                if len(idx_on) == 0 or len(idx_off) == 0:
                    print("Skipping pitch {}: Dati temporali insufficienti nell'intervallo richiesto.".format(n_nnn))
                    continue
                    
                idx_shut = idx_on[-1]
                k_shut = k_eff[idx_shut]
                err_shut = k_err[idx_shut]
                
                idx_max_rel = np.argmax(k_eff[idx_off])
                idx_max = idx_off[idx_max_rel]
                t_max = tempi[idx_max]
                k_max = k_eff[idx_max]
                err_max = k_err[idx_max]
                
                delta = k_max - k_shut
                delta_rho_pcm = (delta / (k_ideale * (k_ideale + delta))) * 1e5
                err_delta_rho_pcm = 1e5 * np.sqrt((err_shut / k_shut**2)**2 + (err_max / k_max**2)**2)
                
                f_out.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.2f}\t{:.2f}\n".format(
                    n_nnn, k_ideale, k_shut, err_shut, k_max, err_max, delta_rho_pcm, err_delta_rho_pcm))
                
                # --- Plot K_eff Completo ---
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.errorbar(tempi, k_eff, yerr=k_err, fmt='o-', color='#000000', ecolor='#789DD1', capsize=4, elinewidth=1.5, label='$k_{eff}$')
                ax.axvline(x=spegnimento, color='red', linestyle='-.', label='Shutdown')
                ax.plot(t_max, k_max, marker='*', color='red', markersize=16, linestyle='None', label='Peak ({:.1f} d)'.format(t_max))
                format_plot(ax, "Evolution of $k_{eff}$ - Pitch " + n_nnn, "Time [Days]", "Multiplication Factor $k_{eff}$")
                salva_plot(fig, folder, "k_eff_evolution_{}.png".format(n_nnn))
                
            # --- Plot Zoom K_eff ---

                # --- Rimozione del punto a 400.05 giorni ---
                # Crea una maschera che esclude il valore specifico
                mask_clean = ~np.isclose(tempi, 400.05, atol=1e-3)
                
                # Applica la maschera a tutti i vettori associati
                tempi = tempi[mask_clean]
                k_eff = k_eff[mask_clean]
                k_err = k_err[mask_clean]

                # --- Plot Zoom K_eff ---
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.errorbar(tempi, k_eff, yerr=k_err, fmt='o-', color='#000000', ecolor='#789DD1', capsize=4, elinewidth=1.5, linewidth=1.5, label='$k_{eff}$ (Zoom)')
                ax.axvline(x=spegnimento, color='red', linestyle='-.', linewidth=4,  label='Shutdown')
                
                # Trova il valore di k_eff esattamente allo spegnimento per la linea orizzontale
                idx_spegnimento = np.argmin(np.abs(tempi - spegnimento))
                k_spegnimento = k_eff[idx_spegnimento]
                
                # Linee orizzontali verdi per i due stati (spegnimento e massimo)
                ax.axhline(y=k_spegnimento, color='green', linestyle='--', linewidth=4, alpha=0.8, label='$k_{eff}$ Shutdown')
                ax.axhline(y=k_max, color='green', linestyle='-', linewidth=4, alpha=0.8, label='$k_{max}$ Peak')
                
                if t_max <= spegnimento + 10: 
                    ax.plot(t_max, k_max, marker='*', color='red', markersize=16, linestyle='None', label='Peak')
                    
                    # Doppia freccia verticale tra i due livelli di k_eff
                    x_arrow = (spegnimento + t_max) / 2 if t_max != spegnimento else spegnimento + 2
                    ax.annotate('', xy=(x_arrow, k_spegnimento), xytext=(x_arrow, k_max),
                                arrowprops=dict(arrowstyle='<->', color='green', lw=4, mutation_scale=20))
                
                ax.set_xlim(spegnimento - 5, spegnimento + 10)
                mask_zoom = (tempi >= spegnimento - 5) & (tempi <= spegnimento + 10)
                if np.any(mask_zoom):
                    y_min = np.min(k_eff[mask_zoom] - k_err[mask_zoom])
                    y_max = np.max(k_eff[mask_zoom] + k_err[mask_zoom])
                    pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.001
                    ax.set_ylim(y_min - pad, y_max + pad)
                
                format_plot(ax, "Zoom $k_{eff}$ - Pitch " + n_nnn, "Time [Days]", "$k_{eff}$")
                salva_plot(fig, folder, "k_eff_zoom_{}.png".format(n_nnn))


            except Exception as e:
                print("Skipping pitch {}: Errore elaborazione dati k_eff ({})".format(n_nnn, e))
                continue

            # --- 2. Analisi Isotopi e Potenza ---
            file_h5 = os.path.join(folder, "depletion_results.h5")
            if HAS_OPENMC:
                # --- Estrazione Potenza da StatePoint ---
                sp_files = glob.glob(os.path.join(folder, 'openmc_simulation_n*.h5'))
                sp_files = sorted(sp_files, key=lambda x: int(os.path.basename(x).split('_n')[1].split('.')[0]))
                
                pot_nu, pot_nu_err = [], []
                
                if sp_files:
                    # Correggiamo l'indicizzazione iterando in parallelo (fermandosi al più corto)
                    for sp_file, t_val in zip(sp_files, tempi):
                        with openmc.StatePoint(sp_file) as sp:
                            t_glob = sp.get_tally(name='fission_tot')
                            slice_n = t_glob.get_slice(scores=['nu-fission'])
                            n_val = slice_n.mean.flatten()[0]
                            n_std = slice_n.std_dev.flatten()[0]
                            
                            curr_S = S_rate if t_val <= spegnimento else 0.0
                            
                            pot_w = (n_val / nu_medio_approx) * curr_S * E_fission_approx_J
                            pot_err = (n_std / nu_medio_approx) * curr_S * E_fission_approx_J
                            
                            pot_nu.append(pot_w)
                            pot_nu_err.append(pot_err)
                            
                    pot_nu = np.array(pot_nu)
                    pot_nu_err = np.array(pot_nu_err)

                    # Plot Potenza Completo
                    fig, ax = plt.subplots(figsize=(16, 10))
                    ax.errorbar(tempi[:len(pot_nu)], pot_nu, yerr=pot_nu_err, fmt='o-', color='#000000', ecolor='#789DD1', capsize=4, elinewidth=1.5, label='Power (nu-fission)')
                    ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                    format_plot(ax, "Power Evolution - Pitch " + n_nnn, "Time [Days]", "Power [W]")
                    salva_plot(fig, folder, "power_evolution_{}.png".format(n_nnn))
                else:
                    print("Nessun StatePoint trovato per il pitch {}, plot potenza saltato.".format(n_nnn))

                # --- Estrazione Isotopi da Depletion Results ---
                if os.path.exists(file_h5):
                    try:
                        res = openmc.deplete.Results(file_h5)
                        tempi_d = res.get_times("d")
                        
                        material_ids = list(res[0].volume.keys())
                        
                        xe_atoms = np.zeros_like(tempi_d)
                        i_atoms = np.zeros_like(tempi_d)
                        sm_atoms = np.zeros_like(tempi_d)
                        pm_atoms = np.zeros_like(tempi_d)
                        
                        for mat_id in material_ids:
                            try:
                                _, xe_m = res.get_atoms(mat_id, "Xe135")
                                xe_atoms += xe_m
                            except Exception:
                                pass
                            try:
                                _, i_m = res.get_atoms(mat_id, "I135")
                                i_atoms += i_m
                            except Exception:
                                pass
                                
                            try:
                                _, sm_m = res.get_atoms(mat_id, "Sm149")
                                sm_atoms += sm_m
                            except Exception:
                                pass
                            try:
                                _, pm_m = res.get_atoms(mat_id, "Pm149")
                                pm_atoms += pm_m
                            except Exception:
                                pass
                        
                        # Xenon e Iodio Completo
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, xe_atoms, 'o-', color='#000000', label='$^{135}$Xe (Total)')
                        ax.plot(tempi_d, i_atoms, 's--', color='#789DD1', label='$^{135}$I (Precursor)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                        format_plot(ax, "Xenon-135 and Iodine-135 Evolution - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "xenon_iodine_evolution_{}.png".format(n_nnn))
                        
                        # Zoom Xenon e Iodio
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, xe_atoms, 'o-', color='#000000', label='$^{135}$Xe (Zoom)')
                        ax.plot(tempi_d, i_atoms, 's--', color='#789DD1', label='$^{135}$I (Zoom)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5)
                        
                        ax.set_xlim(spegnimento - 5, spegnimento + 10)
                        mask_z = (tempi_d >= spegnimento - 5) & (tempi_d <= spegnimento + 10)
                        if np.any(mask_z):
                            y_min = min(np.min(xe_atoms[mask_z]), np.min(i_atoms[mask_z]))
                            y_max = max(np.max(xe_atoms[mask_z]), np.max(i_atoms[mask_z]))
                            pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.001
                            ax.set_ylim(y_min - pad, y_max + pad)
                            
                        format_plot(ax, "Zoom $^{135}$Xe e $^{135}$I - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "xenon_iodine_zoom_{}.png".format(n_nnn))

                        # Samario e Promezio Completo
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, sm_atoms, 'o-', color='#000000', label='$^{149}$Sm (Total)')
                        ax.plot(tempi_d, pm_atoms, 's--', color='#789DD1', label='$^{149}$Pm (Precursor)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                        format_plot(ax, "Samarium-149 and Promethium-149 Evolution - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "samarium_promethium_evolution_{}.png".format(n_nnn))
                        
                        # Zoom Samario e Promezio
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, sm_atoms, 'o-', color='#000000', label='$^{149}$Sm (Zoom)')
                        ax.plot(tempi_d, pm_atoms, 's--', color='#789DD1', label='$^{149}$Pm (Zoom)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5)
                        
                        ax.set_xlim(spegnimento - 5, spegnimento + 20)
                        if np.any(mask_z):
                            y_min = min(np.min(sm_atoms[mask_z]), np.min(pm_atoms[mask_z]))
                            y_max = max(np.max(sm_atoms[mask_z]), np.max(pm_atoms[mask_z]))
                            pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.001
                            ax.set_ylim(y_min - pad, y_max + pad)
                            
                        format_plot(ax, "Zoom $^{149}$Sm e $^{149}$Pm - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "samarium_promethium_zoom_{}.png".format(n_nnn))

                    except Exception as e:
                        print("Attenzione pitch {}: Errore estrazione isotopi da {} ({})".format(n_nnn, file_h5, e))
                else:
                    print("Attenzione pitch {}: File {} mancante, plots isotopi saltati.".format(n_nnn, file_h5))
            
            print("Analisi completata con successo per il pitch {}.".format(n_nnn))

# --- Parametri Esecuzione ---
lista_da_analizzare = [
  # "1_000", "1_250",  "1_500", "1_875",  "2_000", "2_250","2_500","2_750","3_000","3_250", "3_500", "3_750", "4_000", "4_250"
  #  "0_750" , "0_500"
   "1_000"	
]

# Esecuzione
analizza_pitch(lista_da_analizzare, path_result_final="result_final.txt")
