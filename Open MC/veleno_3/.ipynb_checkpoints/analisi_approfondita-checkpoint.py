import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import openmc.deplete
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False

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
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color='#2F4F4F')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='#2F4F4F')
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', alpha=0.6, color='#778899')
    ax.grid(which='minor', linestyle=':', alpha=0.3, color='#778899')
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=12, facecolor='#F0F8FF')

def salva_plot(fig, folder, filename):
    plt.tight_layout()
    fig.savefig(os.path.join(folder, filename), dpi=300)
    plt.close(fig)

def analizza_pitch(lista_pitch, giorni_on, path_result_final="result_final.txt"):
    spegnimento = sum(giorni_on)
    k_dict = load_k_ideale_dict(path_result_final)
    
    summary_path = "riepilogo_reattivita.txt"
    with open(summary_path, "w") as f_out:
        f_out.write("Pitch\tk_ideale\tk_shut\terr_shut\tk_max\terr_max\tdelta_rho_pcm\terr_delta_rho_pcm\n")
        
        for n_nnn in lista_pitch:
            pitch_float = float(n_nnn.replace('_', '.'))
            k_ideale = k_dict.get(pitch_float, 0.98942)
            
            folder = "depletion_pitch_{}".format(n_nnn)
            file_txt = os.path.join(folder, "k_eff_transitorio_pitch_{}.txt".format(n_nnn))
            
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
                idx_off = np.where(tempi > spegnimento)[0]
                
                if len(idx_on) == 0 or len(idx_off) == 0:
                    print("Skipping pitch {}: Dati temporali insufficienti.".format(n_nnn))
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
                ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                ax.plot(t_max, k_max, marker='*', color='red', markersize=12, linestyle='None', label='Peak ({:.1f} d)'.format(t_max))
                format_plot(ax, "Evolution of $k_{eff}$ - Pitch " + n_nnn, "Time [Days]", "Multiplication Factor $k_{eff}$")
                salva_plot(fig, folder, "k_eff_evolution_{}.png".format(n_nnn))
                
                # --- Plot Zoom K_eff ---
                fig, ax = plt.subplots(figsize=(16, 10))
                mask_zoom = (tempi >= spegnimento - 5) & (tempi <= spegnimento + 20)
                ax.errorbar(tempi[mask_zoom], k_eff[mask_zoom], yerr=k_err[mask_zoom], fmt='o-', color='#000000', ecolor='#789DD1', capsize=4, elinewidth=1.5, label='$k_{eff}$ (Zoom)')
                ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                ax.plot(t_max, k_max, marker='*', color='red', markersize=15, linestyle='None', label='Peak')
                format_plot(ax, "Zoom $k_{eff}$ Transitorio - Pitch " + n_nnn, "Time [Days]", "$k_{eff}$")
                salva_plot(fig, folder, "k_eff_zoom_{}.png".format(n_nnn))
                
            except Exception as e:
                print("Skipping pitch {}: Errore elaborazione dati k_eff ({})".format(n_nnn, e))
                continue

            # --- 2. Analisi Isotopi Totali (Somma Multimateriale Automatizzata) ---
            file_h5 = os.path.join(folder, "depletion_results.h5")
            if HAS_OPENMC:
                if os.path.exists(file_h5):
                    try:
                        res = openmc.deplete.Results(file_h5)
                        tempi_d = res.get_times("d")
                        
                        # Ottieni la lista completa di tutti i materiali presenti nel file h5
                        material_ids = list(res[0].volume.keys())
                        
                        # Inizializza gli array per sommare i contributi di ogni regione combustibile
                        xe_atoms = np.zeros_like(tempi_d)
                        sm_atoms = np.zeros_like(tempi_d)
                        
                        for mat_id in material_ids:
                            try:
                                _, xe_m = res.get_atoms(mat_id, "Xe135")
                                xe_atoms += xe_m
                            except Exception:
                                pass # Salta materiali che non generano/contengono Xenon (es. refrigerante o guaine)
                            try:
                                _, sm_m = res.get_atoms(mat_id, "Sm149")
                                sm_atoms += sm_m
                            except Exception:
                                pass # Salta materiali senza Samario
                        
                        # Xenon Completo
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, xe_atoms, 'o-', color='#000000', label='$^{135}$Xe (Total)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                        format_plot(ax, "Xenon-135 Evolution - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "xenon_evolution_{}.png".format(n_nnn))
                        
                        # Zoom Xenon
                        fig, ax = plt.subplots(figsize=(16, 10))
                        mask_z = (tempi_d >= spegnimento - 5) & (tempi_d <= spegnimento + 20)
                        ax.plot(tempi_d[mask_z], xe_atoms[mask_z], 'o-', color='#000000', label='$^{135}$Xe (Zoom)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5)
                        format_plot(ax, "Zoom $^{135}$Xe - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "xenon_zoom_{}.png".format(n_nnn))

                        # Samario Completo
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d, sm_atoms, 'o-', color='#000000', label='$^{149}$Sm (Total)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5, label='Shutdown')
                        format_plot(ax, "Samarium-149 Evolution - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "samarium_evolution_{}.png".format(n_nnn))
                        
                        # Zoom Samario
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.plot(tempi_d[mask_z], sm_atoms[mask_z], 'o-', color='#000000', label='$^{149}$Sm (Zoom)')
                        ax.axvline(x=spegnimento, color='red', linestyle='-.', alpha=0.5)
                        format_plot(ax, "Zoom $^{149}$Sm - Pitch " + n_nnn, "Time [Days]", "Total Atoms")
                        salva_plot(fig, folder, "samarium_zoom_{}.png".format(n_nnn))

                    except Exception as e:
                        print("Attenzione pitch {}: Errore estrazione isotopi da {} ({})".format(n_nnn, file_h5, e))
                else:
                    print("Attenzione pitch {}: File {} mancante, plots isotopi saltati.".format(n_nnn, file_h5))
            
            print("Analisi completata con successo per il pitch {}.".format(n_nnn))

# --- Parametri Esecuzione ---
lista_da_analizzare = [
    "1_000", "1_250", "1_500", "1_875", 
    "2_000", "2_250", "2_500", "2_750", 
    "3_000", "3_250", "3_500", "3_750" , "4_000" , "4_250"
]
giorni_on_run = [0.5, 0.5, 4.0, 5.0, 10.0, 30.0, 300.0, 50.0]

# Esecuzione (presuppone esecuzione da dentro 'veleno_3')
analizza_pitch(lista_da_analizzare, giorni_on_run, path_result_final="result_final.txt")