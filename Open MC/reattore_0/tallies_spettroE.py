import matplotlib.pyplot as plt
import numpy as np
import openmc
import glob
import os

# --- 1. CARICAMENTO SMART ---
list_of_files = glob.glob('statepoint.*.h5')
if not list_of_files:
    raise FileNotFoundError("Nessun file statepoint trovato!")
latest_file = max(list_of_files, key=os.path.getctime)
print(f"--- Lettura dati da: {latest_file} ---")

sp = openmc.StatePoint(latest_file)

# --- 2. ESTRAZIONE DATI ---
t_spec = sp.get_tally(name='spettro_energetico')
energy_filter = t_spec.find_filter(openmc.EnergyFilter)

# Qui sta il trucco: controlliamo la forma dei bins
bins = energy_filter.bins
flux = t_spec.get_values(scores=['flux']).flatten()

# Logica adattiva per i bin
if bins.ndim == 1:
    # Caso A: Array 1D di bordi continui [e0, e1, e2...] (N+1 elementi)
    lower_bounds = bins[:-1]
    upper_bounds = bins[1:]
else:
    # Caso B: Array 2D di coppie [[min, max], ...] (N righe, 2 colonne)
    lower_bounds = bins[:, 0]
    upper_bounds = bins[:, 1]

# Verifica di sicurezza dimensionale
if len(lower_bounds) != len(flux):
    # Se il flusso ha una dimensione in più (es. nuclide), sommiamo
    print(f"Attenzione: dimensioni mismatch. Flux: {flux.shape}, Bins: {lower_bounds.shape}")
    # Tentativo di fix se il flusso è (1, N)
    flux = flux.reshape(-1) 

# --- 3. CALCOLI FISICI ---
# Centro del bin (media geometrica)
energy_centers = np.sqrt(lower_bounds * upper_bounds)
# Larghezza del bin
energy_width = upper_bounds - lower_bounds

# Flusso per unità di letargia: phi(u) = E * phi(E)
unit_lethargy_flux = flux * energy_centers / energy_width

# --- 4. PLOT ---
plt.figure(figsize=(10, 6))

plt.semilogx(energy_centers, unit_lethargy_flux, color='royalblue', lw=1.5, label='Spettro')

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.xlabel('Energia [eV]')
# Usiamo r'' (raw string) per evitare il SyntaxWarning sul LaTeX
plt.ylabel(r'Flusso per unità di Letargia ($E \cdot \phi(E)$)') 
plt.title(f'Spettro Energetico Neutroni\n({latest_file})')

plt.axvline(x=0.025, color='r', linestyle='--', alpha=0.7, label='Termico (0.025 eV)')
plt.axvline(x=1e6, color='orange', linestyle='--', alpha=0.7, label='Sorgente (1 MeV)')

plt.legend()
plt.show()

sp.close()


### --- COSA STO VISUALIZZANDO DAVVERO? (Tally: spettro_energetico) ---
#
# 1. DOVE E COSA: 
#    È una misura GLOBALE. Non avendo filtri spaziali, integra il flusso su tutto il volume 
#    (Fuel + Acqua + Cladding + Riflettore). Rappresenta la popolazione media "residente".
#
# 2. LOGICA STATISTICA (Track Length):
#    OpenMC non conta i neutroni come "individui", ma somma la lunghezza dei loro percorsi 
#    (cm) in ogni intervallo di energia. Lo stesso neutrone viene registrato PIÙ VOLTE: 
#    ogni volta che urta e cambia energia, inizia a contribuire al "cestino" (bin) successivo.
#
# 3. SIGNIFICATO DELLA LETARGIA (u):
#    La letargia definisce quanto un neutrone è "rallentato" rispetto alla sorgente.
#    Formula: u = ln(E_max / E). 
#    Nel plot usiamo E * phi(E) (che è d_phi/d_u) per due motivi:
#    - NORMALIZZAZIONE: Dà lo stesso peso visivo a ogni decade (es. 1-10 eV pesano come 1-10 MeV).
#    - FISICA: In un moderatore ideale, questa curva è piatta (regione 1/E). Se vedi picchi 
#      o buchi, significa che in quel punto i neutroni "ristagnano" o vengono mangiati.
#
# 4. ZONE CHIAVE NEL PLOT:
#    - PICCO @ 2 MeV: Neutroni "giovani" appena sparati dalla sorgente.
#    - PLATEAU CENTRALE: Neutroni che stanno perdendo energia nell'idrogeno (rallentamento).
#    - GOBBA TERMICA @ 0.025 eV: Neutroni "vecchi" in equilibrio termico con l'acqua.
