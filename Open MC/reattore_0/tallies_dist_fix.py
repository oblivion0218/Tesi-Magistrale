import openmc
import pyvista as pv
import numpy as np
import glob
import os

# 1. CARICAMENTO SMART (Trova l'ultimo file statepoint generato)
list_of_files = glob.glob('statepoint.*.h5')
if not list_of_files:
    raise FileNotFoundError("Nessun file statepoint trovato nella cartella!")
    
# Prende il file più recente in base all'ultima modifica
latest_file = max(list_of_files, key=os.path.getctime)
print(f"--- Lettura dati da: {latest_file} ---")

sp = openmc.StatePoint(latest_file)
tally = sp.get_tally(name='mappa_fissioni')

# Gestione Mesh
mesh = tally.find_filter(openmc.MeshFilter).mesh
dims = np.array(mesh.dimension)
origin = np.array(mesh.lower_left)
spacing = (np.array(mesh.upper_right) - origin) / dims

# Dati
data = tally.get_values(scores=['nu-fission']).reshape((dims[2], dims[1], dims[0])) # ZYX -> OpenMC standard
data = data.transpose(2, 1, 0) # ZYX -> XYZ per PyVista
values = data.flatten()

# Creazione Griglia
grid = pv.ImageData()
grid.dimensions = dims + 1
grid.origin = origin
grid.spacing = spacing
grid.cell_data["Fissioni"] = values

# --- PLOTTING ---
pl = pv.Plotter()

# SOGLIA: Togliamo solo lo zero assoluto. 
# Dato che useremo la scala logaritmica, anche valori piccoli sono importanti.
# Filtriamo valori piccolissimi (rumore numerico)
valid_mask = values > 1e-20 
grid_filtered = grid.threshold(1e-20) 

# Aggiungiamo il volume con SCALA LOGARITMICA
# Questo è il trucco per vedere sia il picco che la coda

valori_positivi = values[values > 0]
fiss_min = valori_positivi.min()
fiss_max = valori_positivi.max()

pl.add_mesh(grid_filtered, 
            scalars="Fissioni",
            cmap='plasma', 
            opacity=0.8,
            log_scale=True,
            clim=[fiss_min, fiss_max], # Imposta il range completo
            scalar_bar_args={
                'title': 'Fission Rate (Log Scale)',
                'n_labels': 5, # Numero di tacche sulla legenda
                'fmt': '%.1e'  # Formato scientifico per la legenda
            })

# Aggiungiamo il contorno del reattore per capire dove siamo
outline = grid.outline()
pl.add_mesh(outline, color="black")

pl.add_text("Distribuzione Fissioni (Fixed Source)", position='upper_left')
pl.view_xz() # Vista laterale per vedere il canale
pl.show()
sp.close()

### --- COSA STO VISUALIZZANDO NEL PLOT 3D? ---
# 1. NON sono singole particelle, ma una MESH (Griglia di Voxel).
# 2. OGNI CUBETTO: Il colore rappresenta la densità di fissioni integrata in quel volumetto.
# 3. PERCHÉ VEDO LE CELLE? Il comando .threshold() elimina i cubetti della mesh dove 
#    non avvengono fissioni (acqua, vuoto, cladding). Quello che resta è la 
#    "forma" del combustibile 'accesa' dalla sorgente.
# 4. IL GRADIENTE: Mostra la penetrazione della sorgente. Il colore sfuma man mano 
#    che i neutroni si allontanano dal punto di ingresso (0, 0, half_width - 1).