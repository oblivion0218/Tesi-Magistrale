import openmc
import pyvista as pv
import numpy as np
import glob
import os

# 1. CARICAMENTO SMART
list_of_files = glob.glob('statepoint.*.h5')
if not list_of_files:
    raise FileNotFoundError("Nessun file statepoint trovato nella cartella!")
    
latest_file = max(list_of_files, key=os.path.getctime)
print(f"--- Lettura dati da: {latest_file} ---")

sp = openmc.StatePoint(latest_file)
t_eff = sp.get_tally(name='mappa_fissioni')

mesh_filter = t_eff.find_filter(openmc.MeshFilter)
dims = np.array(mesh_filter.mesh.dimension)
lower_left = np.array(mesh_filter.mesh.lower_left)
upper_right = np.array(mesh_filter.mesh.upper_right)
spacing = (upper_right - lower_left) / dims

# 2. ESTRAZIONE DATI FISICI
# nu-fission = rateo di produzione neutroni (nu * Sigma_f * phi)
nu_fiss = t_eff.get_values(scores=['nu-fission']).reshape((dims[2], dims[1], dims[0])).transpose(2, 1, 0)
# absorption = rateo di assorbimento (Sigma_a * phi)
abs_    = t_eff.get_values(scores=['absorption']).reshape((dims[2], dims[1], dims[0])).transpose(2, 1, 0)

# 3. CALCOLO k_inf LOCALE
# Il flusso (phi) si elide: k_inf = (nu * Sigma_f) / Sigma_a
k_inf = nu_fiss / (abs_ + 1e-15)

grid = pv.ImageData()
grid.dimensions = dims + 1
grid.origin = lower_left
grid.spacing = spacing
grid.cell_data["k_inf"] = k_inf.flatten()

# --- PLOT VOLUMETRICO 3D ---
pl = pv.Plotter(window_size=[1000, 800])

# Volume rendering del k_inf
pl.add_volume(grid, 
              scalars="k_inf",
              cmap="turbo",       # 'turbo' evidenzia bene i picchi di reattività
              opacity="linear",   # Zone a basso k_inf (moderatore) diventano trasparenti
              shade=True,         
              scalar_bar_args={'title': "Local k_inf (nu*Sigma_f / Sigma_a)"} 
             )

# Box esterno
outline = grid.outline()
pl.add_mesh(outline, color="black")

pl.add_text("Mappa Volumetrica Reattività Locale (k_inf)", position='upper_left')

pl.view_isometric() 
pl.add_axes()
pl.show()

sp.close()

### --- ANALISI FISICA: MAPPA k_INFINITO ---
# DOVE: Mesh 3D globale.
# COSA: Fattore di moltiplicazione infinito locale.
# FISICA: 
#   - Se k_inf > 1.0: Zona SUPERCRITICA locale. Il materiale produce più neutroni di quanti ne assorba. 
#     Tipico del fuel fresco o ben moderato (es. valori 1.2 - 1.5).
#   - Se k_inf = 1.0: Zona CRITICA. Bilancio perfetto.
#   - Se k_inf < 1.0: Zona SUBCRITICA. Prevale l'assorbimento o scarsa fissione (es. veleni, moderatore puro, fuel esaurito).
#   - Se k_inf ~ 0.0: Riflettore o schermature (assorbimento puro senza produzione).