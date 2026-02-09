import openmc
import pyvista as pv
import numpy as np
import glob
import os

# --- CARICAMENTO ---
list_of_files = glob.glob('statepoint.*.h5')
if not list_of_files:
    raise FileNotFoundError("Nessun file statepoint trovato!")
latest_file = max(list_of_files, key=os.path.getctime)
print(f"--- Lettura dati da: {latest_file} ---")

sp = openmc.StatePoint(latest_file)

# Recupero Tally (adatta il nome se necessario)
try:
    t_mfp = sp.get_tally(name='mappa_fissioni') # O 'mappa_mfp'
except ValueError:
    t_mfp = sp.get_tally(id=1) 

mesh_filter = t_mfp.find_filter(openmc.MeshFilter)
dims = np.array(mesh_filter.mesh.dimension)
lower_left = np.array(mesh_filter.mesh.lower_left)
upper_right = np.array(mesh_filter.mesh.upper_right)
spacing = (upper_right - lower_left) / dims

rr_total = t_mfp.get_values(scores=['total']).reshape((dims[2], dims[1], dims[0])).transpose(2, 1, 0)
flux = t_mfp.get_values(scores=['flux']).reshape((dims[2], dims[1], dims[0])).transpose(2, 1, 0)

# --- CALCOLO MFP ---
with np.errstate(divide='ignore', invalid='ignore'):
    sigma_t = np.divide(rr_total, flux, out=np.zeros_like(rr_total), where=flux!=0)
    mfp = np.divide(1.0, sigma_t, out=np.full_like(sigma_t, np.nan), where=sigma_t>1e-10)

# DEBUG: Stampa i valori min e max per capire su cosa filtrare
print(f"Statistiche MFP: Min={np.nanmin(mfp):.2f} cm, Max={np.nanmax(mfp):.2f} cm")

soglia_vuoto = 50.0

# 2. Sostituiamo i valori enormi con NaN (Not a Number). 
# PyVista renderizza i NaN come "invisibili" o grigi, pulendo il grafico.
mfp_clean = np.where(mfp > soglia_vuoto, np.nan, mfp)

print(f"Statistiche PULITE: Min={np.nanmin(mfp_clean):.2f}, Max={np.nanmax(mfp_clean):.2f}")

# --- PYVISTA ---
grid = pv.ImageData()
grid.dimensions = dims + 1
grid.origin = lower_left
grid.spacing = spacing
grid.cell_data["MFP"] = mfp.flatten()

pl = pv.Plotter(shape=(1, 2), window_size=[1400, 600])

# --- VISTA 1: SLICE (TAGLIO) ---
# Tagliamo il reattore a metà lungo l'asse Y per vedere l'interno
pl.subplot(0, 0)
pl.add_text("Sezione Centrale (Slice)", font_size=10)

# slice_x = grid.slice(normal='x') # Se vuoi tagliare lungo X
#slice_y = grid.slice(normal='y') # Taglio lungo Y
slice_z = grid.slice(normal='z') # Taglio lungo Z (pianta)

pl.add_mesh(slice_z, cmap="jet", scalar_bar_args={'title': "MFP [cm]"})
pl.add_mesh(grid.outline(), color="black")
pl.view_xz() # Vista frontale del taglio

# --- VISTA 2: CLIP (SPACCATO) ---
# Tagliamo via un angolo del reattore per vedere dentro (effetto spaccato 3D)
pl.subplot(0, 1)
pl.add_text("Spaccato 3D (Clip)", font_size=10)

# Clip: Taglia via tutto ciò che sta "davanti" a un piano
clipped = grid.clip(normal='-y', origin=grid.center) 

pl.add_mesh(clipped, 
            scalars="MFP", 
            cmap="jet", 
            clim=[0, 15],
            show_edges=False,
            scalar_bar_args={'title': "MFP [cm]"}
           )
pl.add_mesh(grid.outline(), color="black")
pl.view_isometric()

pl.show()
sp.close()