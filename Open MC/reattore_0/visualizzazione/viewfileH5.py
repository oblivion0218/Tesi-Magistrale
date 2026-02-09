import h5py
import numpy as np
import pyvista as pv
import os

filename = 'mappa_3d.h5'

print(f"Lettura da {filename}...")

with h5py.File(filename, 'r') as f:
    dims = f.attrs['num_voxels']
    spacing = f.attrs['voxel_width']
    origin = f.attrs['lower_left']
    raw_data = np.array(f['data'], dtype=np.int32)

# Griglia base
grid = pv.ImageData()
grid.dimensions = np.array(dims) + 1
grid.origin = origin
grid.spacing = spacing
grid.cell_data["Mat_ID"] = raw_data.flatten(order='C')

# --- CONFIGURAZIONE PLOT ---
pl = pv.Plotter()

# 1. CORE (Fuel=1, Clad=2) -> OPACO
# Range: 0.9 a 2.1 prende ID 1 e 2
core_mesh = grid.threshold([0.9, 2.1]) 
pl.add_mesh(core_mesh, 
            cmap=['yellow', 'gray'], # 1=Yellow, 2=Gray
            opacity=1.0, 
            categories=True,
            label='Fuel & Clad')

# 2. SORGENTE (Marker=5) -> OPACO E ROSSO
# Range: 4.9 a 5.1 prende solo ID 5
source_mesh = grid.threshold([4.9, 5.1])
pl.add_mesh(source_mesh, 
            cmap=['red'],  # 5=Red
            opacity=1.0, 
            point_size=10, # Opzionale: rende i punti più visibili se piccoli
            categories=True,
            label='Sorgente')

# 3. ESTERNO (Acqua=3,) -> TRASPARENTE
shell_mesh = grid.threshold([2.9, 3.1])
pl.add_mesh(shell_mesh, 
            cmap=['blue'], # 3=Blue, 4=Green
            opacity=0.35,  # Trasparenza per vedere dentro
            categories=True,
            label='Water')

# 3. ESTERNO (Riflettore=4) -> TRASPARENTE
shell_mesh = grid.threshold([3.9, 4.1])
pl.add_mesh(shell_mesh, 
            cmap=['green'], # 4=Green
            opacity=0.15,  # Trasparenza per vedere dentro
            categories=True,
            label='Refl')

pl.add_title("GEOMETRIA 3D DEL REATTORE", font_size=10)
pl.show()