# PRIMA ESEGUO REATTORE.PY PER OTTENERE IL FILE H5,
# POI ESEGUO QUESTO SCRIPT PER CONVERTIRLO IN VTI PER PARAVIEW
#DOPO APRO PARAWIEW E CARICO REATTORE.VTI
# DEVI PREMERE APPLY PER VEDERE I DATI
# POI METTERE I COLORI SULLA COLONNA CELL DATA -> Mat_ID
# POI METTI SURFACE AFFIANCO

import h5py
import numpy as np
import pyvista as pv

filename = 'mappa_3d.h5'
output_filename = 'reattore.vti'

print("Conversione in corso...")
with h5py.File(filename, 'r') as f:
    dims = f.attrs['num_voxels']
    spacing = f.attrs['voxel_width']
    origin = f.attrs['lower_left']
    data = np.array(f['data'], dtype=np.int32)

grid = pv.ImageData()
grid.dimensions = np.array(dims) + 1
grid.origin = origin
grid.spacing = spacing
grid.cell_data["Mat_ID"] = data.flatten(order='C') # Usa 'F' se vedi strisce strane

grid.save(output_filename)
print(f"Salvato: {output_filename}")