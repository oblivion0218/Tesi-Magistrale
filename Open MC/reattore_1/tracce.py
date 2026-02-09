import h5py
import numpy as np
import pyvista as pv
import os

# --- Percorsi file ---
path_tracks = 'tracks.h5'  # Percorso modificato come richiesto

pl = pv.Plotter()

# --- 1. CARICAMENTO GEOMETRIA DI SFONDO ---
with h5py.File(path_tracks, 'r') as f:
    if 'num_voxels' in f.attrs:
        dims = f.attrs['num_voxels']
        spacing = f.attrs['voxel_width']
        origin = f.attrs['lower_left']
        data = np.array(f['data'], dtype=np.int32)
        
        grid = pv.ImageData(dimensions=np.array(dims)+1, spacing=spacing, origin=origin)
        grid.cell_data["Mat_ID"] = data.flatten(order='C')
        
        # Fuel (Tue impostazioni: opacity=0.05)
        fuel = grid.threshold([0.9, 1.1]) 
        if fuel.n_cells > 0:
            pl.add_mesh(fuel, color='yellow', opacity=0.05, label='Fuel')
        
        # Moderatore (Tue impostazioni: opacity=0)
        water = grid.threshold([2.9, 3.1])
        if water.n_cells > 0:
            pl.add_mesh(water, color='blue', opacity=0.0, label='Water')

#-- 2. ANALISI E VISUALIZZAZIONE TRACCE ---
if os.path.exists(path_tracks):
    print(f"Analisi file: {path_tracks}")
    with h5py.File(path_tracks, 'r') as f:
        
       # A) IDENTIFICAZIONE E CALCOLO LIMITI GLOBALI
        track_keys = [k for k in f.keys() if k.startswith('track_')]
        total_tracks = len(track_keys)
        
        all_log_energies = []
        if total_tracks > 0:
            for k in track_keys:
                dset = f[k]
                # Se 'data' non esiste come campo, il dataset stesso contiene i campi 'E', 'r', ecc.
                if 'data' in dset.dtype.names:
                    e_data = dset['data']['E']
                else:
                    e_data = dset['E']
                
                all_log_energies.append(np.log10(e_data + 1e-5))
            
            full_arr = np.concatenate(all_log_energies)
            global_clim = [np.min(full_arr), np.max(full_arr)]
            print(f"Range Energia Globale (Log10): {global_clim}")

        # B) CICLO DI VISUALIZZAZIONE
        for idx, key in enumerate(track_keys):
            dset = f[key]
            data = dset[:] 
            
            # Conversione coordinate (Fix per PyVista)
            coords = np.array(data['r'].tolist()) 
            energies = data['E']
            mat_ids = data['material_id']
            
            # Info testuali (Primi 10 passi)
            print(f"\n--- Neutrone: {idx + 1} ---")
            print(f"Passi totali: {len(coords)}")
        
            for i in range(min(10, len(coords))):
                pos = coords[i]
                energy = energies[i]
                mat_id = mat_ids[i]
                print(f" Step {i+1}: Pos {pos}, E={energy:.2e} eV, Mat_ID={mat_id}")

            if len(coords) > 1:
                line = pv.lines_from_points(coords)
                line['Energy'] = np.log10(energies + 1e-5) 
                
                # C) GESTIONE BARRA DEI COLORI UNICA
                # Mostriamo la barra solo se siamo all'ultima iterazione del ciclo
                is_last_track = (idx == total_tracks - 1)
                
                pl.add_mesh(line, 
                            render_lines_as_tubes=True, 
                            line_width=5, 
                            cmap="plasma",
                            clim=global_clim,              # FONDAMENTALE: Scala fissa per tutti
                            show_scalar_bar=is_last_track, # True solo per l'ultimo
                            scalar_bar_args={
                                'title': 'Log(E) [eV]',
                                'height': 0.2,    # Altezza 20%
                                'width': 0.4,    # Larghezza 5%
                                'position_x':  0.5, 
                                'position_y': 0.05,
                                'color': 'black'
                            } if is_last_track else None
                           )
                
                # Punti Start/End
                pl.add_points(coords[0], color='red', point_size=8, render_points_as_spheres=True)
                pl.add_points(coords[-1], color='green', point_size=8, render_points_as_spheres=True)

else:
    print(f"File {path_tracks} non trovato! Esegui prima la simulazione OpenMC.")

pl.add_axes()
pl.show()