import os
import pandas as pd
import openmc
import openmc.deplete
import numpy as np
from CoolProp.CoolProp import PropsSI
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
root_dir = os.getcwd()

def load_parameters(filename='parametri.txt'):
    params = {}
    context = {'np': np}
    filepath = os.path.join(root_dir, filename)
    with open(filepath, 'r') as f:
        exec(f.read(), context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

def densita_mix_dinamica(T_K, P_Pa, frac_mass_H2O):
    rho_H2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water') / 1000     
    rho_D2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'HeavyWater') / 1000
    return 1 / (frac_mass_H2O / rho_H2O + (1 - frac_mass_H2O) / rho_D2O) 

def parse_result_final(filename="result_final.txt"):
    data = []
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath): return pd.DataFrame()
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('=') or 'Pres' in line: continue
            cols = line.split()
            if len(cols) >= 12:
                try: data.append(list(map(float, cols[:12])))
                except ValueError: continue
    columns = ['pressione_atm', 'pitch', 'perc_water', 'arricch_INT', 'arricch_EXT', 
               't_water', 't_fuel', 'k_max', 'std_k_max', 'k_auto', 'std_k_auto', 'compatibilita']
    return pd.DataFrame(data, columns=columns)

def build_k_eff_model(moltiplicatore, p, a_int, a_ext, t_fuel, t_water, perc_water):
    # Parametri Scalati
    REF_SIDE, REF_BOT, H_CORE = p['REF_SIDE'], p['REF_BOT'], p['H_CORE']
    R_PIPE, R_FUEL, CLAD_THICK = p['R_PIPE'], p['R_FUEL'], p['CLAD_THICK']
    pressione_funzionamento = p['Pressione_funzionamento']
    
    R_CORE = p['R_CORE'] * moltiplicatore
    PITCH = p['PITCH'] * moltiplicatore
    
    num_rings = int(np.ceil(R_CORE / PITCH)) + 1
    n_rings_removed = int(np.ceil(R_PIPE / PITCH))
    EDGE_PIPE_LARGE = (n_rings_removed - 0.5) * PITCH if n_rings_removed > 0 else 0.1
    
    # Materiali Costanti
    cladding = openmc.Material(name='cladding')
    cladding.add_element('Zr', 1.0)
    cladding.set_density('g/cm3', 6.49)
    cladding.temperature = t_fuel

    marker_mat = openmc.Material(name='marker')
    marker_mat.set_density('g/cm3', 1e-10)
    marker_mat.add_nuclide('He4', 1.0)

    water = openmc.Material(name='water')
    water.add_nuclide('H1', perc_water)
    water.add_nuclide('H2', 1.0 - perc_water)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', densita_mix_dinamica(t_water, pressione_funzionamento, perc_water))
    water.add_s_alpha_beta('c_H_in_H2O')
    water.add_s_alpha_beta('c_D_in_D2O')
    water.temperature = t_water

    reflector = openmc.Material(name='reflector')
    reflector.add_element('C', 1.0)
    reflector.set_density('g/cm3', 1.75)
    reflector.temperature = t_fuel

    void_air = openmc.Material(name='void_air')
    void_air.set_density('g/cm3', 1e-10)
    void_air.add_nuclide('N14', 1.0)

    # Vettori Fuel per Geometria
    fuel_mats = []
    for i in range(num_rings):
        e_i = a_int - (a_int - a_ext) * (i / (num_rings - 1)) if num_rings > 1 else a_int
        f = openmc.Material(name=f'fuel_ring_{i}')
        f.add_nuclide('U235', e_i)
        f.add_nuclide('U238', 1.0 - e_i)
        f.add_nuclide('O16', 2.0)
        f.set_density('g/cm3', 10.96)
        f.temperature = t_fuel
        
        ring_radius = i * PITCH
        # FIX: Casting esplicito a bool nativo di Python
        is_active = bool((ring_radius > EDGE_PIPE_LARGE) and (ring_radius - R_FUEL < R_CORE))
        
        f.depletable = is_active
        if is_active: f.volume = 1 
        fuel_mats.append(f)

    materials = openmc.Materials(fuel_mats + [cladding, water, reflector, void_air, marker_mat])

    # Geometria
    z_bot_ref = openmc.ZPlane(z0=-REF_BOT, boundary_type='vacuum')
    z_bot_core = openmc.ZPlane(z0=0.0)
    z_top_core = openmc.ZPlane(z0=H_CORE)
    z_top_world = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')

    hex_fuel_prism = openmc.model.HexagonalPrism(edge_length=R_FUEL, orientation='x')
    hex_clad_prism = openmc.model.HexagonalPrism(edge_length=R_FUEL + CLAD_THICK, orientation='x')
    hex_pipe_prism = openmc.model.HexagonalPrism(edge_length=EDGE_PIPE_LARGE, orientation='x')

    fuel_universes = []
    for i, f_mat in enumerate(fuel_mats):
        c_f = openmc.Cell(fill=f_mat, region=-hex_fuel_prism)
        c_c = openmc.Cell(fill=cladding, region=+hex_fuel_prism & -hex_clad_prism)
        c_w = openmc.Cell(fill=water, region=+hex_clad_prism)
        fuel_universes.append(openmc.Universe(cells=[c_f, c_c, c_w]))

    u_water = openmc.Universe(cells=[openmc.Cell(fill=water)])
    
    lattice_universes = []
    for i in range(num_rings):
        lattice_universes.append([fuel_universes[i]] * (1 if i == 0 else 6 * i))

    lattice = openmc.HexLattice()
    lattice.center = (0.0, 0.0)
    lattice.pitch = (PITCH,)
    lattice.outer = u_water
    lattice.universes = lattice_universes[::-1]
    lattice.orientation = 'x'

    hex_core_edge = num_rings * PITCH
    hex_ref_edge = hex_core_edge + REF_SIDE
    prism_core = openmc.model.HexagonalPrism(edge_length=hex_core_edge, orientation='x')
    prism_reflector = openmc.model.HexagonalPrism(edge_length=hex_ref_edge, orientation='x', boundary_type='vacuum')

    c_main_core = openmc.Cell(fill=lattice, region=-prism_core & +hex_pipe_prism & +z_bot_core & -z_top_core)
    c_void_gap = openmc.Cell(fill=void_air, region=-hex_pipe_prism & +z_bot_core & -z_top_core)
    c_ref_side = openmc.Cell(fill=reflector, region=-prism_reflector & +prism_core & +z_bot_core & -z_top_core)
    c_ref_bot = openmc.Cell(fill=reflector, region=-prism_reflector & +z_bot_ref & -z_bot_core)
    c_top_void = openmc.Cell(fill=void_air, region=-prism_reflector & +z_top_core & -z_top_world)

    geometry = openmc.Geometry([c_main_core, c_void_gap, c_ref_side, c_ref_bot, c_top_void])

    # Settings Autovalore
    settings = openmc.Settings()
    settings.run_mode = 'eigenvalue'
    settings.temperature = {'method': 'interpolation'}
    
    # 1. Troviamo analiticamente l'indice del primo anello di combustibile attivo
    i_active = None
    for i in range(num_rings):
        ring_radius = i * PITCH
        if (ring_radius > EDGE_PIPE_LARGE) and (ring_radius - R_FUEL < R_CORE):
            i_active = i
            break
            
    if i_active is None:
        raise ValueError(f"Nessun pin attivo trovato per moltiplicatore {moltiplicatore}")

    # 2. In un HexLattice con orientation='x', i pin sull'asse X si trovano esattamente a (i * PITCH, 0)
    x_source = i_active * PITCH 
    y_source = 0.0
    z_source = H_CORE / 2.0  # Fissiamo la sorgente a metà altezza
    
    # 3. Definiamo una singola sorgente puntiforme deterministica
    source_space = openmc.stats.Point((x_source, y_source, z_source))
    settings.source = openmc.IndependentSource(space=source_space)

    return openmc.model.Model(geometry=geometry, materials=materials, settings=settings)
    
def main():
    os.environ["OMP_NUM_THREADS"] = "40"
    p = load_parameters('parametri.txt')
    df_results = parse_result_final('result_final.txt')
    
    path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"
    path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
    base_path = path_arco if os.path.exists(path_arco) else path_pc
    openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"

    moltiplicatori_to_test = [2.750 , 3.250 , 3.750 , 4.250]
 

    batches_norm, particles_norm , inactive_norm = p['batches_auto'], p['particles_auto'], p['inactive_auto']
    batches_high, particles_high , inactive_high = 200, 250000 , 40

    for moltiplicatore in moltiplicatori_to_test:
        row = df_results[np.isclose(df_results['pitch'], moltiplicatore, atol=1e-4)]
        if row.empty: continue
            
        a_int = row['arricch_INT'].values[0] / 100.0
        a_ext = row['arricch_EXT'].values[0] / 100.0
        t_fuel = row['t_fuel'].values[0]
        t_water = row['t_water'].values[0]
        perc_water = row['perc_water'].values[0]

        work_dir = os.path.join(root_dir, f"depletion_pitch_{moltiplicatore:.3f}".replace('.', '_'))
        if not os.path.exists(work_dir): continue
        
        os.chdir(work_dir)
        print(f"\n=======================================================")
        print(f"Analisi transitorio K_eff per moltiplicatore {moltiplicatore:.3f}")
        print(f"=======================================================")

        results_file = "depletion_results.h5"
        if not os.path.exists(results_file):
            print(f"File {results_file} non trovato. Salto.")
            os.chdir(root_dir)
            continue

        # 1. Carichiamo i risultati per estrarre gli ID reali usati nel fit HDF5
        results = openmc.deplete.Results(results_file)
        times_days = results.get_times(time_units="d")
        hdf5_mat_ids = sorted([int(mat_id) for mat_id in results[0].volume.keys()])

        # 2. Generiamo il modello geometrico base (che parte da ID=1)
        openmc.reset_auto_ids()
        model = build_k_eff_model(moltiplicatore, p, a_int, a_ext, t_fuel, t_water, perc_water)
        
        # 3. Troviamo i materiali depletabili corrispondenti nel nuovo modello
        model_depletable_mats = sorted([m for m in model.materials if m.depletable], key=lambda m: m.id)

        # 4. ALGORITMO DI RIALLINEAMENTO: Calcolo del delta di sfasamento degli ID
        if hdf5_mat_ids and model_depletable_mats:
            delta = hdf5_mat_ids[0] - model_depletable_mats[0].id
            if delta != 0:
                print(f"[INFO] Rilevato sfasamento ID. Applico correzione geometrica: Delta = {delta}")
                for mat in model.materials:
                    mat.id += delta  # Trasliamo l'ID dell'oggetto Python (aggiorna automaticamente la cella associata)

        # 5. Esportiamo la geometria corretta e creiamo il template pulito
        model.export_to_xml()
        
        import shutil
        shutil.copy("materials.xml", "materials_base.xml")

        if os.path.exists("tallies.xml"):
            os.remove("tallies.xml")

        output_file = f"ALTA_STAT_k_eff_transitorio_pitch_{moltiplicatore:.3f}".replace('.', '_') + ".txt"
        with open(output_file, "w") as f_out:
            f_out.write(f"Step  Tempo[gg]  K_EFF  STD_DEV  STATISTICA\n")

        # 6. Loop temporale di trasporto
        for step, t in enumerate(times_days):
            if 395.0 <= t <= 410.0 :
                model.settings.batches = batches_high
                model.settings.particles = particles_high
                model.settings.inactive = inactive_high
                stat_label = "HIGH"
            else:
                model.settings.batches = batches_norm
                model.settings.particles = particles_norm
                model.settings.inactive = inactive_norm
                stat_label = "NORM"
            
            model.settings.export_to_xml()

            # Ora l'estrazione funzionerà perfettamente perché gli ID in materials_base.xml coincidono con l'HDF5
            mats = results.export_to_materials(step, path="materials_base.xml")
            
            for mat in mats:
                if mat.name == 'water':
                    mat.add_s_alpha_beta('c_D_in_D2O')
                    mat.add_s_alpha_beta('c_H_in_H2O')
            mats.cross_sections = f"{base_path}/cross_sections.xml"
            mats.export_to_xml()

            openmc.run(output=False)

            sp_file = f"statepoint.{model.settings.batches}.h5"
            if os.path.exists(sp_file):
                with openmc.StatePoint(sp_file) as sp:
                    k_val = sp.k_combined.nominal_value
                    k_std = sp.k_combined.std_dev
                    
                    print(f"Step {step} (t={t:.2f} d): k = {k_val:.5f} +/- {k_std:.5f} ({stat_label})")
                    with open(output_file, "a") as f_out:
                        f_out.write(f"{step:<6} {t:<15.4f} {k_val:<15.5f} {k_std:<15.5f} {stat_label:<15}\n")
                os.remove(sp_file)
            if os.path.exists('summary.h5'): os.remove('summary.h5')

        os.chdir(root_dir)

if __name__ == "__main__":
    main()
