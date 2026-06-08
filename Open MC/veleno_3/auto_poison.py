import os
import pandas as pd
import openmc
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# SETUP INIZIALE E CARICAMENTO PARAMETRI
# ==========================================
root_dir = os.getcwd()

def load_parameters(filename='parametri.txt'):
    params = {}
    context = {'np': np}
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filename} non trovato in {root_dir}")
        
    with open(filepath, 'r') as f:
        exec(f.read(), context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

def densita_mix_dinamica(T_K, P_Pa, frac_mass_H2O):
    rho_H2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water') / 1000     
    rho_D2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'HeavyWater') / 1000
    return 1 / (frac_mass_H2O / rho_H2O + (1 - frac_mass_H2O) / rho_D2O) 

def parse_kmax_file(filename):
    data = []
    if not os.path.exists(filename):
        return pd.DataFrame()
        
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    for line in lines[2:]:
        line = line.strip()
        if not line: continue
        cols = [col.strip() for col in line.split('|')]
        if len(cols) == 4:
            data.append([
                float(cols[0]), float(cols[1]), 
                *map(float, cols[2].split('+/-')), 
                *map(float, cols[3].split('+/-'))
            ])
            
    return pd.DataFrame(data, columns=['Pressione_atm', 'Pitch', 'k_max', 'sigma_k', 'Delta_rho_pcm', 'err_rho_pcm'])

def parse_result_final(filename="result_final.txt"):
    data = []
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('=') or 'Pres' in line:
                continue
            cols = line.split()
            if len(cols) >= 12:
                try:
                    data.append(list(map(float, cols[:12])))
                except ValueError:
                    continue
    columns = ['pressione_atm', 'pitch', 'perc_water', 'arricch_INT', 'arricch_EXT', 
               't_water', 't_fuel', 'k_max', 'std_k_max', 'k_auto', 'std_k_auto', 'compatibilita']
    return pd.DataFrame(data, columns=columns)

# ==========================================
# GENERAZIONE MODELLO GEOMETRICO
# ==========================================
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

    materials = openmc.Materials(fuel_mats + [cladding, water, reflector, void_air])

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
    settings.batches = 30
    settings.inactive = 10
    settings.particles = 100000
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
    
    # --- PLOTTING ---
    color_map = {
        cladding: 'lightgray',
        water: 'lightblue',
        reflector: 'green',
        void_air: 'white'
    }

    cmap = plt.get_cmap('YlOrBr')
    for i, f in enumerate(fuel_mats):
        norm_val = i / (num_rings - 1) if num_rings > 1 else 0.5
        rgba = cmap(0.9 - norm_val * 0.6)
        color_map[f] = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    p1 = openmc.Plot()
    p1.basis = 'xy'
    p1.origin = (0.0, 0.0, H_CORE / 2.0)
    p1.width = (3 * (R_CORE + REF_SIDE + 2.0), 3 * (R_CORE + REF_SIDE + 2.0))
    p1.pixels = (800, 800)
    p1.color_by = 'material'
    p1.colors = color_map
    p1.filename = f'reactor_xy_mult_{moltiplicatore}'

    p2 = openmc.Plot()
    p2.basis = 'xz'
    p2.origin = (0.0, 0.0, (H_CORE - REF_BOT) / 2.0) 
    p2.width = (3 * (R_CORE + REF_SIDE + 2.0), H_CORE + REF_BOT + 15.0)
    p2.pixels = (800, 800)
    p2.color_by = 'material'
    p2.colors = color_map
    p2.filename = f'reactor_xz_mult_{moltiplicatore}'

    plots = openmc.Plots([p1, p2])

    return openmc.model.Model(geometry=geometry, materials=materials, settings=settings), plots, R_CORE, PITCH

# ---
# LOOP PRINCIPALE
# ---
def main():
    os.environ["OMP_NUM_THREADS"] = "40"
    
    p = load_parameters('parametri.txt')
    df_results = parse_result_final('result_final.txt')
    
    moltiplicatori_to_test = [4] # <-- INSERISCI QUI I MOLTIPLICATORI DA TESTARE
    
    path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"
    path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
    base_path = path_arco if os.path.exists(path_arco) else path_pc
    openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"

    for moltiplicatore in moltiplicatori_to_test:
        
        
        # Cerca i risultati di ottimizzazione associati al pitch scalato
        row = df_results[np.isclose(df_results['pitch'], moltiplicatore, atol=1e-4)]

        a_int = row['arricch_INT'].values[0] / 100.0
        a_ext = row['arricch_EXT'].values[0] / 100.0
        t_fuel = row['t_fuel'].values[0]
        t_water = row['t_water'].values[0]
        perc_water = row['perc_water'].values[0]
        
        work_dir = os.path.join(root_dir, f"eigenvalue_mult_{moltiplicatore:.3f}".replace('.', '_'))
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"\n{'='*50}\nAVVIO EIGENVALUE: Moltiplicatore {moltiplicatore:.3f} (Pitch {moltiplicatore:.3f} cm)\n{'='*50}")
        
        os.chdir(work_dir)
        
        try:
            model, plots, r_core_eff, pitch_eff = build_k_eff_model(moltiplicatore, p, a_int, a_ext, t_fuel, t_water, perc_water)
            
            # Esportazione
            model.export_to_xml()
            plots.export_to_xml()
            
            # Generazione grafici geometrici
            openmc.plot_geometry() 
            print(f"Grafici generati in {work_dir}")
            
            # Esecuzione simulazione K-eff
            sp_filename = model.run(output=False)
            
            # Parsing StatePoint
            with openmc.StatePoint(sp_filename) as sp:
                k = getattr(sp, 'keff', getattr(sp, 'k_combined', None))
                if k is not None:
                    print(f"-> R_CORE: {r_core_eff:.3f} cm | PITCH: {pitch_eff:.3f} cm")
                    print(f"-> Simulazione conclusa. k_eff = {k.nominal_value:.5f} ± {k.std_dev:.5f}")
                else:
                    print("-> Errore: Impossibile trovare k_eff nel file StatePoint.")
            
            # Pulizia automatica file pesanti se necessario
            if os.path.exists('summary.h5'):
                os.remove('summary.h5')
                
        except Exception as e:
            print(f"-> ERRORE CRITICO sul moltiplicatore {moltiplicatore}: {e}")
            
        finally:
            os.chdir(root_dir)

if __name__ == "__main__":
    main()