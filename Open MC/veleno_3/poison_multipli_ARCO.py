import os
import shutil
import pandas as pd
import openmc
import openmc.deplete
import numpy as np
from CoolProp.CoolProp import PropsSI
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP INIZIALE E CARICAMENTO PARAMETRI
# ==========================================
root_dir = os.getcwd()

def load_parameters(filename='parametri.txt'):
    """Legge i parametri dal file txt nella root directory."""
    params = {}
    context = {'np': np}
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filename} non trovato in {root_dir}")
        
    with open(filepath, 'r') as f:
        exec(f.read(), context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

p = load_parameters()

# Estrazione grandezze geometriche e fisiche
R_CORE = p['R_CORE']
REF_SIDE = p['REF_SIDE']
REF_BOT = p['REF_BOT']
H_CORE = p['H_CORE']
R_PIPE = p['R_PIPE']
R_FUEL = p['R_FUEL']
CLAD_THICK = p['CLAD_THICK']
P_TARGET = p['P_TARGET'] # Serve ancora per sapere la quota z_mid della sorgente

S_val = p['S']
yield_fn = 5.87905e-03 # Conversione fotoni -> neutroni
S_rate = S_val * yield_fn

# Configurazione Monte Carlo (fallback di sicurezza inserito)
batches = p.get('batches_fix', 100)
particles = p.get('particles_fix', 5000)
pressione_funzionamento = p['Pressione_funzionamento']

# ==========================================
# 2. MODULI FISICI E TERMODINAMICI
# ==========================================
def densita_mix_dinamica(T_K, P_Pa=pressione_funzionamento, frac_mass_H2O=1.0):
    rho_H2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water') / 1000
    rho_D2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'HeavyWater') / 1000
    return 1 / (frac_mass_H2O / rho_H2O + (1 - frac_mass_H2O) / rho_D2O)

# ==========================================
# 3. GENERAZIONE DINAMICA DEL MODELLO
# ==========================================
def build_model(current_pitch, a_int, a_ext, t_fuel, t_water, perc_water, root_dir):
    num_rings = int(np.ceil(R_CORE / current_pitch)) + 1
    n_rings_removed = int(np.ceil(R_PIPE / current_pitch))
    EDGE_PIPE_LARGE = (n_rings_removed - 0.5) * current_pitch if n_rings_removed > 0 else 0.1
    
    # Area esatta per un prisma esagonale
    AREA_FUEL = (3 * np.sqrt(3) / 2) * (R_FUEL**2)
    
    # --- MATERIALI ---
    cladding = openmc.Material(name='cladding')
    cladding.add_element('Zr', 1.0)
    cladding.set_density('g/cm3', 6.49)
    cladding.temperature = t_fuel

    water = openmc.Material(name='water')
    water.add_nuclide('H1', perc_water)
    water.add_nuclide('H2', 1.0 - perc_water)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', densita_mix_dinamica(t_water, frac_mass_H2O=perc_water))
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

    marker_mat = openmc.Material(name='marker')
    marker_mat.set_density('g/cm3', 1e-10)
    marker_mat.add_nuclide('He4', 1.0)

    fuel_mats = []
    for i in range(num_rings):
        e_i = a_ext if num_rings == 1 else a_int - (a_int - a_ext) * (i / (num_rings - 1))
        f = openmc.Material(name=f'fuel_ring_{i}')
        f.add_nuclide('U235', e_i)
        f.add_nuclide('U238', 1.0 - e_i)
        f.add_nuclide('O16', 2.0)
        f.set_density('g/cm3', 10.96)
        f.temperature = t_fuel
        
        ring_radius = i * current_pitch
        # Condizione di esistenza fisica per la depletion:
        is_active = (ring_radius + R_FUEL > EDGE_PIPE_LARGE) and (ring_radius - R_FUEL < R_CORE)
        
        if is_active:
            f.depletable = True 
            n_pins = 1 if i == 0 else 6 * i
            f.volume = n_pins * AREA_FUEL * H_CORE
        else:
            f.depletable = False
            
        fuel_mats.append(f)

    materials = openmc.Materials(fuel_mats + [cladding, water, reflector, void_air, marker_mat])
    materials.export_to_xml()

    # --- GEOMETRIA ---
    z_bot_ref = openmc.ZPlane(z0=-REF_BOT, boundary_type='vacuum')
    z_bot_core = openmc.ZPlane(z0=0.0)
    z_top_core = openmc.ZPlane(z0=H_CORE)
    z_top_world = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')

    z_mid = H_CORE - P_TARGET
    sphere_src = openmc.Sphere(x0=0, y0=0, z0=H_CORE + 2, r=0.5)

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
    lattice_universes = [[fuel_universes[0]]] + [[fuel_universes[i]] * (6 * i) for i in range(1, num_rings)]

    lattice = openmc.HexLattice()
    lattice.center = (0.0, 0.0)
    lattice.pitch = (current_pitch,)
    lattice.outer = u_water
    lattice.universes = lattice_universes[::-1]
    lattice.orientation = 'x'

    hex_core_edge = num_rings * current_pitch
    hex_ref_edge = hex_core_edge + REF_SIDE
    prism_core = openmc.model.HexagonalPrism(edge_length=hex_core_edge, orientation='x')
    prism_reflector = openmc.model.HexagonalPrism(edge_length=hex_ref_edge, orientation='x', boundary_type='vacuum')

    # Assemblaggio domini: il canale centrale è interamente vuoto
    region_core = -prism_core & +hex_pipe_prism & +z_bot_core & -z_top_core
    c_main_core = openmc.Cell(fill=lattice, region=region_core)

    region_void_gap = -hex_pipe_prism & +z_bot_core & -z_top_core
    c_void_gap = openmc.Cell(fill=void_air, region=region_void_gap)

    region_ref_side = -prism_reflector & +prism_core & +z_bot_core & -z_top_core
    c_ref_side = openmc.Cell(fill=reflector, region=region_ref_side)

    region_ref_bot = -prism_reflector & +z_bot_ref & -z_bot_core
    c_ref_bot = openmc.Cell(fill=reflector, region=region_ref_bot)

    c_source_marker = openmc.Cell(fill=marker_mat, region=-sphere_src)
    region_top = -prism_reflector & +z_top_core & -z_top_world & +sphere_src
    c_top_void = openmc.Cell(fill=void_air, region=region_top)

    geometry = openmc.Geometry([c_main_core, c_void_gap, c_ref_side, c_ref_bot, c_source_marker, c_top_void])
    geometry.export_to_xml()

    # --- SETTINGS ---
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = batches
    settings.particles = particles
    settings.max_collisions = 10000
    settings.max_lost_particles = 50
    settings.temperature = {'method': 'interpolation'}
    settings.photon_transport = False
    settings.create_fission_neutrons = True

    # Estrazione distribuzioni (Path assoluti)
    energy_file = os.path.join(root_dir, "photoneutrons_energy.txt")
    polar_file = os.path.join(root_dir, "photoneutrons_polar.txt")

    data_energy = np.loadtxt(energy_file, skiprows=1)
    E_mid, p_E = data_energy[:, 0], data_energy[:, 1]
    p_E /= np.trapezoid(p_E, x=E_mid)
    energy_dist = openmc.stats.Tabular(E_mid, p_E, interpolation='linear-linear')

    data_polar = np.loadtxt(polar_file, skiprows=1)
    theta, p_theta = data_polar[:, 0], data_polar[:, 1]
    mu = np.cos(theta)
    
    sort_idx = np.argsort(mu)
    mu_sorted, p_mu_sorted = mu[sort_idx], p_theta[sort_idx]
    p_mu_sorted /= np.trapezoid(p_mu_sorted, x=mu_sorted)

    mu_dist = openmc.stats.Tabular(mu_sorted, p_mu_sorted, interpolation='linear-linear')
    phi_dist = openmc.stats.Uniform(0.0, 2 * np.pi)
    angle_dist = openmc.stats.PolarAzimuthal(mu=mu_dist, phi=phi_dist)

    source = openmc.IndependentSource()
    source.particle = 'neutron'
    source.space = openmc.stats.Point((0.0, 0.0, z_mid))
    source.angle = angle_dist
    source.energy = energy_dist
    settings.source = source

    # --- TALLIES ---
    tallies = openmc.Tallies()
    fission_tot = openmc.Tally(name='fission_tot')
    fission_tot.scores = ['kappa-fission', 'nu-fission', 'absorption']
    tallies.append(fission_tot)

    vacuum_surfaces = [surf for surf in geometry.get_all_surfaces().values() if surf.boundary_type == 'vacuum']
    surface_filter = openmc.SurfaceFilter(vacuum_surfaces)
    tally_leakage = openmc.Tally(name='leakage_tot')
    tally_leakage.filters = [surface_filter]
    tally_leakage.scores = ['current']
    tallies.append(tally_leakage)

    return openmc.model.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)

# ==========================================
# 4. LOOP PRINCIPALE E DEPLETION
# ==========================================
def parse_result_final(filename="result_final.txt"):
    data = []
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filename} non trovato. Esegui prima l'ottimizzatore.")
        
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

def main():
    # Ottimizzazione multithreading
    os.environ["OMP_NUM_THREADS"] = "40"
    
    # ---------------------------------------------------------
    # CONFIGURAZIONE PATH E VARIABILI (DA VERIFICARE/MODIFICARE)
    # ---------------------------------------------------------
    df_results = parse_result_final('result_final.txt')
    pitches_to_test = [1.0, 1.25,1.5, 1.875 , 2.0, 2.25 , 2.5] # <-- INSERISCI I PITCH CHE VUOI SIMULARE QUI
    
    path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"
    path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
    base_path = path_arco if os.path.exists(path_arco) else path_pc
    openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"
    chain_file = f"{base_path}/chain_endfb71_pwr.xml"

    # ---------------------------------------------------------
    # ESECUZIONE
    # ---------------------------------------------------------
    for current_pitch in pitches_to_test:
        row = df_results[np.isclose(df_results['pitch'], current_pitch, atol=1e-4)]
        if row.empty:
            print(f"ATTENZIONE: Pitch {current_pitch} non trovato in result_final.txt. Salto.")
            continue
            
        a_int = row['arricch_INT'].values[0] / 100.0
        a_ext = row['arricch_EXT'].values[0] / 100.0
        t_fuel = row['t_fuel'].values[0]
        t_water = row['t_water'].values[0]
        perc_water = row['perc_water'].values[0]
        
        work_dir = os.path.join(root_dir, f"depletion_pitch_{current_pitch:.3f}".replace('.', '_'))
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"\n{'='*50}\nAVVIO DEPLETION: Pitch {current_pitch:.3f} cm\nArricch. Radiale: Centro = {a_int*100:.2f}%, Esterno = {a_ext*100:.2f}%\nWorkspace: {work_dir}\n{'='*50}")
        
        os.chdir(work_dir)
        
        try:
            # 1. Creazione architettura core
            model = build_model(current_pitch, a_int, a_ext, t_fuel, t_water, perc_water, root_dir)
            
            # 2. Inizializzazione Operatore
            operator = openmc.deplete.CoupledOperator(model, chain_file, normalization_mode="source-rate")
            
            # 3. Profilo Temporale e Flusso Fotonico
            giorni_on_transitorio = [0.1, 0.4, 0.5, 1.0, 3.0, 5.0, 10.0]
            giorni_on_burnup = [30.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
            giorni_off = [0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 3.0] + [3.0] * 4
            giorni_off_coda = [10.0, 15.0, 20.0] 
            
            time_steps = giorni_on_transitorio + giorni_on_burnup + giorni_off + giorni_off_coda
            source_rates = [S_rate] * (len(giorni_on_transitorio) + len(giorni_on_burnup)) + [1e-15] * (len(giorni_off) + len(giorni_off_coda))
            
            # 4. Run Integrazione (Unità di tempo esplicitate in GIORNI)
            integrator = openmc.deplete.PredictorIntegrator(operator, time_steps, source_rates=source_rates, timestep_units='d')
            integrator.integrate()
            
            print(f"-> Simulazione conclusa con successo per pitch {current_pitch:.3f}.")
            
        except Exception as e:
            print(f"-> ERRORE CRITICO sul pitch {current_pitch}: {e}")
            
        finally:
            # Garantisce il ritorno alla directory padre anche in caso di crash
            os.chdir(root_dir)

if __name__ == "__main__":
    main()