import os
import openmc.deplete
import openmc
import numpy as np
import shutil
import warnings
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


# --- 1. CARICAMENTO PARAMETRI ---
def load_parameters(filename='parametri.txt'):
    params = {}
    context = {'np': np} 
    with open(filename, 'r') as f:
        exec(f.read(), context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

# Percorso assoluto della cartella principale (dove risiede poison.py)
root_dir = os.path.dirname(os.path.abspath(__file__))
p = load_parameters(os.path.join(root_dir, "parametri.txt"))

# --- Parametri Geometrici e Logici ---
moltiplicatore = 4.0
R_CORE    = p['R_CORE'] * moltiplicatore
REF_SIDE  = p['REF_SIDE']
REF_BOT   = p['REF_BOT']
H_CORE    = p['H_CORE']
R_PIPE    = p['R_PIPE']
R_TARGET  = p['R_TARGET']
H_TARGET  = p['H_TARGET']
P_TARGET  = p['P_TARGET']
S_val     = p['S']
batches   = p['batches_fix']
particles = p['particles_fix']
R_FUEL    = p['R_FUEL']
CLAD_THICK= p['CLAD_THICK']
PITCH     = p['PITCH'] * moltiplicatore
enrich_target = p['enrich_target']

# Variabili Papermill (Default)
perc_water = 0.15
arricch_max = 45.553
arricch_min = 5.0
T_water = 350.0
T_resto = 600.0
run_id = "default"

# --- 2. MATERIALI ---
def densita_mix_dinamica(T_K, P_Pa=101325.0, frac_mass_H2O=0.15):
    rho_H2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water') / 1000  
    rho_D2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'HeavyWater') / 1000
    return 1 / (frac_mass_H2O / rho_H2O + (1 - frac_mass_H2O) / rho_D2O)

cladding = openmc.Material(name='cladding')
cladding.temperature = T_resto
cladding.add_element('Zr', 1.0) 
cladding.set_density('g/cm3', 6.49)

water = openmc.Material(name='water')
water.temperature = T_water
water.add_nuclide('H1', perc_water)
water.add_nuclide('H2', 1 - perc_water)
water.add_element('O', 1.0)
water.set_density('g/cm3', densita_mix_dinamica(T_water, frac_mass_H2O=perc_water)) 
water.add_s_alpha_beta('c_D_in_D2O') 
water.add_s_alpha_beta('c_H_in_H2O') 

reflector = openmc.Material(name='reflector')
reflector.temperature = T_resto
reflector.add_element('C', 1.0)
reflector.set_density('g/cm3', 1.75) 

void_air = openmc.Material(name='void_air')
void_air.set_density('g/cm3', 1e-10) 
void_air.add_nuclide('N14', 1)

mat_target = openmc.Material(name='mat_target')
mat_target.temperature = T_resto
mat_target.add_nuclide('U235', enrich_target)
mat_target.add_nuclide('U238', 1.0 - enrich_target)
mat_target.add_nuclide('O16', 2.0)
mat_target.set_density('g/cm3', 10.96)
mat_target.volume = np.pi * (R_TARGET**2) * H_TARGET

# Fuel Rings
fuel_mats = []
for i in range(int(np.ceil(R_CORE / PITCH)) + 1):
    e_i = (arricch_max/100.0) - ((arricch_max - arricch_min)/100.0) * (i / (int(np.ceil(R_CORE / PITCH))))
    f = openmc.Material(name=f'fuel_ring_{i}')
    f.temperature = T_resto
    f.add_nuclide('U235', e_i)
    f.add_nuclide('U238', 1.0 - e_i)
    f.add_nuclide('O16', 2.0)
    f.set_density('g/cm3', 10.96)
    f.volume = 1.0 # Verrà calcolato correttamente dall'operatore se necessario
    f.depletable = True
    fuel_mats.append(f)

materials = openmc.Materials(fuel_mats + [cladding, water, reflector, void_air, mat_target])

# --- 3. GEOMETRIA ---

# Piani Z
z_bot_ref   = openmc.ZPlane(z0=-REF_BOT, boundary_type='vacuum')
z_bot_core  = openmc.ZPlane(z0=0.0)
z_top_core  = openmc.ZPlane(z0=H_CORE)
z_top_world = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')
z_mid = H_CORE - P_TARGET

# Sostituzione delle funzioni deprecate con le classi moderne
# Nota: HexagonalPrism restituisce una regione, non una superficie singola
reg_fuel = openmc.model.HexagonalPrism(edge_length=R_FUEL, orientation='x')
reg_clad = openmc.model.HexagonalPrism(edge_length=R_FUEL + CLAD_THICK, orientation='x')

# 1. Creazione degli Universi per i Pin
fuel_universes = []
for f in fuel_mats:
    c_f = openmc.Cell(fill=f, region=-reg_fuel)
    c_c = openmc.Cell(fill=cladding, region=-reg_clad & +reg_fuel)
    c_w = openmc.Cell(fill=water, region=+reg_clad)
    fuel_universes.append(openmc.Universe(cells=[c_f, c_c, c_w]))

# 2. Costruzione della lista di liste per il HexLattice
rings_universes = []
for i in range(len(fuel_universes)):
    if i == 0:
        rings_universes.append([fuel_universes[0]])
    else:
        rings_universes.append([fuel_universes[i]] * (6 * i))

lattice = openmc.HexLattice(name='core_hex_lattice')
lattice.center = (0.0, 0.0)
lattice.pitch = (PITCH,)
lattice.outer = openmc.Universe(cells=[openmc.Cell(fill=water)])
lattice.universes = rings_universes[::-1]
lattice.orientation = 'x'

# 3. Definizione dei prismi per le regioni macroscopiche
num_rings = len(fuel_universes)
hex_core_edge = num_rings * PITCH
hex_ref_edge  = hex_core_edge + REF_SIDE
n_rings_removed = int(np.ceil(R_PIPE / PITCH))
edge_pipe = (n_rings_removed - 0.5) * PITCH if n_rings_removed > 0 else 0.1

reg_prism_core = openmc.model.HexagonalPrism(edge_length=hex_core_edge, orientation='x')
reg_prism_ref  = openmc.model.HexagonalPrism(edge_length=hex_ref_edge, orientation='x', boundary_type='vacuum')
reg_prism_pipe = openmc.model.HexagonalPrism(edge_length=edge_pipe, orientation='x')

# 4. Celle Finali
c_main_core = openmc.Cell(name='main_core', fill=lattice, 
                          region=-reg_prism_core & +reg_prism_pipe & +z_bot_core & -z_top_core)

# Target e Gap nel condotto centrale

region_void_gap = (-reg_prism_pipe & +z_bot_core & -z_top_core)
c_void_gap = openmc.Cell(name='void_gap', fill=void_air, region=region_void_gap)

# Riflettore laterale e inferiore
c_ref_side = openmc.Cell(name='ref_side', fill=reflector, 
                         region=-reg_prism_ref & +reg_prism_core & +z_bot_core & -z_top_core)
c_ref_bot = openmc.Cell(name='ref_bot', fill=reflector, 
                        region=-reg_prism_ref & +z_bot_ref & -z_bot_core)

# Cielo (Top void)
c_top_void = openmc.Cell(name='top_void', fill=void_air, 
                         region=-reg_prism_ref & +z_top_core & -z_top_world)

# 5. Assemblaggio Geometria
geometry = openmc.Geometry([c_main_core, c_void_gap, c_ref_side, c_ref_bot, c_top_void])



# --- 4. SETTINGS E SORGENTE PUNTIFORME ---
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = batches
settings.particles = particles
settings.max_collisions = 10000
settings.max_lost_particles = 50
settings.temperature = {'method': 'interpolation'}
settings.photon_transport = False
settings.create_fission_neutrons = True

# Path dei file salvati
energy_file = os.path.join(root_dir, "target", "photoneutrons_energy.txt")
polar_file = os.path.join(root_dir, "target", "photoneutrons_polar.txt")

if not (os.path.exists(energy_file) and os.path.exists(polar_file)):
    raise FileNotFoundError("File delle distribuzioni mancanti nella cartella 'target'.")

print("Lettura distribuzioni e costruzione della sorgente puntiforme...")

# 1. Distribuzione Energetica
data_energy = np.loadtxt(energy_file, skiprows=1)
E_mid = data_energy[:, 0]
p_E = data_energy[:, 1]

# Normalizzazione PDF energia (np.trapz aggiornato a np.trapezoid)
p_E = p_E / np.trapezoid(p_E, x=E_mid)
energy_dist = openmc.stats.Tabular(E_mid, p_E, interpolation='linear-linear')

# 2. Distribuzione Angolare
data_polar = np.loadtxt(polar_file, skiprows=1)
theta = data_polar[:, 0]
p_theta = data_polar[:, 1]

# Conversione in mu = cos(theta) e ordinamento crescente (da -1 a 1) per OpenMC
mu = np.cos(theta)
sort_idx = np.argsort(mu)
mu_sorted = mu[sort_idx]
p_mu_sorted = p_theta[sort_idx]

# Normalizzazione PDF angolare su mu (np.trapz aggiornato a np.trapezoid)
p_mu_sorted = p_mu_sorted / np.trapezoid(p_mu_sorted, x=mu_sorted)

mu_dist = openmc.stats.Tabular(mu_sorted, p_mu_sorted, interpolation='linear-linear')
phi_dist = openmc.stats.Uniform(0.0, 2 * np.pi)
angle_dist = openmc.stats.PolarAzimuthal(mu=mu_dist, phi=phi_dist)

# 3. Creazione della Sorgente
source = openmc.IndependentSource()
source.particle = 'neutron'

# Posizione esatta al centro del vecchio target (z_mid_tgt calcolato in Geometria)
source.space = openmc.stats.Point((0.0, 0.0, z_mid))
source.angle = angle_dist
source.energy = energy_dist

settings.source = source
print("Generazione completata: Sorgente puntiforme centrale configurata.")

# ------- Tally ----------
tallies = openmc.Tallies()
fission_tot = openmc.Tally(name='fission_tot')
fission_tot.scores = ['kappa-fission', 'nu-fission','absorption']
tallies.append(fission_tot)

vacuum_surfaces = [surf for surf in geometry.get_all_surfaces().values() if surf.boundary_type == 'vacuum']
surface_filter = openmc.SurfaceFilter(vacuum_surfaces)
tally_leakage = openmc.Tally(name='leakage_tot')
tally_leakage.filters = [surface_filter]
tally_leakage.scores = ['current']
tallies.append(tally_leakage)

# --- 5. GESTIONE DIRECTORY ---
os.environ["OMP_NUM_THREADS"] = "40"
nome_base = f"{batches}b_{particles}p"
nome_cartella = nome_base
contatore = 2
while os.path.exists(nome_cartella):
    nome_cartella = f"{nome_base}_{contatore}"
    contatore += 1

os.makedirs(nome_cartella)
print(f"Directory di run: {nome_cartella}")

# Spostamento nella cartella di run
os.chdir(nome_cartella)

# Esportazione XML nella cartella di run
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# --- 6. DEPLETION ---
warnings.filterwarnings("ignore")
model = openmc.Model(geometry=geometry, materials=materials, settings=settings)

# Dati nucleari
path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"
path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
base_path = path_arco if os.path.exists(path_arco) else path_pc
openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"
chain_file = f"{base_path}/chain_endfb71_pwr.xml"

operator = openmc.deplete.CoupledOperator(model, chain_file, normalization_mode="source-rate")

# Rateo sorgente
yield_fn = 5.87905e-03 
S_rate = S_val * yield_fn

# --- FASE ON (Totale: 200 giorni) ---
# Transitorio veleni: traccia la curva di inserzione reattività negativa di Xe e Sm nei primi 20 giorni
giorni_on_transitorio = [0.1, 0.4, 0.5, 1.0, 3.0, 5.0, 10.0] 

# Burn-up profondo: traccia la pendenza lineare del consumo per i restanti 180 giorni
giorni_on_burnup = [30.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0] 

giorni_on = giorni_on_transitorio + giorni_on_burnup

# --- FASE OFF (Totale: 50 giorni) ---
# Picco dello Xeno: altissima risoluzione per le prime 120 ore (5 giorni)
# Gli step più fitti (0.05, 0.15, 0.2) si concentrano attorno alle 8-12 ore (picco massimo)
giorni_off_picco_xe = [0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 3.0] 

# Stabilizzazione finale: decadimento residuo e saturazione finale del Samario nei restanti 45 giorni
giorni_off_coda = [10.0, 15.0, 20.0] 

giorni_off = giorni_off_picco_xe + giorni_off_coda

# --- ASSEGNAZIONE ---
time_steps = giorni_on + giorni_off
dummy_S_rate = 1e-15 
source_rates = [S_rate] * len(giorni_on) + [dummy_S_rate] * len(giorni_off)

integrator = openmc.deplete.PredictorIntegrator(operator, time_steps, source_rates=source_rates, timestep_units='d')
integrator.integrate()
