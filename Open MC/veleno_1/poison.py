import os
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
z_bot_ref = openmc.ZPlane(z0=-REF_BOT, boundary_type='vacuum')
z_bot_core = openmc.ZPlane(z0=0.0)
z_top_core = openmc.ZPlane(z0=H_CORE)
z_top_world = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')
z_mid_tgt = H_CORE - P_TARGET
z_tgt_bot = openmc.ZPlane(z0=z_mid_tgt - (H_TARGET / 2.0))
z_tgt_top = openmc.ZPlane(z0=z_mid_tgt + (H_TARGET / 2.0))

cyl_target = openmc.ZCylinder(r=R_TARGET)
hex_fuel_prism = openmc.model.hexagonal_prism(edge_length=R_FUEL, orientation='x')
hex_clad_prism = openmc.model.hexagonal_prism(edge_length=R_FUEL + CLAD_THICK, orientation='x')

# Lattice
fuel_universes = []
for f in fuel_mats:
    c_f = openmc.Cell(fill=f, region=hex_fuel_prism)
    c_c = openmc.Cell(fill=cladding, region=hex_clad_prism & ~hex_fuel_prism)
    c_w = openmc.Cell(fill=water, region=~hex_clad_prism)
    fuel_universes.append(openmc.Universe(cells=[c_f, c_c, c_w]))

lattice = openmc.HexLattice()
lattice.center = (0, 0)
lattice.pitch = (PITCH,)
lattice.outer = openmc.Universe(cells=[openmc.Cell(fill=water)])
lattice.universes = [fuel_universes[::-1]] 

# Celle Finali
prism_core = openmc.model.hexagonal_prism(edge_length=(len(fuel_mats)) * PITCH, orientation='x')
n_rings_removed = int(np.ceil(R_PIPE / PITCH))
hex_pipe_prism = openmc.model.hexagonal_prism(edge_length=(n_rings_removed - 0.5) * PITCH, orientation='x')

c_main_core = openmc.Cell(fill=lattice, region=prism_core & ~hex_pipe_prism & +z_bot_core & -z_top_core)
c_target = openmc.Cell(fill=mat_target, region=hex_pipe_prism & -cyl_target & +z_tgt_bot & -z_tgt_top)
c_void_gap = openmc.Cell(fill=void_air, region=hex_pipe_prism & ~( -cyl_target & +z_tgt_bot & -z_tgt_top) & +z_bot_core & -z_top_core)

prism_ref = openmc.model.hexagonal_prism(edge_length=(len(fuel_mats)) * PITCH + REF_SIDE, orientation='x', boundary_type='vacuum')
c_ref_side = openmc.Cell(fill=reflector, region=prism_ref & ~prism_core & +z_bot_core & -z_top_core)
c_ref_bot = openmc.Cell(fill=reflector, region=prism_ref & +z_bot_ref & -z_bot_core)
c_top_void = openmc.Cell(fill=void_air, region=prism_ref & +z_top_core & -z_top_world)

geometry = openmc.Geometry([c_main_core, c_target, c_void_gap, c_ref_side, c_ref_bot, c_top_void])

# --- 4. SETTINGS ---
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = batches
settings.particles = particles
settings.surf_source_read = {'path': 'surface_source.h5'} # Link simbolico
settings.photon_transport = False
settings.create_fission_neutrons = True

# --- 5. GESTIONE DIRECTORY E LINK SIMBOLICO ---
os.environ["OMP_NUM_THREADS"] = "40"
nome_base = f"{batches}b_{particles}p"
nome_cartella = nome_base
contatore = 2
while os.path.exists(nome_cartella):
    nome_cartella = f"{nome_base}_{contatore}"
    contatore += 1

os.makedirs(nome_cartella)
print(f"Directory di run: {nome_cartella}")

# Creazione link simbolico alla sorgente presente nella cartella 'target'
path_sorgente_originale = os.path.join(root_dir, "target", "surface_source.h5")
path_link_destinazione = os.path.join(root_dir, nome_cartella, "surface_source.h5")

if not os.path.exists(path_sorgente_originale):
    raise FileNotFoundError(f"Sorgente mancante in: {path_sorgente_originale}")

if os.path.lexists(path_link_destinazione): os.remove(path_link_destinazione)
os.symlink(path_sorgente_originale, path_link_destinazione)

# Spostamento nella cartella di run
os.chdir(nome_cartella)

# Esportazione XML nella cartella di run
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
openmc.Tallies().export_to_xml()

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
yield_fn = 0.006233780865925537 
S_rate = S_val * yield_fn

giorni_on = [10.0] * 10
giorni_off = [0.1, 0.4, 0.5, 1.0, 3.0, 10.0, 30.0, 55.0]
time_steps = giorni_on + giorni_off
source_rates = [S_rate]*len(giorni_on) + [0.0]*len(giorni_off)

integrator = openmc.deplete.PredictorIntegrator(operator, time_steps, source_rates=source_rates, timestep_units='d')
integrator.integrate()
