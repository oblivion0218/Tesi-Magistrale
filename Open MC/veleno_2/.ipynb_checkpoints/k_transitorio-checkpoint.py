#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import openmc
import numpy as np
from IPython.display import Image, display
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# pulizia per sicurezza

#os.system("rm *.h5")
os.system("rm *.xml")
os.system("rm *.out")
os.system("rm *.png")


# In[3]:


param_file = 'parametri.txt'
moltiplicatore = 4.0
pressione = 1.0
water_perc = 1.0
enrich_min = 1
enrich_max = 1
iter = 1
is_first_run = True


# VARIABILI IMPORTANTI

# In[4]:


def load_parameters(filename='parametri.txt'):
    """
    Legge i parametri dal file txt e li restituisce come un dizionario.
    """
    params = {}
    # Forniamo np al contesto di esecuzione per gestire np.sqrt()
    context = {'np': np} 

    with open(filename, 'r') as f:
        code = f.read()
        exec(code, context, params)

    # Rimuoviamo 'np' e altri built-in dal dizionario finale
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

p = load_parameters(param_file)

# --- Parametri Geometrici ---
R_CORE    = p['R_CORE'] * moltiplicatore
REF_SIDE  = p['REF_SIDE']
REF_BOT   = p['REF_BOT']
H_CORE    = p['H_CORE']

# --- Parametri Logici ---
R_PIPE    = p['R_PIPE']

# --- Parametri Pin/Lattice ---
R_FUEL    = p['R_FUEL']
CLAD_THICK= p['CLAD_THICK']
R_WATER_THICK = p['R_WATER_THICK']
R_PIN     = p['R_PIN']
PITCH     = p['PITCH'] * moltiplicatore

# --- Parametri Target ---
R_TARGET  = p['R_TARGET']
H_TARGET  = p['H_TARGET']
P_TARGET  = p['P_TARGET']

# --- Sorgente e Simulazione ---
S             = p['S']

bin_ris       = p['bin_ris']

batches       = p['batches_auto']
particles     = p['particles_auto']
inactive      = p['inactive_auto']

enrich_target = p['enrich_target']

mix = water_perc
pressione_funzionamento = p['Pressione_funzionamento'] * pressione

T_water_HZP = p['T_water_HZP']
T_water_HFP = p['T_water_HFP']
T_fuel_HFP = p['T_fuel_HFP']
T_fuel_HZP = p['T_fuel_HZP']


# MATERIALI

# In[5]:


# materiale di rivestimento
cladding = openmc.Material (name = 'cladding')
cladding.add_element('Zr', 1.0) 
cladding.set_density('g/cm3', 6.49)

# materiale di moderatore
water = openmc.Material (name = 'water')
water.add_nuclide('H1', mix)
water.add_nuclide('H2', 1-mix)
water.add_element('O', 1.0)
water.set_density('g/cm3', 1.10)
water.add_s_alpha_beta('c_D_in_D2O') # aggiungo la tabella di scattering termico
water.add_s_alpha_beta('c_H_in_H2O') # aggiungo la tabella di scattering termico
# n.b.  le sezioni d'urto veloci sono già incluse di default

# aggiungo un riflettore 
reflector = openmc.Material(name='reflector')
reflector.add_element('C',1.0)
reflector.set_density('g/cm3', 1.75) 

# voglio una sorgente puntiforme di neutroni in un punto specifico
marker_mat = openmc.Material(name='marker')
marker_mat.set_density('g/cm3', 1e-10) # Densità quasi nulla per non disturbare la fisica
marker_mat.add_nuclide('He4', 1.0) # elio, non interagisce praticamente

void_air = openmc.Material(name='void_air')
void_air.set_density('g/cm3', 1e-10) # Densità
void_air.add_nuclide('N14', 1 )

mat_target = openmc.Material(name='mat_target')
mat_target.add_nuclide('U235', enrich_target)
mat_target.add_nuclide('U238', 1.0 - enrich_target)
mat_target.add_nuclide('O16', 2.0)
mat_target.set_density('g/cm3', 10.96)


# GEOMETRIA

# In[6]:


# --- 0. PIANI Z (ASSOLUTI) ---
z_bot_ref = openmc.ZPlane(z0=-REF_BOT, boundary_type='vacuum')
z_bot_core = openmc.ZPlane(z0=0.0)
z_top_core = openmc.ZPlane(z0=H_CORE)
z_top_world = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')

# Piani Z Target
z_mid = H_CORE - P_TARGET
z_tgt_bot = openmc.ZPlane(z0=z_mid - (H_TARGET / 2.0))
z_tgt_top = openmc.ZPlane(z0=z_mid + (H_TARGET / 2.0))

# Superficie Sorgente
sphere_src = openmc.Sphere(x0=0, y0=0, z0=H_CORE + 2, r=0.5)

# --- 1. CALCOLO DIMENSIONI TAGLIO CENTRALE ---
# Calcoliamo quanto deve essere grande il buco esagonale centrale
# Rimuoviamo gli anelli che cadono dentro R_PIPE
n_rings_removed = int(np.ceil(R_PIPE / PITCH))
# Il lato del prisma che useremo per "bucare" il lattice:
EDGE_PIPE_LARGE = (n_rings_removed - 0.5) * PITCH
if EDGE_PIPE_LARGE == 0: EDGE_PIPE_LARGE = PITCH # Sicurezza minima

# --- 2. FORME GEOMETRICHE BASE ---

EDGE_FUEL = R_FUEL
EDGE_CLAD = R_FUEL + CLAD_THICK

# A. Prismi PIN
hex_fuel_prism = openmc.model.get_hexagonal_prism(edge_length=EDGE_FUEL, orientation='x')
hex_clad_prism = openmc.model.get_hexagonal_prism(edge_length=EDGE_CLAD, orientation='x')

# B. Prisma "TAGLIO" CENTRALE (Il buco grande nel reticolo)
hex_pipe_prism = openmc.model.get_hexagonal_prism(edge_length=EDGE_PIPE_LARGE, orientation='x')

# C. Cilindro Target (Unico centrale)
cyl_target = openmc.ZCylinder(r=R_TARGET)

# --- 3. UNIVERSI ---
fuel_mats = []
fuel_universes = []

num_rings = int(np.ceil(R_CORE / PITCH)) + 1

for i in range(num_rings):
    # Interpolazione lineare arricchimento
    if num_rings > 1:
        e_i = enrich_max - (enrich_max - enrich_min) * (i / (num_rings - 1))
    else:
        e_i = enrich_max

    # Materiale
    f = openmc.Material(name=f'fuel_ring_{i}')
    f.add_nuclide('U235', e_i)
    f.add_nuclide('U238', 1.0 - e_i)
    f.add_nuclide('O16', 2.0)
    f.set_density('g/cm3', 10.96)
    fuel_mats.append(f)

    # Cella e Universo
    c_f = openmc.Cell(fill=f, region=hex_fuel_prism)
    c_c = openmc.Cell(fill=cladding, region=hex_clad_prism & ~hex_fuel_prism)
    c_w = openmc.Cell(fill=water, region=~hex_clad_prism)
    fuel_universes.append(openmc.Universe(cells=[c_f, c_c, c_w]))

# Esportazione xml iniziale
materials = openmc.Materials(fuel_mats + [cladding, water, reflector, marker_mat, void_air, mat_target])

path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"
path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
base_path = path_arco if os.path.exists(path_arco) else path_pc
openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"

materials.export_to_xml()


# C. Universo ACQUA ESTERNA (Riflettore/Gap)
c_water_full = openmc.Cell(fill=water)
u_water = openmc.Universe(cells=[c_water_full])


# --- 4. LATTICE (SOLO COMBUSTIBILE) ---
# Creiamo un lattice uniforme pieno SOLO di pin.
# Il buco centrale lo creiamo geometricamente nello step successivo, non qui.

hex_core_edge = num_rings * PITCH
hex_ref_edge = hex_core_edge + REF_SIDE

lattice_universes = []
for i in range(num_rings):
    if i == 0:
        lattice_universes.append([fuel_universes[i]])
    else:
        lattice_universes.append([fuel_universes[i]] * (6 * i))

lattice = openmc.HexLattice(name='core_hex_lattice')
lattice.center = (0.0, 0.0)
lattice.pitch = (PITCH,)
lattice.outer = u_water
lattice.universes = lattice_universes[::-1] # Inversione fondamentale: Esterno -> Interno
lattice.orientation = 'x'

# --- 5. ASSEMBLAGGIO FINALE (GEOMETRIA DIRETTA) ---

prism_core = openmc.model.get_hexagonal_prism(edge_length=hex_core_edge, orientation='x')
prism_reflector = openmc.model.get_hexagonal_prism(edge_length=hex_ref_edge, orientation='x', boundary_type='vacuum')

# A. Cella Core (Lattice CON BUCO)
# ---------------------------------------------------------
# Prendiamo il prisma grande del core, e sottraiamo (~) il prisma del canale centrale.
# In questo modo il lattice viene "tagliato" via dal centro.
region_core = prism_core & ~hex_pipe_prism & +z_bot_core & -z_top_core
c_main_core = openmc.Cell(fill=lattice, region=region_core)


# B. Canale Centrale (Target + Vuoto) - DEFINIZIONE DIRETTA
# ---------------------------------------------------------
# 1. Definiamo il volume TOTALE del canale esagonale (da 0 a H_CORE)
region_central_hex_total = hex_pipe_prism & +z_bot_core & -z_top_core

# 2. Definiamo la regione del TARGET SOLIDO (Cilindro limitato in Z)
# Nota: Intersechiamo con region_central_hex_total per sicurezza, ma basterebbe il cilindro.
region_target_solid = region_central_hex_total & -cyl_target & +z_tgt_bot & -z_tgt_top

# 3. Definiamo la regione del VUOTO (Tutto il canale MENO il target)
# Usiamo l'operatore ~ (NOT) sulla regione del target.
region_void_gap = region_central_hex_total & ~region_target_solid

# Creazione delle celle
c_target = openmc.Cell(fill=mat_target, region=region_target_solid)
c_void_gap = openmc.Cell(fill=void_air, region=region_void_gap) 

# C. Riflettori (Rimangono uguali)
region_ref_side = prism_reflector & ~prism_core & +z_bot_core & -z_top_core
c_ref_side = openmc.Cell(fill=reflector, region=region_ref_side)

region_ref_bot = prism_reflector & +z_bot_ref & -z_bot_core
c_ref_bot = openmc.Cell(fill=reflector, region=region_ref_bot)

# D. Top World & Source (Rimangono uguali)
c_source_marker = openmc.Cell(fill=marker_mat, region=-sphere_src)
region_top = prism_reflector & +z_top_core & -z_top_world & +sphere_src
c_top_void = openmc.Cell(fill=void_air, region=region_top)

all_cells = [c_main_core, c_target, c_void_gap, c_ref_side, c_ref_bot, c_source_marker, c_top_void]
geometry = openmc.Geometry(all_cells)
geometry.export_to_xml()
print("Geometria generata correttamente.")

def count_fuel_pins(num_rings, pitch, r_pipe, r_core):
    total_pins = 0
    # Ring 0 è il centro
    for i in range(num_rings):
        ring_radius = i * pitch
        # Se l'anello è fuori dalla pipe centrale e dentro il raggio del core
        if ring_radius >= r_pipe and ring_radius <= r_core:
            if i == 0:
                total_pins += 1
            else:
                total_pins += (6 * i)
    return total_pins

n_barre = count_fuel_pins(num_rings, PITCH, R_PIPE, R_CORE)
print(f"Numero di barre di combustibile effettive: {n_barre} \n\n")


# SETTING EIGENVALUED

# In[8]:


settings_k = openmc.Settings()
settings_k.run_mode = 'eigenvalue'
settings_k.inactive = inactive

# Definiamo un box che racchiude l'intero core
lower_left = [-R_CORE, -R_CORE, 0.0]
upper_right = [R_CORE, R_CORE, H_CORE]

# Creazione di una distribuzione spaziale uniforme confinata ai materiali fessili
uniform_dist = openmc.stats.Box(lower_left, upper_right, only_fissionable=True)

settings_k.source = openmc.IndependentSource(space=uniform_dist)

# Override della frazione di rigetto
settings_k.source_rejection_fraction = 0.001 
settings_k.temperature = {'method': 'interpolation'}

# Esportazione
settings_k.export_to_xml()


# In[9]:


from CoolProp.CoolProp import PropsSI

def densita_mix_dinamica(T_K, P_Pa=pressione_funzionamento, frac_mass_H2O=mix):
    # Calcolo in kg/m^3 e conversione in g/cm^3
    rho_H2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water') / 1000     # D sta per densità
    rho_D2O = PropsSI('D', 'T', T_K, 'P', P_Pa, 'HeavyWater') / 1000

    # Calcolo miscela assumendo volumi ideali additivi
    rho_mix = 1 / (frac_mass_H2O / rho_H2O + (1 - frac_mass_H2O) / rho_D2O) # Formula di additività dei volumi specifici
    return rho_mix


# In[11]:


import openmc
import openmc.deplete
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# 1. SETUP E CARICAMENTO RISULTATI DEPLETION
# ==============================================================================
# Percorso alla cartella dove poison.py ha salvato i file .h5
PATH_DEPLETION = "25b_100000p_2" 
results_file = os.path.join(PATH_DEPLETION, "depletion_results.h5")

if not os.path.exists(results_file):
    raise FileNotFoundError(f"File {results_file} non trovato. Assicurati che il percorso sia corretto.")

results = openmc.deplete.Results(results_file)
times_days = results.get_times(time_units="d")

# --- DEFINIZIONE STATISTICHE DINAMICHE ---
# Usa i parametri base già definiti altrove per i run normali
batches_norm = batches  
particles_norm = particles

# Definisci qui la statistica maggiorata (MODIFICA A PIACIMENTO)
batches_high = 200
particles_high = 1000000

# ==============================================================================
# 2. INIZIALIZZAZIONE FILE DI OUTPUT
# ==============================================================================
now = datetime.now()
PATH  = "25b_1000000p_2"
output_file = "k_eff_transitorio_" + PATH + ".txt"

if is_first_run:
    timestamp = now.strftime("\n\n------ ANALISI K_EFF CONGELATO DEL %d/%m/%Y ALLE ORE %H:%M ------\n")
    with open(output_file, "a") as f_out:
        f_out.write(timestamp)
        f_out.write(f"Settings base: {batches_norm} b, {particles_norm} p | Settings high: {batches_high} b, {particles_high} p\n")
        f_out.write("===============================================================\n")
        f_out.write(f"{'Step':<6} {'Tempo [gg]':<15} {'K_EFF':<15} {'STD_DEV':<15} {'STATISTICA':<15}\n")

k_eff_series = []
k_eff_errors = []

# La geometria non cambia mai, la esportiamo una volta sola
geometry.export_to_xml()

# ==============================================================================
# 3. CICLO DI CALCOLO AUTOVALORE PER OGNI STEP
# ==============================================================================
for step in range(len(times_days)):
    openmc.reset_auto_ids()
    t = times_days[step]
    
    # --- LOGICA STATISTICA DINAMICA ---
    # np.isclose evita errori di arrotondamento floating-point (es. 199.99999)
    if np.isclose(t, 200.0, atol=1e-3) or np.isclose(t, 205.0, atol=1e-3):
        curr_batches = batches_high
        curr_particles = particles_high
        stat_label = "HIGH"
        print(f"\n[Post-Processing] Step {step} | Tempo: {t:.2f} gg ---> ALTA STATISTICA")
    else:
        curr_batches = batches_norm
        curr_particles = particles_norm
        stat_label = "NORM"
        print(f"\n[Post-Processing] Step {step} | Tempo: {t:.2f} gg")

    # Aggiorna ed esporta i settings per il run corrente
    settings_k.batches = curr_batches
    settings_k.particles = curr_particles
    settings_k.export_to_xml()
    
   # A. Esportazione materiali burnati
    mats = results.export_to_materials(step)
    
    # FIX: Ripristino delle tabelle S(alpha, beta) perse dall'operatore di depletion
    for mat in mats:
        if mat.name == 'water':
            mat.add_s_alpha_beta('c_D_in_D2O')
            mat.add_s_alpha_beta('c_H_in_H2O')
            
    mats.cross_sections = f"{base_path}/cross_sections.xml"
    mats.export_to_xml()
    
    # B. Esecuzione OpenMC
    openmc.run(output=False)
    
    # C. Lettura dello Statepoint (il nome dipende dal numero di batches usato!)
    sp_file = f"statepoint.{curr_batches}.h5"
    if os.path.exists(sp_file):
        with openmc.StatePoint(sp_file) as sp:
            k_val = sp.k_combined.nominal_value
            k_std = sp.k_combined.std_dev
            
            k_eff_series.append(k_val)
            k_eff_errors.append(k_std)
            
            print(f"Risultato: k = {k_val:.5f} +/- {k_std:.5f}")
            
            with open(output_file, "a") as f_out:
                f_out.write(f"{step:<6} {t:<15.4f} {k_val:<15.5f} {k_std:<15.5f} {stat_label:<15}\n")
        
        os.remove(sp_file)
    
    if os.path.exists('summary.h5'):
        os.remove('summary.h5')

