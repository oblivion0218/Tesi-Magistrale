import os
import glob
import openmc
import numpy as np

# --- Pulizia file pregressi ---
for f in glob.glob("summary.h5") + glob.glob("*.xml") + glob.glob("*.out") + glob.glob("*.png"):
    if os.path.exists(f):
        os.remove(f)

# --- 1. SETUP DATI NUCLEARI ---
path_pc = "/home/bossi_ricky/openmc_data/mcnp_endfb71"
path_arco = "/raid1/users/rbossi/MC/Magistrale/openmc_data/mcnp_endfb71"

if os.path.exists(path_arco):
    base_path = path_arco
elif os.path.exists(path_pc):
    base_path = path_pc
else:
    raise FileNotFoundError("Directory dei dati nucleari non trovata.")

openmc.config['cross_sections'] = f"{base_path}/cross_sections.xml"

# --- 2. LETTURA PARAMETRI ---
def load_parameters(filename='parametri.txt'):
    params = {}
    context = {'np': np} 
    with open(filename, 'r') as f:
        code = f.read()
        exec(code, context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

p = load_parameters("../parametri.txt")

R_TARGET  = p['R_TARGET']
H_CORE    = p['H_CORE']
P_TARGET  = p['P_TARGET']
H_TARGET  = p['H_TARGET']
batches   = p['batches_fix']
particles = p['particles_fix']
enrich_target = p['enrich_target']

# --- 3. MATERIALI ---
mat_target = openmc.Material(name='mat_target')
mat_target.temperature = 600.0
mat_target.add_nuclide('U235', enrich_target)
mat_target.add_nuclide('U238', 1.0 - enrich_target)
mat_target.add_nuclide('O16', 2.0)
mat_target.set_density('g/cm3', 10.96)

void_air = openmc.Material(name='void_air')
void_air.temperature = 300.0
void_air.set_density('g/cm3', 1e-10) 
void_air.add_nuclide('N14', 1)

materials = openmc.Materials([mat_target, void_air])
materials.export_to_xml()

# --- 4. GEOMETRIA E SUPERFICI ---
# Target Centrato nell'origine

z_mid = H_CORE - P_TARGET
z_tgt_bot = openmc.ZPlane(z0=z_mid - (H_TARGET / 2.0))
z_tgt_top = openmc.ZPlane(z0=z_mid + (H_TARGET / 2.0))
cyl_target = openmc.ZCylinder(r=R_TARGET)

# Sfera per ispezione isotropia e Boundary mondo
R_SPHERE = max(R_TARGET, H_TARGET / 2.0) + 2.0
R_SPHERE = max(R_TARGET, H_TARGET / 2.0) + 2.0
# La sfera deve avere lo stesso centro (0, 0, z_mid) del target
sphere_det = openmc.Sphere(x0=0, y0=0, z0=z_mid, r=R_SPHERE)
world_sphere = openmc.Sphere(x0=0, y0=0, z0=z_mid, r=R_SPHERE + 5.0, boundary_type='vacuum')

# Definizione Celle
region_target = -cyl_target & +z_tgt_bot & -z_tgt_top
c_target = openmc.Cell(name='target', fill=mat_target, region=region_target)

# Volume di controllo compreso tra il target e la superficie sferica
region_void_in = -sphere_det & ~region_target
c_void_in = openmc.Cell(name='void_in', fill=void_air, region=region_void_in)

# Volume esterno alla sfera fino al confine del mondo
region_void_out = +sphere_det & -world_sphere
c_void_out = openmc.Cell(name='void_out', fill=void_air, region=region_void_out)

geometry = openmc.Geometry([c_target, c_void_in, c_void_out])
geometry.export_to_xml()

# --- 5. SORGENTE E SETTINGS ---
data = np.loadtxt("synchrotron_1_37_MeV.txt")
energy_midpoints, pdf_array = data[:,0], data[:,1]
energy_dist = openmc.stats.Tabular(energy_midpoints, pdf_array, interpolation='linear-linear')

source = openmc.IndependentSource()
# Posizionata al di sopra del target, sull'asse z
source.space = openmc.stats.Point((0, 0, z_mid + (H_TARGET / 2.0) + 2.0))
source.angle = openmc.stats.Monodirectional((0, 0, -1))
source.energy = energy_dist
source.particle = 'photon'

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = batches
settings.particles = particles
settings.max_lost_particles = 50
settings.cutoff = {'weight': 1.0e-5, 'energy_photon': 1.0e6}
settings.temperature = {'method': 'interpolation'}
settings.source = source
settings.photon_transport = True
settings.photonuclear_physics = True
settings.create_fission_neutrons = True

settings.surf_source_write = {
    'surface_ids': [cyl_target.id, z_tgt_bot.id, z_tgt_top.id],
    'max_particles': 10000000  # Numero massimo di particelle salvabili nel file h5
}

settings.export_to_xml()

# --- 6. TALLIES ---
tallies = openmc.Tallies()

# Filtri
surface_filter_target = openmc.SurfaceFilter([cyl_target, z_tgt_bot, z_tgt_top])
surface_filter_sphere = openmc.SurfaceFilter([sphere_det])
particle_filter = openmc.ParticleFilter(['neutron'])

energy_bins = np.logspace(2, 8, 75)
energy_filter = openmc.EnergyFilter(energy_bins)

polar_bins = np.linspace(0, np.pi, 180)
polar_filter = openmc.PolarFilter(polar_bins)

# 1. Tally Spettro Energetico (Corrente uscente dalle facce del cilindro)
tally_energy_tgt = openmc.Tally(name='spettro_energia_neutroni_target')
tally_energy_tgt.filters = [surface_filter_target, particle_filter, energy_filter]
tally_energy_tgt.scores = ['current']
tallies.append(tally_energy_tgt)

# 2. Tally Spettro Angolare (Corrente uscente dalle facce del cilindro)
tally_polar_tgt = openmc.Tally(name='spettro_angolare_neutroni_target')
tally_polar_tgt.filters = [surface_filter_target, particle_filter, polar_filter]
tally_polar_tgt.scores = ['current']
tallies.append(tally_polar_tgt)

# 3. Tally Distribuzione Isotropia (Corrente incrociante la Sfera esterna)
tally_isotropy = openmc.Tally(name='distribuzione_isotropia_sfera')
tally_isotropy.filters = [surface_filter_sphere, particle_filter, polar_filter]
tally_isotropy.scores = ['current']
tallies.append(tally_isotropy)

tallies.export_to_xml()

# --- 7. ESECUZIONE ---
openmc.run(output=True, geometry_debug=False)

