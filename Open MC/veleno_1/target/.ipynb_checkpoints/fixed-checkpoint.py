#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
import openmc
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# --- Pulizia file pregressi ---
for f in glob.glob("summary.h5") + glob.glob("*.xml") + glob.glob("*.out") + glob.glob("*.png"):
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
def load_parameters(filename='../parametri.txt'):
    params = {}
    context = {'np': np} 
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            exec(f.read(), context, params)
    return {k: v for k, v in params.items() if not k.startswith('__') and k != 'np'}

p = load_parameters()

# Variabili Papermill / Default
pressione_atm = 1.0
moltiplicatore = 4.0
perc_water = 0.15
arricch_target = 45.553 / 100.0
t_water = 350.0
t_fuel = 600.0

# Assegnazioni
Pressione_funzionamento = pressione_atm * 101325
R_CORE = p.get('R_CORE', 50.0) * moltiplicatore
H_CORE = p.get('H_CORE', 100.0)
R_TARGET = p.get('R_TARGET', 5.0)
H_TARGET = p.get('H_TARGET', 10.0)
P_TARGET = p.get('P_TARGET', 10.0)
batches = p.get('batches_fix', 50)
particles = p.get('particles_fix', 10000)

# --- 3. MATERIALI ---
mat_target = openmc.Material(name='mat_target')
mat_target.temperature = t_fuel
mat_target.add_nuclide('U235', arricch_target)
mat_target.add_nuclide('U238', 1.0 - arricch_target)
mat_target.add_nuclide('O16', 2.0)
mat_target.set_density('g/cm3', 10.96)

materials = openmc.Materials([mat_target])
materials.export_to_xml()

# --- 4. GEOMETRIA E SUPERFICI ---
# Mondo
world_cyl = openmc.ZCylinder(r=R_TARGET + 10.0, boundary_type='vacuum')
world_bot = openmc.ZPlane(z0=-10.0, boundary_type='vacuum')
world_top = openmc.ZPlane(z0=H_CORE + 5.0, boundary_type='vacuum')

# Superfici Target
z_mid = H_CORE - P_TARGET
z_tgt_bot = openmc.ZPlane(z0=z_mid - (H_TARGET / 2.0))
z_tgt_top = openmc.ZPlane(z0=z_mid + (H_TARGET / 2.0))
cyl_target = openmc.ZCylinder(r=R_TARGET)

# Celle
region_target = -cyl_target & +z_tgt_bot & -z_tgt_top
c_target = openmc.Cell(name='target', fill=mat_target, region=region_target)

region_world = -world_cyl & +world_bot & -world_top
region_void_gap = region_world & ~region_target
c_void_gap = openmc.Cell(name='void_gap', fill=None, region=region_void_gap)

geometry = openmc.Geometry([c_target, c_void_gap])
geometry.export_to_xml()

# --- 5. SORGENTE E SETTINGS ---
energy_dist = openmc.stats.Discrete([1.37e6], [1.0])
if os.path.exists("synchrotron_1_37_MeV.txt"):
    data = np.loadtxt("synchrotron_1_37_MeV.txt")
    energy_dist = openmc.stats.Tabular(data[:,0], data[:,1], interpolation='linear-linear')

source = openmc.IndependentSource()
source.space = openmc.stats.Point((0, 0, H_CORE + 2.0))
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
settings.export_to_xml()

# --- 6. TALLIES CORRETTI ---
tallies = openmc.Tallies()

# Filtro sulle superfici esterne del target per la misurazione esatta dell'emissione
surface_filter = openmc.SurfaceFilter([cyl_target, z_tgt_bot, z_tgt_top])
particle_filter = openmc.ParticleFilter(['neutron'])

energy_bins = np.logspace(2, 8, 75)
energy_filter = openmc.EnergyFilter(energy_bins)

polar_bins = np.linspace(0, np.pi, 180)
polar_filter = openmc.PolarFilter(polar_bins)

# Tally Spettro Energetico (Corrente uscente)
tally_energy = openmc.Tally(name='spettro_energia_neutroni')
tally_energy.filters = [surface_filter, particle_filter, energy_filter]
tally_energy.scores = ['current']
tallies.append(tally_energy)

# Tally Spettro Angolare (Corrente uscente)
tally_polar = openmc.Tally(name='spettro_angolare_neutroni')
tally_polar.filters = [surface_filter, particle_filter, polar_filter]
tally_polar.scores = ['current']
tallies.append(tally_polar)

tallies.export_to_xml()

# --- 7. ESECUZIONE ---
openmc.run(output=True, geometry_debug=False)

