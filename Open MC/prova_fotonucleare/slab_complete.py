#!/usr/bin/env python
# coding: utf-8

# In[17]:


import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import random

#openmc.config['cross_sections'] = "/home/renato/Desktop/OpenMC/xsdata/withPhotonuclear/mcnp_endfb71/cross_sections.xml"


# In[18]:


os.system("rm *.h5")
os.system("rm *.xml")
os.system("rm *.out")


# # 1. Materials

# In[29]:


# Create homogeneous U-water mixture
fuel = openmc.Material(name='Homogeneous')

fuel.set_density('g/cm3', 5)

# 1
fuel.add_nuclide("H1", 2)
fuel.add_nuclide("U235", 0.05)
fuel.add_nuclide("U238", 0.95)

materials = openmc.Materials([fuel])
materials.cross_sections = "/home/bossi_ricky/openmc_data/mcnp_endfb71/cross_sections.xml"

materials.export_to_xml()


# # 2. Geometry

# In[20]:


H = 40 # Total slab length (cm)

# Define slab surfaces
left = openmc.XPlane(x0= -H/2, boundary_type='vacuum')
right = openmc.XPlane(x0= H/2, boundary_type='vacuum')

# Define the slab cell
slab_region = +left & -right
slab_cell = openmc.Cell(region=slab_region)
slab_cell.fill = fuel

# Create universe and geometry
universe = openmc.Universe(cells=[slab_cell])
geometry = openmc.Geometry(universe)

geometry.export_to_xml()


# In[21]:


# Create the plot
fig = geometry.plot(basis='xy', origin=(0, 0, 0), width=(2*H, 2*H), pixels=(400, 400))

# Set custom x and y labels
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Slab Geometry (XY plane)')
plt.show()


# # 3. Settings

# In[22]:


seed = random.randint(1, int(1e9))

source = openmc.IndependentSource()
source.space = openmc.stats.Point((-H/2 + 0.001, 0.0, 0.0))
source.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
source.energy = openmc.stats.Discrete([12.5e6], [1.0])  # 12.5 MeV
source.particle = 'photon'

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = 10
settings.particles = 1000
settings.photonuclear_physics = True
settings.photon_transport = True
settings.source = source
settings.seed = seed


settings.export_to_xml()


# # 4. Tallies

# In[23]:


tallies = openmc.Tallies()

# --- Filters ---
# 1. Cell
cell_filter = openmc.CellFilter(slab_cell)

# 2. Energy
energy_bins = np.logspace(-5, 7, num=100)  # 100 bins
energy_filter = openmc.EnergyFilter(energy_bins)

# 3. Space 
mesh = openmc.RegularMesh()
mesh.dimension = [100, 1, 1]             
mesh.lower_left = [-H/2, -1e32, -1e32]       
mesh.upper_right = [H/2, 1e32, 1e32] 

#
mesh_filter = openmc.MeshFilter(mesh)

#
n_filter = openmc.ParticleFilter("neutron")

# TALLIES
# --- Tally 1: Energy-dependent Flux ---
flux_tally_E = openmc.Tally(name='Phi(E)')
flux_tally_E.filters = [cell_filter, energy_filter, n_filter]
flux_tally_E.scores = ['flux']

# --- Tally 2: Space Flux ---
flux_tally_x = openmc.Tally(name='Phi(x)')
flux_tally_x.filters = [mesh_filter, n_filter]
flux_tally_x.scores = ['flux']

# --- Tally 3: Total reaction rate ---
tot_RR = openmc.Tally(name='PhiSigma_t')
tot_RR.filters = [n_filter]
tot_RR.scores = ['total']

# --- Tally 4: Total flux ---
tot_Phi = openmc.Tally(name='Phi_tot')
tot_Phi.filters = [n_filter]
tot_Phi.scores = ['flux']

# --- Register the tallies ---
tallies = openmc.Tallies([flux_tally_E, flux_tally_x, tot_RR, tot_Phi])

tallies.export_to_xml()


# # 5. Run

# In[24]:


# Run the simulation and create statepoint file
openmc.run()


# # Postprocessing

# In[25]:


sp = openmc.StatePoint('statepoint.10.h5')


# Also, statepoint stores the tally data

# In[26]:


# Energy-dependent flux
flux_tally = sp.get_tally(name='Phi(E)')
flux_vals = flux_tally.mean.flatten()
Uflux_vals = flux_tally.std_dev.flatten()

# Energy bins (midpoints)
energy_mid = 0.5 * (energy_bins[:-1] + energy_bins[1:])

# Plot the flux spectrum
plt.figure(figsize = (6,5))
plt.step(energy_mid, flux_vals, color = "black", where = "pre")
plt.fill_between(energy_mid, flux_vals - Uflux_vals, flux_vals + Uflux_vals, step='pre', color='black', alpha=0.25)
plt.xscale("log")
plt.xlabel("Energy (eV)")
plt.ylabel("Flux (n cm / src)")
plt.title("Neutron Energy Flux")
plt.grid(True, linestyle = ":")
#plt.savefig("./phiE.pdf", format = "pdf", bbox_inches = "tight", dpi = 400)
plt.show()


# Space flux

# In[27]:


# Space flux
tally = sp.get_tally(name='Phi(x)')
flux_mean = tally.mean.flatten()   # 1D array: [bin0, bin1, ..., binN]
Uflux_mean = tally.std_dev.flatten()

# Access the mesh from the filter
mesh = tally.filters[0].mesh
x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], mesh.dimension[0] + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

plt.figure(figsize = (6,5))
plt.step(x_centers, flux_mean, color = "red", where = "pre")
plt.fill_between(x_centers, flux_mean - Uflux_mean, flux_mean + Uflux_mean, step='pre', color='red', alpha=0.25)
plt.xlabel('x (cm)')
plt.ylabel('Flux (n  cm / src)')
plt.title('1D Flux Profile Along X')
plt.grid(True, linestyle = ":")
plt.legend()
plt.show()


# Total reaction rate

# In[28]:


tot_RR = sp.get_tally(name='PhiSigma_t').mean.flatten() # reac / source
tot_Phi = sp.get_tally(name='Phi_tot').mean.flatten()   # n cm / source

Sigma_Tot = tot_RR / tot_Phi # 1/cm
MeanFreePath = 1 / Sigma_Tot # cm

print(f"Sigma total = {Sigma_Tot[0]:.3f} cm^-1")
print(f"Mean Free Path = {MeanFreePath[0]:.3f} cm")

