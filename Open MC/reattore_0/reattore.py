import openmc
import numpy as np

# _________________ materiali _________________

arricchimento = 0.04  # Frazione atomica (a/o) in questo contesto
fuel = openmc.Material(material_id=1, name='fuel')
fuel.add_nuclide('U235', arricchimento)
fuel.add_nuclide('U238', 1.0 - arricchimento)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.96)

# materiale di rivestimento
cladding = openmc.Material (material_id=2, name = 'cladding')
cladding.add_element('Zr', 1.0) #naturale, non ho specificato le A.I.
cladding.set_density('g/cm3', 6.49)

# materiale di moderatore
water = openmc.Material (material_id=3, name = 'water')
water.add_nuclide('H2', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.75)
water.add_s_alpha_beta('c_D_in_D2O') # aggiungo la tabella di scattering termico
# n.b.  le sezioni d'urto veloci sono già incluse di default

# aggiungo unriflettore
riflettore = openmc.Material(material_id=4, name='riflettore')
riflettore.add_element('C', 1.0)
riflettore.set_density('g/cm3', 2.26)

# voglio una sorgente puntiforme di neutroni in un punto specifico
marker_mat = openmc.Material(material_id=5, name='marker')
marker_mat.set_density('g/cm3', 1e-10) # Densità quasi nulla per non disturbare la fisica
marker_mat.add_nuclide('He4', 1.0) # elio, non interagisce praticamente

# Carburo di Boro B4C per le barre di controllo
control_rod = openmc.Material(name='B4C')
control_rod.add_element('B', 4.0)
control_rod.add_element('C', 1.0)
control_rod.set_density('g/cm3', 2.52) # Densità teorica B4C

materials = openmc.Materials([fuel, cladding, water, riflettore, marker_mat])
materials.export_to_xml()

# _________________ geometria _________________

pitch  = 1.4 # cm, lato esterno del moderatore
cladding_thickness = 0.1 # cm, spessore del rivestimento
fuel_side = 1 # cm, lato del combustibile

# -------- creo singolo blocco (pin)--------

# creo le superfici
f_min , f_max = -fuel_side/2, fuel_side/2 # lati del fuel
fuel_box = openmc.model.RectangularParallelepiped(f_min, f_max, f_min, f_max, f_min, f_max ) # box del fuel

# superfici del rivestimento
c_side = fuel_side + 2*cladding_thickness
c_min, c_max = -c_side/2, c_side/2
clad_box = openmc.model.RectangularParallelepiped(c_min, c_max, c_min, c_max, c_min, c_max)

# moderatore
p_min, p_max = -pitch/2, pitch/2
moderator_box = openmc.model.RectangularParallelepiped(p_min, p_max, p_min, p_max, p_min, p_max, boundary_type = 'transmission') 

# ora creo l'Universo "Pin" (il cubetto standard)
u_pin = openmc.Universe(name='Cubetto Fuel')
c_fuel = openmc.Cell(fill=fuel, region=-fuel_box) # riempio ciascun volume prima definito con un materiale
c_clad = openmc.Cell(fill=cladding, region=+fuel_box & -clad_box)
c_mod  = openmc.Cell(fill=water, region=+clad_box & -moderator_box)
u_pin.add_cells([c_fuel, c_clad, c_mod]) # aggiungo la cella all'universo, così ne ho creata una

# ora creo lo spazio vuoto 
u_void = openmc.Universe(name='Vuoto')
c_void = openmc.Cell(fill=None, region=-moderator_box) # fill=None indica vuoto
u_void.add_cell(c_void) # essenzilmente dico che fuori dal cubetto c'è il vuoto, ma poi lo reitero nelle caselle successive

# ora reitero il cubetto in una matrice 3D 
lattice = openmc.RectLattice() # creo la struttura a reticolo
lattice.lower_left = (-pitch*2.5, -pitch*2.5, -pitch*2.5) # posizione del vertice inferiore sinistro
lattice.pitch = (pitch, pitch, pitch) # distanza tra i centri dei cubetti

N = 9  # numero di cubetti per lato DEVE ESSERE DISPARI

half_width = (N * pitch) / 2  # metà larghezza del reticolo
center_index = N // 2 # Indice centrale per matrici (es. se N=5, center=2)

lattice = openmc.RectLattice() # creo la struttura a reticolo
lattice.lower_left = (-half_width, -half_width, -half_width) # posizione del vertice inferiore sinistro
lattice.pitch = (pitch, pitch, pitch) # distanza tra i centri dei cubetti

u_matrix = np.full((N, N, N), u_pin) # nota, le coordinate in open mc sono (z,y,x)
 
# Il canale parte dalla faccia superiore (z_max) e arriva al centro (z_center)
# Usiamo l'indice center_index per centrarlo in X e Y
for z in range(center_index, N): 
    u_matrix[z, center_index, center_index] = u_void

lattice.universes = u_matrix

# --- AVVOLGO TUTTO NEL RIFLETTORE ---
reflector_thickness = 0.3  # spessore del riflettore in cm

# Definisco la superficie che delimita il core (dove finisce la lattice)
core_boundary = openmc.model.RectangularParallelepiped(
    -half_width, half_width, 
    -half_width, half_width, 
    -half_width, half_width
)

# Definisco la superficie esterna del riflettore (il confine del "mondo")
# Usiamo i limiti già aggiornati con lo spessore
outer_reflector_box = openmc.model.RectangularParallelepiped(
    -half_width - reflector_thickness, half_width + reflector_thickness,
    -half_width - reflector_thickness, half_width + reflector_thickness,
    -half_width - reflector_thickness, half_width + reflector_thickness,
    boundary_type='vacuum'
)

# Creiamo una sferetta di raggio 0.1 cm nella posizione della sorgente
source_sphere = openmc.Sphere(x0=0, y0=0, z0=half_width - 1, r=0.1)
source_marker_cell = openmc.Cell(fill=marker_mat, region=-source_sphere)

# Creazione delle CELLE
# La main_cell deve contenere la lattice ma fermarsi alla core_boundary
main_cell = openmc.Cell(fill=lattice, region=-core_boundary & +source_sphere)

# La reflector_cell occupa lo spazio tra core_boundary (+) e outer_reflector_box (-)
reflector_cell = openmc.Cell(fill=riflettore, region=+core_boundary & -outer_reflector_box)

geometry = openmc.Geometry([main_cell, reflector_cell, source_marker_cell])
geometry.export_to_xml()

# _________________ plot ________________

import os

output_folder = 'visualizzazione'
# Crea la cartella se non esiste (evita errori di "directory not found")
os.makedirs(output_folder, exist_ok=True)

# Calcoliamo la larghezza totale (Core + Riflettore su entrambi i lati)
total_width = 2 * (half_width + reflector_thickness)

plot = openmc.Plot()
plot.filename = os.path.join(output_folder, 'reattore_2D_plot')
plot.origin = (0, 0, 0)
plot.width = (total_width, total_width) # <--- Modificato qui
plot.pixels = (1000, 1000) # Aumentiamo un po' i pixel per vedere meglio il dettaglio
plot.basis = 'xz' 
plot.color_by = 'material' 
plot.colors = {fuel: 'yellow', cladding: 'gray', water: 'blue', riflettore: 'green', marker_mat: 'red'}

# PLOT 3D (Voxel)
plot_3d = openmc.Plot()
plot_3d.filename = os.path.join(output_folder, 'mappa_3d')
plot_3d.type = 'voxel' 
plot_3d.width = (2*half_width, 2*half_width, 2*half_width) 
plot_3d.pixels = (150, 150, 150) 
plot_3d.color_by = 'material'

# Esportazione
plots = openmc.Plots([plot, plot_3d])
plots.export_to_xml()


# ________________ settings _________________
settings = openmc.Settings()
settings.batches = 10
settings.particles = 50000
#settings.run_mode = 'eigenvalue' # calcolo del k-eff
settings.run_mode = 'fixed source' # simulazione a sorgente fissata

# Definisco la sorgente: un punto sopra l'imboccatura del canale
# Il canale è centrato in (0,0) nel piano XY e parte da Z = half_width
source_pos = openmc.stats.Point((0, 0, half_width -1))
# Direzione: verso il basso (lungo l'asse Z negativo)
source_dir = openmc.stats.Monodirectional((0, 0, -1))
# Energia: 2 MeV (neutroni veloci)
source_energy = openmc.stats.Discrete([1e6], [1.0]) 


settings.source = openmc.IndependentSource(space=source_pos, angle=source_dir, energy=source_energy)

# Abilitiamo il tracciamento delle particelle
# Scriviamo i file .h5 delle tracce per le prime 10 particelle del primo batch
settings.track = [(1, 1, i) for i in range(1, 11)]

settings.export_to_xml()


# _________________ tallies _________________

#In OpenMC, i Tallies sono "stimatori": strumenti statistici che registrano eventi o tracciano particelle per calcolare grandezze fisiche medie. Non sono semplici contatori, ma integrali sul dominio dello spazio delle fasi (posizione, energia, angolo, tempo)

tallies = openmc.Tallies()

# DEFINIZIONE DELLA MESH (La griglia di campionamento)
# Usiamo una mesh 2D sul piano XZ per vedere la sezione del canale
mesh = openmc.RegularMesh()
mesh.dimension = [150, 150 , 150] # Risoluzione: 150x150 pixel
# Copriamo l'intero sistema, dal bordo del riflettore al bordo opposto
mesh.lower_left = [-half_width, -half_width, -half_width] 
mesh.upper_right = [half_width, half_width, half_width]

# FILTRI
mesh_filter = openmc.MeshFilter(mesh)
# Filtro per cella: ci permette di isolare i dati solo nel combustibile
fuel_filter = openmc.CellFilter([c_fuel]) 

# TALLY: MAPPA SPAZIALE DELLE FISSIONI
# Utile per vedere graficamente dove "si accende" il reattore
tally_map = openmc.Tally(name='mappa_fissioni')
tally_map.filters = [mesh_filter]
tally_map.scores = ['nu-fission', 'fission', 'absorption', 'flux' , 'total']
tallies.append(tally_map)

# Filtro Energetico per lo Spettro (500 bin logaritmici tra 1e-5 eV e 20 MeV)
energies = np.logspace(-5, 7, 1001)
energy_filter = openmc.EnergyFilter(energies)

# Tally per lo Spettro Energetico Globale
tally_spec = openmc.Tally(name='spettro_energetico')
tally_spec.filters = [energy_filter]
tally_spec.scores = ['flux']
tallies.append(tally_spec)


tallies.export_to_xml()

openmc.run()