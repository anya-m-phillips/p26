#### run this notebook in an envoronment with 
# both COSMIC and amuse-framework installed!!
# also requires the hacked multidim script that returns single masses.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

from scipy.optimize import root_scalar # used for estimating how to scale COSMIC outputs for desired sim size.

import numpy as np
from amuse.units import units, constants, quantities, nbody_system
from amuse import datamodel
from amuse.datamodel import Particles
from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.ic.plummer import new_plummer_sphere
import scipy.stats as stats
from astropy.io import fits
from astropy.table import Table

from cosmic.sample.initialcmctable import InitialCMCTable
from cosmic.sample.sampler import cmc
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import multidim


#### use AMUSE to generate positions and velocities of a in plummer profile
def create_plummer_cluster(masses, N=10000, radius = 3.13 | units.pc,
                           r_cutoff = 5):
    """
    create a cluster following a Kroupa IMF
    masses should be AMUSE quantities | u.MSun.
    radius is the virial radius and should be an AMUSE quantity 
    r_cutoff is the max radius in virial radii
    defaults set so that half mass radius is ~3pc
    """
    # print("doing convert nbody")
    convert_nbody = nbody_system.nbody_to_si(masses.sum(), radius)
    # print("generating plummer sphere in n-body units")
    cluster = new_plummer_sphere(N, convert_nbody, radius_cutoff=r_cutoff)
    # print("painting on masses")
    cluster.mass = masses
    # print("moving to center")
    cluster.move_to_center()
    # print("converting back from nbody units")
    cluster.scale_to_standard(convert_nbody)
    return cluster

def calculate_half_mass_radius(cluster):
    """
    this is for checking the half mass radius
    for sanity purposes
    """
    radii = np.sqrt(
        cluster.x.value_in(units.parsec)**2 + cluster.y.value_in(units.parsec)**2 + cluster.z.value_in(units.parsec)**2
    )
    masses = cluster.mass.value_in(units.MSun)

    # Sort stars by radius
    sorted_indices = np.argsort(radii)
    sorted_radii = np.array(radii)[sorted_indices]
    sorted_masses = np.array(masses)[sorted_indices]

    # Compute cumulative mass
    cumulative_mass = np.cumsum(sorted_masses)

    # Total mass
    total_mass = np.sum(masses)

    # Find the half-mass radius
    half_mass = total_mass / 2
    half_mass_radius_index = np.searchsorted(cumulative_mass, half_mass)
    half_mass_radius = sorted_radii[half_mass_radius_index]
    # print("half mass radius = %.3f pc"%half_mass_radius)
    return half_mass_radius 

## for orbit orientations
# deal with orbit orientation: 
def random_unit_vector():
    theta = np.arccos(1-2*np.random.rand()) 
    phi = 2 * np.pi * np.random.rand()
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def rotation_matrix_from_vectors(v1, v2):
    """
    compute the rotation matrix that rotates from v1 to v2
    """
    # normalize the vectors
    v1_norm = v1/np.linalg.norm(v1) 
    v2_norm = v2/np.linalg.norm(v2)
    
    u = np.cross(v1_norm, v2_norm)
    c = np.dot(v1_norm, v2_norm)

    s = np.linalg.norm(u)
    if s==0:
        return np.identity(3) # case where v1 and v2 are already aligned
    
    u_norm = u/np.linalg.norm(u)
    u_x = np.array([[0, -u_norm[2], u_norm[1]],
                    [u_norm[2], 0, -u_norm[0]],
                    [-u_norm[1], u_norm[0], 0]])
    
    u_u = np.outer(u_norm, u_norm.T)
    

    
    R = c*np.identity(3) + s*u_x + (1-c)*(u_u) # https://en.wikipedia.org/wiki/Rotation_matrix (rotation matrix from axis and angle)
                                                # see also: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula ; equivalent - rodrigues' formula
    return R


### convert btwn period/orb separation with Kepler's 3rd law
# convert between period and smimajor axis:
def calc_P(a, Mtot):
    """
    period from semimajor axis
    provide amuse quantities for both inputs
    """
    P = 2*np.pi * np.sqrt((a**3)/(constants.G * Mtot))
    return P

def calc_a(P, Mtot):
    """
    semimajor axis from period
    provide amuse quantities for both inputs
    """
    a = (constants.G*Mtot * (P/(2*np.pi))**2)**(1/3)
    return a

def new_binary_orbit(M1, M2, a, e):
    """
    M1, M2: primary and secondary mass; amuse quantity
    a: semimajor axis, amuse quantity ? 
    e: eccentricity
    """  
    # m1 = M.value_in(units.Msun)
    # m2 = M2.value_in(units.Msun)

    Mtot = M1 + M2
    Mfrac_1 = M1/Mtot

    binary = Particles(2)
    binary[0].mass = M1
    binary[1].mass = M2

    mu = constants.G * Mtot

    # velocity and radius at perihelion
    vel_peri = np.sqrt(
        (mu/a) * ((1.+e)/(1.-e))
    )
    r_peri = a * (1.-e)

    # initial phase space positions before rotating
    pos1 = np.array((1.-Mfrac_1) * r_peri * [1.,0.,0.])
    pos2 = -np.array(Mfrac_1 * r_peri * [1.,0.,0.])
    vel1 = np.array((1.-Mfrac_1)*vel_peri * [0.,1.,0.])
    vel2 = -np.array(Mfrac_1 * vel_peri * [0.,1.,0.])

    # rotate angular momentum vector randomly
    L0 = np.array([0,0,1])
    L_new = random_unit_vector()
    R = rotation_matrix_from_vectors(L0, L_new)

    binary[0].position = np.dot(R, pos1)
    binary[1].position = np.dot(R, pos2)
    
    binary[0].velocity = np.dot(R, vel1)
    binary[1].velocity = np.dot(R, vel2)

    return binary


def add_secondaries(CoMs, primary_masses, secondary_masses,
                     Nbin, Ps, es):# , mmin=0.08 | units.MSun):
    """
    everything is AMUSE quantities...
    """
    print('instantiating particles objects')
    singles_in_binaries=Particles() # confusingly named. will first populate with binaries then add the single star CoM phase space data
    binary_CoMs = CoMs[:Nbin] #.random_sample(Nbin); 
    #### note, removed random sample so that secondary masses are added in the same order they were to compute the plummer profile.

    n = 0
    # print('looping through binarie to add orbits')
    for CoM in binary_CoMs:
        # print(n)

        # print('get m1, m2')
        m1 = primary_masses[n] #| units.MSun #binary.mass
        m2 = secondary_masses[n] #| units.MSun

        # print('get P, compute a')
        P = Ps[n] # mimicks vedant's notebook.
        a = calc_a(P, m1+m2)
        # print('get e')
        e = es[n]
        
        # create an orbiting binary centered at 0
        # print('create binary orbit')
        bin_with_orbit = new_binary_orbit(m1, m2, a, e)

        # move the CM position to that of the original particle
        # print('move binary to CM position')
        bin_with_orbit.position += CoM.position
        bin_with_orbit.velocity += CoM.velocity

        # append to "singles_in_binaries"
        # print('append to singles in binaries')
        bin_with_orbit = singles_in_binaries.add_particles(bin_with_orbit)
        n+=1

    single_stars = CoMs - binary_CoMs # isolate things we did not add companions to
    singles_in_binaries.add_particles(single_stars) # add the binaries we made in the loop to the list of isolated single stars.
    # note if this is a simple append at the end situation, no matter which stars we choose, we are ending up with binaries listed first always.
    return singles_in_binaries

# saving data in petar readable format:
def save_data(M, X, Y, Z, VX, VY, VZ, filename):
    # print("min x")
    data = np.column_stack((M, X, Y, Z, VX, VY, VZ))
    np.savetxt(filename, data, delimiter='\t', fmt='%3.16f')  # i think this mimicks mcluster...
    print(f"Data saved to {filename}")   


#--------------------------------------#
#         Main program below           #
#--------------------------------------#


rh = 2.0 | units.pc 
factor = (1/(0.5**(2/3)) - 1)**(-0.5)
a = rh/factor
virial_radius = (16/(3*np.pi)) * a

# virial_radius = 2.0 | units.parsec
cutoff_radius = 5
N_stars_desired = 15000
filename = "clust_15K_rh2"

lines = []

string = "generating binary population with COSMIC"
print(string)
lines+=[string]
InitialBinaries, mass_singles, mass_binaries, n_singles, n_binaries, singles_mass_list = InitialBinaryTable.sampler('multidim',final_kstar1=[6], final_kstar2=[6], rand_seed=2, nproc=1, SF_start=13700.0, SF_duration=0.0, met=0.0002, size=int(1e4))

def estimate_scale(scale, N_target, n_singles, n_binaries):
    """
    scale: frac. by which to scale the sampling of singles/binaries 
    N_target: target number of stars in simulation
    n_singles, n_binaries: as generated by cosmic
    """
    N = int(n_singles*scale) + 2*int(n_binaries*scale)
    # print(N, N_target)
    diff = N-N_target
    return diff
# use scipy root_scalar to find desired scale
result = root_scalar(f=estimate_scale, x0=0.5, method='bisect',
                       args = (N_stars_desired, n_singles, n_binaries), bracket = [0, 1])
if result.converged==False:
    string+="WARNING! SCIPY SCALE FACTOR SEARCH FAILED!!"
    print(string)
    lines+=[string]
if result.converged==True:
    string="scipy.optimize.root_scalar successfully converged on a scale factor."
    print(string)
    lines+=[string]
scale = result.root
N_stars = int(n_singles * scale) + 2*int(n_binaries*scale)
string="scaled COSMIC outputs to generate %i stars"%N_stars
print(string)
lines+=[string]

# select scaled amount of singles/binaries
string="sampling singles/binaries"
print(string)
lines+=[string]
sampled_binaries = InitialBinaries.sample(int(scale*n_binaries), replace=True)
sampled_binaries["Mtot"] = sampled_binaries["mass_1"] + sampled_binaries["mass_2"]

sel_single_ind = np.random.randint(low=0, high=len(singles_mass_list), size=int(scale*n_singles))
single_masses = singles_mass_list[sel_single_ind]
all_masses = np.concatenate([sampled_binaries['mass_1'].values, sampled_binaries['mass_2'].values, single_masses])

string="sanity check: %i masses"%len(all_masses)
print(string)
lines+=[string]

string="gathering data"
print(string)
lines+=[string]
M1s = sampled_binaries["mass_1"].values
M2s = sampled_binaries['mass_2'].values
Mtots = sampled_binaries['Mtot'].values

all_CMs = np.concatenate([Mtots, single_masses])

binary_properties = sampled_binaries[["mass_1", "mass_2", "porb", "ecc"]].values

primary_masses, secondary_masses, Ps, es, = binary_properties.T


string="counting stars..."
print(string)
lines+=[string]
N_singles = len(single_masses)
N_binaries = len(primary_masses)
N_stars = N_singles + (2*N_binaries)
fbin = N_binaries / (N_singles+N_binaries)

string="adding AMUSE units to masses, periods"
print(string)
lines+=[string]
primary_masses = primary_masses | units.MSun
secondary_masses = secondary_masses | units.MSun
Ps = Ps | units.day
total_masses = all_CMs | units.MSun
single_masses = single_masses | units.MSun
string="data prepped for AMUSE"
print(string)
lines+=[string]


# generate a plummer cluster based on singles and binary centers of mass
string="generating cluster wo binaries"
print(string)
lines+=[string]
init_cluster = create_plummer_cluster(masses = total_masses, N = len(total_masses.value_in(units.MSun)), 
                                      radius=virial_radius, r_cutoff=cutoff_radius)
string="done"
print(string)
lines+=[string]

string="adding %i binaries"%N_binaries
print(string)
lines+=[string]
cluster = add_secondaries(init_cluster,
                          primary_masses=primary_masses, secondary_masses=secondary_masses,
                          Nbin=N_binaries, Ps=Ps, es=es)
string="done"
print(string)
lines+=[string]

hmr = calculate_half_mass_radius(cluster)

string = "tabulating phase space coordinates"
print(string)
lines+=[string]
x_vals = cluster.x.value_in(units.parsec)
y_vals = cluster.y.value_in(units.parsec)
z_vals = cluster.z.value_in(units.parsec)

vx_vals = cluster.vx.value_in(units.kms)
vy_vals = cluster.vy.value_in(units.kms)
vz_vals = cluster.vz.value_in(units.kms)

m_vals = cluster.mass.value_in(units.MSun)


string="saving data in McLuster format"
print(string)
lines+=[string]
save_data(m_vals, x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, filename=filename+".dat.10")

# write to an explain-y file
lines += [
    "nstars           : "+str(N_stars),
    "n_binaries       : "+str(N_binaries), # for petar
    "vir. radius      : "+str(virial_radius.value_in(units.parsec))+" pc",
    "cutoff radius    : "+str(cutoff_radius)+" virial radii",
    "fbin             : "+str(fbin),
    "half-mass radius : "+str(hmr)+" pc", 
]
output_file = filename+".txt"

with open(output_file, "w") as file:
    for line in lines:
        file.write(line + "\n")

print(f"IC info printed to {output_file}")
