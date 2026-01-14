#-------------------------------------------------------#
# STEP 2 OF GENERATING INITIAL CONDITIONS!              #     
#    saves a .dat.10 file to give to petar.init         #
#    saves a .txt file summarizing cluster properties   #
#-------------------------------------------------------#

import numpy as np
from numpy.random import uniform, normal
import pandas as pd

import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.table import Table

from scipy.optimize import root_scalar
from scipy.integrate import RK45, quad, simpson, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.special import erf
import scipy.stats as stats

# from cosmic.sample.initialcmctable import InitialCMCTable
# from cosmic.sample.sampler import cmc
# from cosmic.sample.initialbinarytable import InitialBinaryTable
# from cosmic.sample.sampler import multidim


import argparse

# the king profile script
import make_king_profile as king

### an amuse-like Particles object
class Particles():
    def __init__(self, n=None, length_unit = u.pc, v_unit = u.km/u.s, mass_unit = u.Msun):
        if n==None:
            self.pos = None
            self.vel = None
            self.mass = None
        if n!=None:
            self.pos = np.zeros((n, 3))*length_unit
            self.vel = np.zeros((n, 3))*v_unit
            self.mass = np.zeros(n)*mass_unit

    # def add_particles(self, new_particles):
    #     """
    #     append particles object: THIS IS WRONG AND APPENDS DUPLICATES IF STARTING FROM NO PARTICLES!
    #     """
    #     if self.pos==None:
    #         self.pos = new_particles.pos
    #     if self.pos!=None:
    #         self.pos = np.append(self.pos, new_particles.pos, axis=0)
    
    #     if self.vel==None:
    #         self.vel = new_particles.vel
    #     if self.vel!=None:
    #         self.vel = np.append(self.vel, new_particles.vel, axis=0)
    
    #     if self.mass==None:
    #         self.mass = new_particles.mass
    #     if self.mass!=None:
    #         self.mass = np.append(self.mass, new_particles.mass, axis=0)
    def add_particles(self, new_particles):
        if self.pos is None:
            self.pos = new_particles.pos
        else:
            self.pos = np.append(self.pos, new_particles.pos, axis=0)

        if self.vel is None:
            self.vel = new_particles.vel
        else:
            self.vel = np.append(self.vel, new_particles.vel, axis=0)

        if self.mass is None:
            self.mass = new_particles.mass
        else:
            self.mass = np.append(self.mass, new_particles.mass, axis=0)


### generate positions and velocities in a king profile given a list of masses
def create_king_cluster(w0, masses, virial_radius, tidal_boundary=1e-6):
    """
    returns a particles object: mass, position, velocity in a king profile
    the king density profile with this w0, virial radius, and total mass.

    virial radius should be an astropy quantity
    masses should be an astropy quantity
    """
    Mclust = np.sum(masses)
    masses = masses.to(u.Msun).value 
    N = len(masses)
    r, v, m = king.draw_r_v(masses, w0, tidal_boundary) 

    # rescale and orient position
    theta_pos = np.arccos(1-2*np.random.rand(N))
    phi_pos = 2*np.pi * np.random.rand(N)
    r*=virial_radius.to(u.pc).value
    r*=u.pc
    x = r*np.sin(theta_pos)*np.cos(phi_pos)
    y = r*np.sin(theta_pos)*np.sin(phi_pos)
    z = r*np.cos(theta_pos)

    pos = np.stack([x, y, z]).T

    # rescale and orient velocity
    velocity_scale = np.sqrt(const.G * Mclust / virial_radius)
    v*=velocity_scale.to(u.km/u.s).value
    v*=(u.km/u.s)

    theta_vel = np.arccos(1-2*np.random.rand(N))
    phi_vel = 2*np.pi*np.random.rand(N)
    vx = v*np.sin(theta_vel)*np.cos(phi_vel)
    vy = v*np.sin(theta_vel)*np.cos(phi_vel)
    vz = v*np.cos(theta_vel)

    vel = np.stack([vx, vy, vz]).T

    # rescale M 
    #### not strictly necessary since king.draw_r_v should return masses in the same order as they are input
    m *= Mclust.to(u.Msun).value
    m *= u.Msun

    cluster = Particles()
    cluster.mass = m
    cluster.pos = pos
    cluster.vel = vel

    return cluster # also return a density profile???


### calculate half mass radius, for sanity checking purposes
def calculate_half_mass_radius(cluster):
    """
    cluster is a particles object
    """
    x, y, z = cluster.pos.T
    radii = np.sqrt(x**2 + y**2 + z**2)
    masses = cluster.mass

    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    sorted_masses = masses[sorted_indices]

    cumulative_mass = np.cumsum(sorted_masses)
    total_mass = np.sum(masses)

    half_mass = total_mass/2
    hmr_index = np.searchsorted(cumulative_mass, half_mass)
    hmr = sorted_radii[hmr_index]

    return hmr



############### BINARIES ##################

### utility functions
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

def calc_P(a, Mtot):
    """
    period from semimajor axis
    provide astropy quantities for both inputs
    returns an astropy quantity with units.
    """
    P = 2*np.pi * np.sqrt((a**3)/(const.G * Mtot))
    return P

def calc_a(P, Mtot):
    """
    semimajor axis from period
    provide astropy quantities for both inputs
    returns an astropy quantity with units.
    """
    a = (const.G*Mtot * (P/(2*np.pi))**2)**(1/3)
    return a


### adding the binaries
def new_binary_orbit(M1, M2, a, e):
    """
    provide M1, M2, a, as astropy quantities
    """
    Mtot = M1 + M2
    Mfrac_1 = M1/Mtot

    binary = Particles(2)
    binary.mass[0] = M1
    binary.mass[1] = M2
    
    mu = const.G * Mtot

    # relative velocity and position at periastron
    vel_peri = np.sqrt(
        (mu/a) * ((1.+e)/(1.-e))
    )
    r_peri = a * (1.-e)

    # scale positions and velocities based on relative mass
    pos1 = (1.-Mfrac_1) * r_peri * np.array([1.,0.,0.])
    pos2 = Mfrac_1 * r_peri * np.array([-1.,0.,0.])
    vel1 = (1.-Mfrac_1)*vel_peri * np.array([0.,1.,0.])
    vel2 = Mfrac_1 * vel_peri * np.array([0.,-1.,0.])

    # rotate the angular momentum vector randomly
    L0 = np.array([0,0,1])
    L_new = random_unit_vector()
    R = rotation_matrix_from_vectors(L0, L_new)

    binary.pos[0] = np.dot(R, pos1)
    binary.pos[1] = np.dot(R, pos2)

    binary.vel[0] = np.dot(R, vel1)
    binary.vel[1] = np.dot(R, vel2)

    return binary

def add_secondaries(CoMs, primary_masses, secondary_masses, Ps, es):
    """
    add binaries to a Particles object 
    the first 2*Nbin elements of the Particles object will be converted to paired binaries
    where Nbin = len(primary_masses)
    """
    Nbin = len(primary_masses)

    singles_in_binaries = Particles() # will populate this with binaries, then add the single objects to it.

    CoM_positions = CoMs.pos
    binary_cm_positions = CoM_positions[:Nbin]
    CoM_velocities = CoMs.vel 
    binary_cm_velocities = CoM_velocities[:Nbin]

    for n in range(Nbin):
        m1 = primary_masses[n]
        m2 = secondary_masses[n]

        P = Ps[n]
        a = calc_a(P, m1+m2)
        e = es[n]

        binary = new_binary_orbit(m1, m2, a, e)

        # move the binary from the origin to the original CM phase space position from the king profile
        binary.pos += binary_cm_positions[n]
        binary.vel += binary_cm_velocities[n]

        # add new binary particles to singles_in_binaries
        singles_in_binaries.add_particles(binary)

    # isolate single stars; somewhat clumsy without amuse but whatever.
    single_stars = Particles()
    single_stars.pos = CoMs.pos[Nbin:]
    single_stars.vel = CoMs.vel[Nbin:] 
    single_stars.mass = CoMs.mass[Nbin:] 

    singles_in_binaries.add_particles(single_stars)

    return(singles_in_binaries)

# save data as McLuster would! ! ! 
def save_data(cluster, filename):
    x, y, z = cluster.pos.T.to(u.pc).value
    vx, vy, vz = cluster.vel.T.to(u.km/u.s).value
    m = cluster.mass.to(u.Msun).value
    data = np.column_stack((m, x, y, z, vx, vy, vz))
    np.savetxt(filename+".dat.10", data, delimiter='\t', fmt='%3.16f')
    print(f"Data saved to {filename}.dat.10")



#----------------------------------------------#
#              Main Program Below              #
#----------------------------------------------#

### parameters are N, w0, rvir, 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--w0', type=float, default=3.0)
    parser.add_argument('--rvir_pc', type=float, default=3.0)
    parser.add_argument('--f_in', type=str, default="mass_binary_info") #.npz
    parser.add_argument("--f_out", type=str, default='clust')
    args = parser.parse_args()

    rvir = args.rvir_pc * u.pc
    w0 = args.w0

    input_mass_data = args.f_in
    output_filename = args.f_out


    # load and unpack COSMIC data
    print("loading and unpacking COSMIC data")
    mass_data = np.load(input_mass_data+".npz")
    primary_masses = mass_data["primary_masses"] * u.Msun
    secondary_masses = mass_data["secondary_masses"] * u.Msun
    Mtots = mass_data["Mtots"] * u.Msun
    Ps = mass_data["periods"] * u.day
    es = mass_data["es"] 
    all_CMs = mass_data["all_CMs"] * u.Msun

    # some accounting
    print("accounting:")
    print("# primaries, secondaries, total masses = ", len(primary_masses), len(secondary_masses), len(Mtots))
    print("length of centers of masses", len(all_CMs))
    print("# of singles", len(all_CMs)-len(Mtots))
    n_singles = len(all_CMs) - len(Mtots)
    n_binaries = len(primary_masses)
    fbin = n_binaries / (n_singles+n_binaries)
    nstars = n_singles + 2*n_binaries
    print("unpacked %i stars (%i singles and %i binaries)"%(nstars, n_singles, n_binaries))

    # put into a king cluster
    print("creating a king cluster with singles + binary centers of mass")
    init_cluster = create_king_cluster(w0=w0, masses=all_CMs, virial_radius=rvir)
    print("length of init_cluster", len(init_cluster.mass))
    print("before binaries should be", len(all_CMs))

    print("adding %i binaries to the King cluster"%n_binaries)
    cluster = add_secondaries(CoMs=init_cluster, primary_masses=primary_masses,
                              secondary_masses=secondary_masses,
                              Ps=Ps, es=es)
    print("does the cluster have the right number of stars though??")
    print(len(cluster.mass))

    print("cluster prepared; saving to a .dat.10 file")
    save_data(cluster, output_filename)

    hmr = calculate_half_mass_radius(cluster).to(u.pc).value
    print("half mass radius=%.2f pc"%hmr)
    lines = [
        "nstars           : "+str(len(cluster.mass)),
        "n_binaries       : "+str(n_binaries), # for petar
        "vir. radius      : "+str(rvir.to(u.pc).value)+" pc",
        "fbin             : "+str(fbin),
        "half-mass radius : "+str(hmr)+" pc", 
    ]

    output_filename+=".txt"
    with open(output_filename, "w") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"IC info printed to {output_filename}")