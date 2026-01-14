import numpy as np
from amuse.units import units, constants, quantities, nbody_system
from amuse import datamodel
from amuse.datamodel import Particles
from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.ic.plummer import new_plummer_sphere
import scipy.stats as stats
from astropy.io import fits
from astropy.table import Table
# full send, only amuse units ;-;


### Create a cluster with all single stars:

def gen_kroupa_IMF(m_max, N):
    m0, m1, m2 = (0.08, 0.5, m_max) #* u.Msun
    if m_max<0.5:
        return "maximum mass <0.5 Msun not supported sry"
    
    # define normalization coefficients
    X1 = m0**(-0.3) - m1**(-0.3)
    X2 = m1**(-1.3) - m2**(-1.3)

    A1 = 1/((X1/0.3) + m1*(X2/1.3))
    A2 = m1/((X1/0.3) + m1*(X2/1.3))

    def C1(m):
        return A1 * (m0**(-0.3) - m**(-0.3))/0.3
    # print(C1(m1))
    def C2(m): # don't need actually
        return (A1 * (m0**(-0.3) - m1**(-0.3))/0.3) + (A2 * (m1**(-1.3) - m**(-1.3))/1.3)
    # print(C1(m1)==C2(m1), "should be equal")

    masses = []
    for i in range(N):
        u_rand = np.random.rand()
        if u_rand < C1(m1):
            m = (m0**(-0.3) - (0.3/A1)*u_rand)**(-1/0.3)
            masses.append(m)
        if u_rand >= C1(m1):
            m = (m1**(-1.3) - (1.3/A2)*(u_rand - (A1*X1)/0.3))**(-1/1.3)
            masses.append(m)
        
    return masses # returned as a list in solar units. #np.array(masses) | units.MSun

def add_secondary_masses(masses, nbin): 
    """
    compute a list of secondary masses.
    """
    mmin=0.08
    m2_values = []

    unchanged_masses = masses[nbin:]
    binary_total_masses = []

    for i in range(nbin):
        m1 = masses[i]
        m2 = np.random.uniform(mmin, m1, size=1)[0] #| units.MSun
        m2_values.append(m2)
        # masses[i] = m1 + m2
        mtot = m1+m2
        binary_total_masses.append(mtot)

    total_masses = binary_total_masses + unchanged_masses

    return m2_values, total_masses


def create_plummer_cluster(masses, N=10000, radius = 3.13 | units.pc,
                           r_cutoff = 5):
    """
    create a cluster following a Kroupa IMF
    masses should be AMUSE quantities | u.MSun.
    radius is the virial radius and should be an AMUSE quantity 
    r_cutoff is the max radius in virial radii
    M_max is in solar units but is not an amuse quantity (dimensionless)
    defaults set so that half mass radius is ~3pc
    """
    # print("creating cluster with %i stars." %N)
    # print("Virial radius = %.3f pc"%radius.value_in(units.parsec))
    # print("cutoff radius: %.3f pc"%r_cutoff*radius.value_in(units.parsec))
    # print("maximum stellar mass: %.2f Msun"%M_max)
    
    masses = np.array(masses) | units.MSun

    convert_nbody = nbody_system.nbody_to_si(masses.sum(), radius)
    cluster = new_plummer_sphere(N, convert_nbody, radius_cutoff=r_cutoff)
    cluster.mass = masses
    cluster.move_to_center()
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


###### generating binaries :D

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

def random_lognormal_period(Pmin = 0.2 | units.day, Pmax = 1e+9 | units.day,
                             center=5.03, sigma=2.28):
    """
    provide Pmin and Pmax as amuse quantities. 
    returns amuse quantity period in days.
    """
    loc = center
    scale = sigma

    logPmin = np.log10(Pmin.value_in(units.day))
    logPmax = np.log10(Pmax.value_in(units.day))

    a, b,  = (logPmin - loc)/scale, (logPmax - loc)/scale
    logP = stats.truncnorm(loc=loc, scale=scale, a=a, b=b).rvs(1)
    P = (10**logP) | units.day

    return P

def random_ecc_thermal():
    """
    drawn from 
    f(e) = 2e
    "thermal" distribution
    """
    e = np.sqrt(np.random.rand())
    return e

###
# generate the orbits

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
                     Nbin, P_min, e_dist="thermal"):# , mmin=0.08 | units.MSun):
    """
    something I'm wondering -- might be good to have the binaries
    add up to the mass of the original particle? not totally possible
    but how does McLuster deal with this changeing the 
    plummer potential? does it matter that much?
    """
    singles_in_binaries=Particles() # confusingly named. will first populate with binaries then add the single star CoM phase space data
    binary_CoMs = CoMs[:Nbin] #.random_sample(Nbin); 
    #### note, removed random sample so that secondary masses are added in the same order they were to compute the plummer profile.

    n = 0
    for CoM in binary_CoMs:
        m1 = primary_masses[n] | units.MSun #binary.mass
        m2 = secondary_masses[n] | units.MSun


        P = random_lognormal_period(Pmin=P_min, Pmax=1e+9 | units.day)
        a = calc_a(P, m1+m2)

        if e_dist=="thermal":
            e = random_ecc_thermal()
        if e_dist=="uniform":
            e = np.random.rand()
        if e_dist=="circ":
            e = 0
        
        # create an orbiting binary centered at 0
        bin_with_orbit = new_binary_orbit(m1, m2, a, e)

        # move the CM position to that of the original particle
        bin_with_orbit.position += CoM.position
        bin_with_orbit.velocity += CoM.velocity

        # append to "singles_in_binaries"
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


#------------------------------------------------#
#               Main program below               #
#------------------------------------------------#

## change these things
nstars = 10000
M_max = 0.8
P_min = 0.2 | units.day
virial_radius = 3.9 | units.parsec
cutoff_radius = 5 # in units of virial radii
fbin = 0.5
e_dist="thermal" # or "uniform" or "circ"
filename="clust_N1e4_fbin05"


# create the cluster
Nstart = int(nstars - (fbin*nstars)/2) # generate an appropriate number of starting particles so that adding binaries brings us to nstars.

primary_masses = gen_kroupa_IMF(m_max = M_max, N=Nstart)

if fbin>0.0:
    nbin = int(fbin*nstars / 2)
    secondary_masses, total_masses = add_secondary_masses(primary_masses, nbin)
    init_cluster = create_plummer_cluster(masses = total_masses, N = Nstart,
                                          radius=virial_radius, r_cutoff=cutoff_radius)
    print("generating "+str(nbin)+" primordial binaries")
    cluster = add_secondaries(init_cluster,
                              primary_masses=primary_masses, secondary_masses=secondary_masses,
                              Nbin=nbin, P_min=P_min,
                              e_dist=e_dist)
    # print("minimum x coordinate after adding binaries", min(cluster.x.value_in(units.pc)))

else:
    print("no primordial binaries")
    cluster = create_plummer_cluster(masses=primary_masses, N=Nstart, 
                                     radius=virial_radius, r_cutoff=cutoff_radius)


hmr = calculate_half_mass_radius(cluster)

x_vals = cluster.x.value_in(units.parsec)
# print("minimum x value in x_vals", min(x_vals))
y_vals = cluster.y.value_in(units.parsec)
z_vals = cluster.z.value_in(units.parsec)

vx_vals = cluster.vx.value_in(units.kms)
vy_vals = cluster.vy.value_in(units.kms)
vz_vals = cluster.vz.value_in(units.kms)

m_vals = cluster.mass.value_in(units.MSun)

save_data(m_vals, x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, filename=filename+".dat.10")

# write to an explain-y file
lines = [
    "nstars           : "+str(nstars),
    "vir. radius      : "+str(virial_radius.value_in(units.parsec))+" pc",
    "cutoff radius    : "+str(cutoff_radius)+" virial radii",
    "M_max            : "+str(M_max)+" Msun",
    "Pmin             : "+str(P_min.value_in(units.day))+" day",
    "ecc. dist.       : "+e_dist,
    "fbin             : "+str(fbin),
    "half-mass radius : "+str(hmr)+" pc", 
]
output_file = filename+".txt"

with open(output_file, "w") as file:
    for line in lines:
        file.write(line + "\n")

print(f"IC info printed to {output_file}")
