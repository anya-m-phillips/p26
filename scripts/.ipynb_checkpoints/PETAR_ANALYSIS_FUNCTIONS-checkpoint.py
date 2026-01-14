import petar
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.table import Table

import galpy
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit

import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianRepresentation, CartesianDifferential
from astropy.coordinates import SkyCoord

import astropy.units as u
import numpy as np
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms

from gala.units import galactic

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

# import sys

# sys.path.append('/Users/anyaphillips/Desktop/harvard/streams/from_cluster/scripts')
# from analyze_petar import analyze_petar
import astropy.constants as const
from streamframe import StreamFrame
from scipy.stats import binned_statistic
import scipy.stats as stats
from scipy.optimize import curve_fit
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import binom
from tqdm import tqdm
 
##### FUNCTIONS
def make_plot_nice(ax):
    """
    """    
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')  # or 'bottom'                                                    
    ax.tick_params(direction='in',axis='both',which='minor',length=3,width=1,labelsize=15)
    ax.tick_params(direction='in',axis='both',which='major',length=6,width=1,labelsize=15)
    ax.minorticks_on()  
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    return


def calc_P(a, Mtot):
    """
    period from semimajor axis, make the inputs astropy unit quantities
    """
    P = 2*np.pi * np.sqrt((a**3)/(const.G * Mtot))
    return P

def draw_inclinations(n):
    """
    draw n isotropic inclinations
    """
    theta = np.arccos(1-2*np.random.rand(n))
    return theta

def calc_amplitude(binary,inclination=np.pi/2):
    """
    calculate the velocity amplitude
    in m/s for a petar binary object assuming some inclination
    (sin(i) = 1)
    """
    Mtot = (binary.mass * u.Msun)#.value
    M2 = min([binary.p1.mass, binary.p2.mass]) * u.Msun

    a = (binary.semi * u.pc).to(u.m)#.value
    e = binary.ecc
    P = calc_P(a, Mtot).to(u.s)

    a1 = (M2/Mtot) * a

    num = 2*np.pi*a1.value *np.sin(inclination)# nominally this would include sin(i)
    denom = P.value * np.sqrt(1-(e**2))
    K = num/denom # velocity semiamplitude - multiply by 2 for full amplitude.
    return K #2*K

def prog_position(init_displacement, i):
        """
        integrate the progenitor orbit up to this point
        init displacement should be in kpc, **km/s (not with astropy units)
        using Gala and the Bovy2014 milky way potential.
        returns (x, y, z, vx, vy, vz) in kpc, km/s
        """
        pos0 = init_displacement[:3] * u.kpc 
        vel0 = init_displacement[3:] * u.km/u.s
        w0 = gd.PhaseSpacePosition(pos0, vel0)
        mwp = gp.BovyMWPotential2014(units=galactic)
        H = gp.Hamiltonian(mwp)
        int_time = i
        dt = 1
        orbit = H.integrate_orbit(w0, dt=dt, n_steps = int_time/dt)

        x = orbit.pos.x[-1].to(u.kpc)
        y = orbit.pos.y[-1].to(u.kpc)
        z = orbit.pos.z[-1].to(u.kpc)
        vx = orbit.vel.d_x[-1].to(u.km/u.s)
        vy = orbit.vel.d_y[-1].to(u.km/u.s)
        vz = orbit.vel.d_z[-1].to(u.km/u.s)
        return np.array([x.value, y.value, z.value, vx.value, vy.value, vz.value])

def load_particle(path, i, interrupt="bse"):
    file = path+"data."+str(i)
    particle = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
    particle.loadtxt(file, skiprows=1)
    return particle
    
def load_tidal(path):
    tidal = petar.external.Tidal()
    tidal.loadtxt(path+"data.tidal")
    return tidal

def is_dissolved(path, i, threshold=100, interrupt="bse"):
    """
    check whether the simulated cluster located at path
    has dissolved at timestep i
    dissolved is defined as fewer than [threshold] stars
    within the tidal radius.
    """
    # load tidal data
    tidal = load_tidal(path)
    n_bound = int(tidal.n[i])
    
    if n_bound<threshold:
        return True
    else:
        return False
    
def load_coords(path, i, return_all_particles=False, return_single_particles=False, interrupt="bse",
                 tdis_estimate = 3000, use_prog=True, init_displacement=None):
    """
    load stream frame coordinates using jake's package
    path is a path to petar output files directory
    i is the output number we care about
    return all particles and return all single particles lets you return those 
    petar objects as well as their stream coordinates
    returns: [all_particles], binaries, [singles], coords, single_coords, binary_coords
    """

    file = path+"data."+str(i)

    dis = is_dissolved(path, i, interrupt=interrupt) # threshold==100
    
    if dis==False:
        ### if the cluster is not dissolved, use the current core position to do the xform
        core = petar.Core(interrupt_mode=interrupt, external="galpy")
        core.loadtxt(path+"data.core")

        core_pos = core.pos[i] * u.pc.to(u.kpc) # dimensionless core position in kpc
        core_vel = core.vel[i] * (u.pc/u.Myr).to(u.kpc/u.Myr) # dimensionless core velocity in kpc/myr
        core_w_jake_units = np.hstack([core_pos, core_vel])
        prog_w_jake_units = core_w_jake_units


    if dis==True: # if dissolved, option to use progenitor or use last core position
        if use_prog==False:
            # if the cluster _is_ dissolved, integrate the core from estimated dissolution
            # time to now
            core = petar.Core(interrupt_mode=interrupt, external="galpy")
            core.loadtxt(path+"data.core")

            core_pos = core.pos[tdis_estimate] * u.pc.to(u.kpc) # dimensionless core position in kpc
            core_vel = core.vel[tdis_estimate] * (u.pc/u.Myr).to(u.km/u.s) # dimensionless core velocity in kpc/myr

            # use prog_position function to integrate from final core position to now
            ### note prog_position requires dimensionless init_displacement in kpc, km/s
            new_init_displacement = np.hstack([core_pos, core_vel])
            time_to_integrate = int(i-tdis_estimate)
            prog_w = prog_position(init_displacement=new_init_displacement, i=time_to_integrate)

            # transform to jake units
            prog_pos = prog_w[:3] * (u.kpc)
            prog_vel = prog_w[3:] * (u.km/u.s)
            # concatenate. 
            prog_w_jake_units = np.hstack([
                prog_pos.value, prog_vel.to(u.kpc/u.Myr).value
            ])

        if use_prog==True:
            prog_w = prog_position(init_displacement=init_displacement, i=i)
            # transform to jake units
            prog_pos = prog_w[:3] * (u.kpc)
            prog_vel = prog_w[3:] * (u.km/u.s)
            # concatenate. 
            prog_w_jake_units = np.hstack([
                prog_pos.value, prog_vel.to(u.kpc/u.Myr).value
            ])       



    header = petar.PeTarDataHeader(path+'data.'+str(i), external_mode='galpy')
    all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')

    all_particles.loadtxt(file, skiprows=1)
    CM_pos = (header.pos_offset*u.pc).to(u.kpc)
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.kpc/u.Myr)
    CM_w = np.hstack([CM_pos.value, CM_vel.value])


    all_pos = (all_particles.pos*u.pc).to(u.kpc) + CM_pos
    all_vel = (all_particles.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + CM_vel
    all_w = np.hstack([all_pos.value, all_vel.value])


    stream_obj = StreamFrame(sim_coords = all_w, prog_sim_coord = prog_w_jake_units)#CM_w)
    coords = stream_obj.GetStreamFrame()

    # load binaries
    binary_file = path+"data."+str(i)+".binary"
    binaries = petar.Binary(member_particle_type=petar.Particle,
                            G=petar.G_MSUN_PC_MYR, interrupt_mode=interrupt,
                            external_mode="galpy")

    binaries.loadtxt(binary_file, skiprows=1)

    # load singles
    single_file = path+"data."+str(i)+".single"
    singles = petar.Particle(interrupt_mode=interrupt,
                                    external_mode='galpy')

    singles.loadtxt(single_file, skiprows=1)

    core_data = petar.Core(interrupt_mode=interrupt, external_mode='galpy')
    
    core_data.loadtxt(path+'data.core')
    core_pos = (core_data.pos[i]*u.pc).to(u.kpc)
    core_vel = (core_data.vel[i]*(u.pc/u.Myr)).to(u.kpc/u.Myr)
    core_w = np.hstack([core_pos.value, core_vel.value])

    single_pos = (singles.pos*u.pc).to(u.kpc) + core_pos
    single_vel = (singles.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + core_vel
    single_w = np.hstack([single_pos.value, single_vel.value])

    binary_pos = (binaries.pos*u.pc).to(u.kpc) + core_pos
    binary_vel = (binaries.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + core_vel
    binary_w = np.hstack([binary_pos.value, binary_vel.value])

    single_stream_obj = StreamFrame(sim_coords = single_w, prog_sim_coord = prog_w_jake_units)#core_w)
    single_coords = single_stream_obj.GetStreamFrame()
    # plt.scatter(single_coords['phi1'], single_coords['phi2'], s=.001, c='cornflowerblue')


    binary_stream_obj = StreamFrame(sim_coords = binary_w, prog_sim_coord = prog_w_jake_units)#core_w)
    binary_coords = binary_stream_obj.GetStreamFrame()
    if return_all_particles==False:
        if return_single_particles==False:
            return binaries, coords, single_coords, binary_coords
        if return_single_particles==True:
            return binaries, singles, coords, single_coords, binary_coords
    if return_all_particles==True:
        if return_single_particles==False:
            return all_particles, binaries, coords, single_coords, binary_coords
        if return_single_particles==True:
            return all_particles, binaries, singles, coords, single_coords, binary_coords

def CM_to_galcen_frame(path, particles, i):
    header = petar.PeTarDataHeader(path+"data."+str(i), external_mode='galpy')
    CM_pos = header.pos_offset*u.pc
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.km/u.s)
    # add center of mass position / move to galctocentric frame
    all_pos = (particles.pos*u.pc) + CM_pos
    all_vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s) + CM_vel
    
    return all_pos.to(u.pc), all_vel.to(u.km/u.s)

def core_to_galcen_frame(path, particles, i, interrupt="bse"):
    pos = particles.pos*u.pc
    vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s)
    
    core = petar.Core(interrupt_mode=interrupt, external="galpy")
    core.loadtxt(path+"data.core")
    core_pos = core.pos[i] * u.pc
    core_vel = (core.vel[i] * (u.pc/u.Myr)).to(u.km/u.s)
    
    pos+=core_pos
    vel+=core_vel
    return pos.to(u.pc), vel.to(u.km/u.s)

def xform_to_core_frame(path, particles, i, interrupt="bse"):
    """
    pass a petar.Particles object and the corresponding i sim timestep
    returns w (phase space position in core frame)
    """
    header = petar.PeTarDataHeader(path+"data."+str(i), external_mode='galpy')
    CM_pos = header.pos_offset*u.pc
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.km/u.s)
    # add center of mass position / move to galctocentric frame
    all_pos = (particles.pos*u.pc) + CM_pos
    all_vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s) + CM_vel

    # subtract core position / move to core frame
    core = petar.Core(interrupt_mode=interrupt, external="galpy")
    core.loadtxt(path+"data.core")
    
    core_pos = core.pos[i] *u.pc
    core_vel = (core.vel[i] * (u.pc/u.Myr)).to(u.km/u.s)

    pos_rel = all_pos - core_pos
    vel_rel = all_vel - core_vel
    
    xrel, yrel, zrel = pos_rel.T
    rrel = np.sqrt(xrel**2 + yrel**2 + zrel**2)
    
    return pos_rel.to(u.pc), vel_rel.to(u.km/u.s), rrel.to(u.pc)

def clip_outside_rtid(path, particles, i, interrupt="bse", in_core_frame=True):
    if not in_core_frame:
        pos_rel, vel_rel, rrel = xform_to_core_frame(path, particles, i, interrupt)
        
    if in_core_frame:
        pos_rel, vel_rel = particles.pos*u.pc, particles.vel*(u.km/u.s)
        x, y, z = pos_rel.T
        rrel = np.sqrt(x**2 + y**2 + z**2)
        
    tidal = load_tidal(path)
    rtid = tidal.rtid[i] * u.pc
    
    clip = rrel<=rtid
    return rrel, clip
    
    

def compute_cluster_structure(path, i_list, init_displacement, interrupt="bse", save=False, savepath=None, density_shell_width=0.15):
    """
    i_list, init_displacement should be np arrays
    path should be a string.
    return an array of half mass radii over i_list
    array of nbounds over i_list
    dissolution time (last timestep at which nbound >100 stars
    """
    hmrs = []
    nbounds = []
    mass_bounds = []
    t_rhs = []
    
    tidal = petar.external.Tidal()
    tidal.loadtxt(path+"data.tidal")
    rtid = tidal.rtid * u.pc
    for j in tqdm(range(len(i_list))):
        i = i_list[j]
        file = path+"data."+str(i)
        # if bse==True:
        all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
        all_particles.loadtxt(file, skiprows=1)
        
        
        pos_rel, vel_rel, rrel = xform_to_core_frame(path, all_particles, i, interrupt=interrupt)
        in_rtid = rrel.to(u.pc) <= rtid[i]
        
        nbound = len(rrel[in_rtid])
        nbounds.append(nbound)
        
        mass_bound = np.sum(all_particles.mass[in_rtid] * u.Msun)
        mass_bounds.append(mass_bound.to(u.Msun).value)
        
        if nbound<=100:
            hmrs.append(0.)
            t_rhs.append(0.)
            
        else:
            hmr = calculate_half_mass_radius(all_particles.mass[in_rtid]*u.Msun, rrel[in_rtid].to(u.pc)) * u.pc
            hmrs.append(hmr.to(u.pc).value)
                        
            shell_width = density_shell_width*u.pc
            inner = hmr-shell_width
            outer = hmr+shell_width
            in_shell = (rrel>inner) & (rrel<outer)
            
            shell_volume = (4*np.pi/3) * (outer**3 - inner**3)
            shell_mass = np.sum(all_particles.mass[in_shell]) * u.Msun
            shell_density = shell_mass / shell_volume
            
            # print(rho_avg.decompose().unit)
            
            t_cross = (const.G * shell_density)**(-1/2)
            # print(t_cross.decompose().unit)
            t_rh = ((0.1*nbound)/(np.log(nbound))) * t_cross
            t_rhs.append(t_rh.to(u.Myr).value)
       
    hmrs = np.array(hmrs)
    nbounds = np.array(nbounds)
    mass_bounds = np.array(mass_bounds) # in msun
    t_rhs = np.array(t_rhs)
    # indices at which nbounds>=100:
    bound_indices = i_list[nbounds>=100]
    t_dis = np.array([max(bound_indices)])
    
    if save==True:
        np.savez(savepath,
                 i_list = i_list,
                 hmrs = hmrs,
                 nbounds = nbounds,
                 mass_bounds = mass_bounds,
                 t_rhs = t_rhs,
                 t_dis = t_dis,
                 allow_pickle=True)
    
    return hmrs, nbounds, mass_bounds, t_rhs, t_dis
    
    
######
# def find_dissolution_time_coarse(path, threshold=100, start=0, end=10000, grain=10, interrupt="bse"):
#     times_to_check = np.arange(start, end+grain, grain)
#     for i in tqdm(times_to_check):
#         if is_dissolved(path, i, threshold):
#             if is_dissolved(path, i+grain, threshold, interrupt):
#                 return i
def find_dissolution_time(path, threshold=100, interrupt="bse"):
    tidal = load_tidal(path)
    bound_times = tidal.time[tidal.n>=threshold]
    tdis = int(max(bound_times))
    return tdis


    
def load_coords_new(path, i, dissolved_threshold=100, interrupt="bse"):
    """
    under construction
    checks that the core is well-defined
    then loads coordinates as I would have before
    but using the PeTar core position
    rather than the progenitor
    orbit positon
    
    returns all_particles, binaries, singles, coords, single_coords, binary_coords
    """
    file = path+"data."+str(i)
    
    ### cluster position
    header = petar.PeTarDataHeader(file, external_mode="galpy")
    CM_pos = (header.pos_offset*u.pc).to(u.kpc)
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.kpc/u.Myr)
    CM_w = np.hstack([CM_pos.value, CM_vel.value])
    all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
    all_particles.loadtxt(file, skiprows=1)
    
    dissolved = is_dissolved(path, i, threshold=dissolved_threshold, interrupt=interrupt)
    
    if dissolved==True:
        raise Exception("cluster is dissolved -- use load_coords() instead? or @anya update this function")
    # else:
    
    # subtract core position / move to core frame
    core = petar.Core(interrupt_mode=interrupt, external="galpy")
    core.loadtxt(path+"data.core")
    
    core_pos = core.pos[i] * u.pc.to(u.kpc) # dimensionless core position in kpc
    core_vel = core.vel[i] * (u.pc/u.Myr).to(u.kpc/u.Myr) # dimensionless core velocity in kpc/myr
    core_w_jake_units = np.hstack([core_pos, core_vel])
    
    all_pos = (all_particles.pos*u.pc).to(u.kpc) + CM_pos
    all_vel = (all_particles.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + CM_vel
    all_w = np.hstack([all_pos.value, all_vel.value]) # in jake units
    
    # initialize the stream object
    stream_obj = StreamFrame(sim_coords = all_w, prog_sim_coord = core_w_jake_units)
    coords = stream_obj.GetStreamFrame()
    
    # singles - these load in the core frame
    single_file = path+"data."+str(i)+".single"
    singles = petar.Particle(interrupt_mode=interrupt, external_mode="galpy")


    singles.loadtxt(single_file, skiprows=1)
    single_pos = singles.pos*u.pc.to(u.kpc) + core_pos
    single_vel = singles.vel*(u.pc/u.Myr).to(u.kpc/u.Myr) + core_vel
    single_w = np.hstack([single_pos, single_vel])
    
    single_stream_obj = StreamFrame(sim_coords = single_w, prog_sim_coord = core_w_jake_units)
    single_coords = single_stream_obj.GetStreamFrame()
    
    # binaries - load in core frame
    binary_file = path+"data."+str(i)+".binary"
    binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR, 
                            interrupt_mode=interrupt, external_mode='galpy')
    binaries.loadtxt(binary_file, skiprows=1)
    binary_pos = binaries.pos*u.pc.to(u.kpc)+core_pos
    binary_vel = binaries.vel*(u.pc/u.Myr).to(u.kpc/u.Myr)+core_vel
    binary_w = np.hstack([binary_pos, binary_vel])
    
    binary_stream_obj = StreamFrame(sim_coords=binary_w, prog_sim_coord=core_w_jake_units)
    binary_coords = binary_stream_obj.GetStreamFrame()
    
    return all_particles, binaries, singles, coords, single_coords, binary_coords   
    


def calc_dispersion(x):
    x2 = x**2
    meanx = np.mean(x)
    meanx2 = np.mean(x2)
    sig2 = meanx2 - (meanx**2)
    return np.sqrt(sig2)

def calc_dispersion_with_3sig_clip(x):
    x2 = x**2
    meanx = np.mean(x)
    meanx2 = np.mean(x2)
    sig2 = meanx2 - (meanx**2)
    sig = np.sqrt(sig2)
    
    clip = np.abs(x-meanx)<=3*sig
    x_new = x[clip]
    x2_new = x_new**2
    meanx_new = np.mean(x_new)
    meanx2_new = np.mean(x2_new)
    sig2_new = meanx2_new - (meanx_new**2)
    return np.sqrt(sig2_new)

def trim_stream(phi1, phi2, r=None, r_apo=None, f_1=3, f_2=3, f_r=1):
    """
    get rid of outliers from the main stream with ...
    sigma clipping?
    """
    mean_phi1 = np.mean(phi1)
    mean_phi2 = np.mean(phi2)
    
    std_phi1 = calc_dispersion(phi1) #np.std(phi1)
    std_phi2 = calc_dispersion(phi2) #np.std(phi2)
    if r==None:
        clip = (np.abs(phi1)<=f_1*std_phi1) & (np.abs(phi2)<=f_2*std_phi2)
    if r!=None:
        clip = np.abs(phi1<=f_1*std_phi1) & (np.abs(phi1<=f_2*std_phi2)) & (r<=f_r*r_apo)
    
    new_phi1, new_phi2 = phi1[clip], phi2[clip]
    return new_phi1, new_phi2, clip # which stars to keep indexed
    


def straighten_stream_polynomial(phi1,phi2, degree=5, show_plot=False, return_poly_fn=False):
    """
    straighten a streamframe object's phi1/phi2 coordinates 
    by fitting a high-order polynomial
    """
    def make_polynomial(order):
        # Create the argument list: a0, a1, ..., an
        arg_list = ", ".join([f"a{i}" for i in range(order + 1)])

        # Build the function definition string
        func_str = f"def poly(x, {arg_list}):\n"
        func_str += "    return " + " + ".join([f"a{i}*x**{i}" for i in range(order + 1)]) + "\n"

        # Local namespace for exec
        local_ns = {}
        exec(func_str, {}, local_ns)
        return local_ns['poly']
    
    poly=make_polynomial(order=degree)

    # next fit a quadratic and subtract to straighten out
    fit, cov = curve_fit(f=poly, xdata=phi1, ydata=phi2)
    new_phi2 = phi2 - poly(phi1, *fit)

    if show_plot==True:
        fig, [ax1,ax2] = plt.subplots(2,1,figsize=[10,20])
        x = np.linspace(min(phi1), max(phi1))
        y = poly(x, *fit)
        ax1.scatter(phi1, phi2, s=.1)
        ax1.plot(x, y) 
                                    
        ax2.scatter(phi1, new_phi2, s=.1)
        ax2.set_aspect('equal')

    if return_poly_fn==False:
        return phi1, new_phi2
    if return_poly_fn==True:
        return phi1, new_phi2, poly, fit

def get_ICRS_coords(path, i, interrupt="bse"):
    """
    return ICRS coordinates object
    provide path to PeTar data.# files and i
    """
    # load coordinates
    file = path+'data.'+str(i)
    header = petar.PeTarDataHeader(path+'data.'+str(i), external_mode='galpy')
    all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
    all_particles.loadtxt(file, skiprows=1)
    pos = all_particles.pos * u.pc
    vel = all_particles.vel * (u.pc/u.Myr)
    CM_pos = header.pos_offset*u.pc
    CM_vel = header.vel_offset * (u.pc/u.Myr)
 
    # transform out of CM frame
    pos+=CM_pos
    vel+=CM_vel

    # create a cartesian representation/differential
    rep = CartesianRepresentation(pos.T)
    rep_vel = CartesianDifferential(vel.T)
    rep = rep.with_differentials(rep_vel)

    # create coords in galactocentric frame
    galcen_frame = Galactocentric()
    coords_galcen = SkyCoord(rep, frame=galcen_frame)

    # transform to ICRS
    coords_ICRS = coords_galcen.transform_to(ICRS())
    return coords_ICRS

def get_streamcoords_from_ICRS(ICRS_coords, try_every=1):
    """
    get phi1, phi2 from transforming to ICRS coordinates, fitting a great circle coordinate frame
    then fitting a quadratic, cubic, or spline and subtracting it. 
    for finding stream widths; progenitor will not necessarily be centered.

    returns gc_frame_best, xformed_coords
    """
    obfun_best = np.inf
    gc_frame_best = None
    point1_best = None
    point2_best = None

    point1 = SkyCoord(
        ra = np.nanmedian(ICRS_coords.ra),
        dec = np.nanmedian(ICRS_coords.dec),
        frame='icrs'
    )
    
    print("iterating to find best frame to minimize phi2")
    for i in tqdm(range(0, len(ICRS_coords), try_every)):
        point2 = ICRS_coords[i]

        try:
            gc_frame = gc.GreatCircleICRSFrame.from_endpoints(point1, point2)
            stars_sc_gc = ICRS_coords.transform_to(gc_frame)
            obfun = np.sum(stars_sc_gc.phi2.value**2)

            if obfun < obfun_best:
                obfun_best = obfun # update objective function
                gc_frame_best = gc_frame
                point2_best = point2

        except Exception as e:
            continue # skipped failed point2s
    
    xformed_coords = ICRS_coords.transform_to(gc_frame_best)
    return gc_frame_best, xformed_coords



def calculate_half_mass_radius(ms, rs):
    """
    this is for checking the half mass radius
    for sanity purposes
    units of ms don't matter, will return r_h in whatever units "rs" is in
    """
    radii = rs
    masses = ms

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


def rho_plummer(M, r, a):
    """
    return the density of a cluster of mass M 
    with plummer parameter a
    at radius r
    """
    return 3*M/(4*np.pi*a**3) * (1 + (r/a)**2)**(-5/2)


def t_cross(rho):
    """
    return the dynamical timescale at local density rho (input as astropy quantity)
    """
    return 1/np.sqrt(const.G * rho)

def t_relax(N, rho):
    """
    return the relaxation timescale for N particles
    with local density rho (use astropy quantity)
    """
    return (0.1 * N / np.log(N)) * (1/np.sqrt(const.G * rho))

def draw_inclinations(n):
    """
    draw n isotropic inclinations
    """
    theta = np.arccos(1-2*np.random.rand(n))
    return theta


def compute_ahs(path, i, shell_width=.3*u.pc, interrupt="bse"):
    # singles - these load in the core frame
    single_file = path+"data."+str(i)+".single"
    singles = petar.Particle(interrupt_mode=interrupt, external_mode="galpy")
    singles.loadtxt(single_file, skiprows=1)
    single_pos = singles.pos*u.pc
    single_vel = (singles.vel*(u.pc/u.Myr)).to(u.km/u.s)
    sx, sy, sz = single_pos.T
    sr = np.sqrt(sx**2 + sy**2 + sz**2)
    svx, svy, svz = single_vel.T
    sv = np.sqrt(svx**2 + svy**2 + svz**2)
    single_masses = singles.mass * u.Msun
    
    
    # binaries - load in core frame
    binary_file = path+"data."+str(i)+".binary"
    binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR, 
                            interrupt_mode=interrupt, external_mode='galpy')
    binaries.loadtxt(binary_file, skiprows=1)
    binary_pos = binaries.pos*u.pc
    binary_vel = (binaries.vel*(u.pc/u.Myr)).to(u.km/u.s)
    bx, by, bz = binary_pos.T
    br = np.sqrt(bx**2 + by**2 + bz**2)
    bvx, bvy, bvz = binary_vel.T
    bv = np.sqrt(bvx**2 + bvy**2 + bvz**2)
    binary_masses = binaries.mass * u.Msun
    
    all_r = np.concatenate([sr, br])
    all_v = np.concatenate([sv, bv])
    all_m = np.concatenate([single_masses, binary_masses])
    
    tidal = petar.external.Tidal()
    tidal.loadtxt(path+"data.tidal")
    rtid = tidal.rtid[i] * u.pc
    
    in_rtid_all = all_r<=rtid
    in_rtid_binary = br<=rtid
    
    r_in = all_r[in_rtid_all]
    v_in = all_v[in_rtid_all]
    m_in = all_m[in_rtid_all]
    
    def find_locally_averaged_m_sigma2(r, all_m, all_r, all_v, shell_width): # .2 pc
        outer = r+shell_width
        if r<shell_width:
            inner=0
            
        if r>=shell_width:
            inner = r-shell_width
        
        in_shell = (all_r>=inner) & (all_r<outer)
        r_in_shell = all_r[in_shell]
        m_in_shell = all_m[in_shell]
        Mtot_in_shell = np.sum(m_in_shell)
        n_in_shell = len(m_in_shell)
        Mavg = Mtot_in_shell / n_in_shell
        
        v_in_shell = all_v[in_shell]
        v2_in_shell = v_in_shell**2
        sigma2 = np.mean(v2_in_shell) - np.mean(v_in_shell)**2
        
        # print(Mavg.unit, sigma2.unit)
        return Mavg, sigma2
    

    
    # stats on the binaries
    m1s = binaries.p1.mass[in_rtid_binary] * u.Msun
    m2s = binaries.p2.mass[in_rtid_binary] * u.Msun
    brs = br[in_rtid_binary]
    semis = binaries.semi[in_rtid_binary] * u.pc
    Mtots = binaries.mass[in_rtid_binary] * u.Msun
    
    
    ahs_list = []
    for k in range(len(brs)):
        br = brs[k]
        
        Mavg, sigma2 = find_locally_averaged_m_sigma2(br, all_m=m_in, all_r=r_in, all_v=v_in, shell_width=shell_width)
        
        m1 = m1s[k]
        m2 = m2s[k]
        
        ahs = const.G * m1 * m2 / (Mavg * sigma2)
        ahs_list.append(ahs.to(u.pc).value)
        
    ahs_list = np.array(ahs_list) * u.pc
    
    rh = calculate_half_mass_radius(m_in.to(u.Msun), r_in.to(u.pc)) * u.pc
    
    return ahs_list, brs, semis, Mtots, rh, in_rtid_binary

def create_binary_matrix(path, i_list, save=False, savepath=None,
                        verbose=False):
    """
    this version does not include phi1
    to-do:
        * include matrices of m1 and m2 type and mass
        * ?? option to compute phi1 (will just be more expensive)
        
    """
    periods_arr = []
    phi1s_arr = [] # the phi1
    avals_arr = [] # semimajor axes
    evals_arr = [] # eccentricities


    m1s_arr = []
    m2s_arr = []
    m1_ids_arr = []
    m2_ids_arr = []
    if verbose==True:
        print("looping through i list")
    for j in tqdm(range(len(i_list))):
        i=i_list[j]

        binary_file = path+"data."+str(i)+".binary"
        binaries = petar.Binary(member_particle_type=petar.Particle,
                                G=petar.G_MSUN_PC_MYR, interrupt_mode='bse',
                                external_mode='galpy')
        binaries.loadtxt(binary_file, skiprows=1)

        periods = calc_P(binaries.semi*u.pc, binaries.mass*u.Msun).to(u.yr).value
        avals = binaries.semi # pc
        evals = binaries.ecc
        # phi1s = binary_coords['phi1'] # degrees

        m1_ids = []
        m2_ids = []
        m1s = []
        m2s = []
        for binary in binaries: #### WE HAVE LOOPED THROUGH BINARY IN BINARIES --- PRESERVES INITIAL ORDERING OF PRIMORDIAL BINARIES AT T=0 I THINK!!!
            m1_ids.append(binary.p1.id)
            m1s.append(binary.p1.mass)
            m2_ids.append(binary.p2.id)
            m2s.append(binary.p2.mass)

        m1_ids_arr.append(m1_ids)
        m2_ids_arr.append(m2_ids)
        m1s_arr.append(m1s)
        m2s_arr.append(m2s)
        avals_arr.append(avals)
        evals_arr.append(evals)
        periods_arr.append(periods)    
        
    #### collect unique IDs: (unique pairs of stars in binaries, with IDs sorted)
    all_binary_IDs = []
    for m1_list, m2_list in zip(m1_ids_arr, m2_ids_arr): # iterate through time steps
        for id1, id2 in zip(m1_list, m2_list): # iterate through binaries at this timestep
            all_binary_IDs.append(tuple(sorted((id1,id2))))

    unique_binary_ids = np.unique(all_binary_IDs, axis=0) # shape=(n_binaries, 2) where n_binaries is over all time
    binary_id_tuples = [tuple(pair) for pair in unique_binary_ids] # type change
    n_binaries = len(binary_id_tuples)
    n_times = len(m1_ids_arr)
    # map ids to row indexes..?
    binary_id_to_row = {bid:i for i, bid in enumerate(binary_id_tuples)} ##???
    
    array_of_bids = np.array(binary_id_tuples) # for saving purposes! 
   
    # fill period matrix, a matrix, e matrix
    period_matrix = np.full((n_binaries, n_times), np.nan)
    # phi1_matrix = np.full((n_binaries, n_times), np.nan)
    a_matrix = np.full((n_binaries, n_times), np.nan)
    e_matrix = np.full((n_binaries, n_times), np.nan)
    
    m1_matrix = np.full((n_binaries, n_times), np.nan)
    m2_matrix = np.full((n_binaries, n_times), np.nan)

    m1_ID_matrix = np.full((n_binaries, n_times), np.nan)
    m2_ID_matrix = np.full((n_binaries, n_times), np.nan)

    ### for each time, id of everything we just pulled out
    for t, (m1_ids, m2_ids, periods, avals, evals, m1s, m2s) in enumerate(zip(m1_ids_arr, m2_ids_arr, periods_arr, avals_arr, evals_arr, m1s_arr, m2s_arr)):
        ## at this time, for each binary
        for id1, id2, period, a, e, m1, m2 in zip(m1_ids, m2_ids, periods, avals, evals, m1s, m2s):
            # construct binary_id to match to a unique one from the mapping by sorting m1s and m2s ids:
            binary_id = tuple(sorted((id1, id2)))
            row = binary_id_to_row[binary_id] # match this to a row number in the unique binary tuples
            ##### now place binary properties at the corresponding matrix locations by [binary row, time]
            period_matrix[row, t] = period 
            # phi1_matrix[row, t] = phi1
            a_matrix[row, t] = a
            e_matrix[row, t] = e
            m1_matrix[row, t] = m1
            m2_matrix[row, t] = m2
            #### to be able to check the types by referencing particles data @ this time
            m1_ID_matrix[row, t] = int(id1)
            m2_ID_matrix[row, t] = int(id2)
            
            
    m1_ID_matrix = np.nan_to_num(m1_ID_matrix, nan=-1).astype(int)
    m2_ID_matrix = np.nan_to_num(m2_ID_matrix, nan=-1).astype(int)
    if save==True:
        np.savez(savepath, 
                 i_list=i_list,
                 period_matrix_yr = period_matrix,
                 a_matrix_au = a_matrix,
                 e_matrix = e_matrix,
                 m1_matrix = m1_matrix,
                 m2_matrix = m2_matrix, 
                 m1_ID_matrix = m1_ID_matrix,
                 m2_ID_matrix = m2_ID_matrix,
                 binary_IDs = array_of_bids,
                 allow_pickle=True)
                 
    return period_matrix, a_matrix, e_matrix, m1_matrix, m2_matrix, m1_ID_matrix, m2_ID_matrix, array_of_bids


# plotting functions

i_list1 = np.array([0, 3095, 4080, 5060, 6045, 7020, 8020, 8995,9980])
i_list2 = np.array([2600,3580, 4565, 5555,6565,7545, 8545, 9535])
peris_i_list = np.sort(np.concatenate([i_list1, i_list2]))
def make_fbin_plot_v2(paths, init_displacements, labels, n, i_list=peris_i_list, 
                      toprow_ylim=(-0.01, 0.12), 
                      bottomrow_ylim=(-0.001, 0.045),
                      nbin=20, 
                      countsel=50,
                      exclude_COs=True):
    """
    need to have defined paths, init_displacements
    """
    # ────────────────────────────── figure & axes ─────────────────────────────
    fig, ax_arr = plt.subplots(2, 3, figsize=[17, 10],
                               sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax_list = ax_arr.flatten()

    # ───────────────────────────── period ranges ──────────────────────────────
    period_ranges = [
        np.array([0, 10])      * u.yr,
        np.array([10, 1e2])    * u.yr,
        np.array([1e2, 1e3])   * u.yr,
        np.array([1e3, 1e4])   * u.yr,
        np.array([1e4, 1e5])   * u.yr,
        np.array([1e5, 2e9])   * u.yr
    ]

    # consistent colour per snapshot
    colors = plt.cm.magma_r(np.linspace(0.1, 1.0, len(i_list)))

    initial_fbins_per_period = []
    
    # ───────────────────────────── outer loop (snapshots) ────────────────────
    for j, i in tqdm(enumerate(i_list)):
        # heavy call done once
        binaries, coords, single_coords, binary_coords = load_coords(
            paths[n], i, init_displacement=init_displacements[n], use_prog=True)

        periods = calc_P(binaries.semi * u.pc,
                             binaries.mass * u.Msun).to(u.yr)

        p1_types = binaries.p1.star.type
        p2_types = binaries.p2.star.type
        type_list = [13,14,15]
        if exclude_COs==True:
            okay_p1 = (~np.isin(p1_types, type_list)) # not in type_list
            okay_p2 = (~np.isin(p2_types, type_list))# not in type_list 
            okay_type = okay_p1 & okay_p2
        if exclude_COs==False:
            okay_type = np.array([True]*len(periods))

        # ───────────────────────── inner loop (sub-plots / period bins) ──────
        for k, (ax, period_range) in enumerate(zip(ax_list, period_ranges)):
       
            # pick binaries in / out of this period range
            period_sel      = (periods >= period_range[0]) & (periods < period_range[1]) & okay_type
            not_period_sel  = ~period_sel & okay_type

            # build combined arrays
            all_phi1 = np.concatenate([
                single_coords['phi1'],
                binary_coords['phi1'][period_sel],
                binary_coords['phi1'][not_period_sel]
            ])
            all_phi2 = np.concatenate([
                single_coords['phi2'],
                binary_coords['phi2'][period_sel],
                binary_coords['phi2'][not_period_sel]
            ])
            binary_indicator = np.concatenate([
                np.zeros(len(single_coords['phi1'])),
                np.ones(len(binary_coords['phi1'][period_sel])),
                np.zeros(len(binary_coords['phi1'][not_period_sel]))
            ])

            ### save the initial binary fraction??
            if i==0:
                total_binary_fraction_periodsel = np.mean(binary_indicator)
                ax.axhline(total_binary_fraction_periodsel, c='k', linestyle="--")
                continue # never plot the initial binary fraction in phi1/phi1max (it will look dumb)

            
            # trim / straighten
            trimmed_phi1, trimmed_phi2, clip = trim_stream(all_phi1, all_phi2)
            binary_indicator = binary_indicator[clip]
            straight_phi1, _ = straighten_stream_polynomial(
                trimmed_phi1, trimmed_phi2, show_plot=False)

            phimax = np.max(straight_phi1)
            phi1_bin_edges = np.linspace(-phimax, phimax, nbin + 1)
            bin_centers = 0.5 * (phi1_bin_edges[:-1] + phi1_bin_edges[1:])

            # stats
            binary_fracs, _, _ = binned_statistic(
                straight_phi1, binary_indicator, statistic='mean',
                bins=phi1_bin_edges)
            counts, _, _ = binned_statistic(
                straight_phi1, straight_phi1, statistic='count',
                bins=phi1_bin_edges)

            # plot
            good = counts > countsel           # ignore under-populated bins
            ax.plot(bin_centers[good]/max(bin_centers),
                    binary_fracs[good],
                    color=colors[j], marker='o',
                    label=f"{i} Myr" if k == 5 else None,
                   zorder=0)  # legend once


     
    # ──────────────────────────── cosmetics per subplot ─────────────────────
    for k, ax in enumerate(ax_list):
        # paf.make_plot_nice(ax)
        ax.set_ylim(toprow_ylim if k < 3 else bottomrow_ylim)
        # ax.axhline(initial_fbins_per_period[k], c='k', linestyle="--")
        if k in [1, 2, 4, 5]:
            ax.set_yticklabels([])

    # titles • axis labels
    titles = [r"$P_{\rm{orb}} < 10$ yr",
              r"$10\leq P_{\rm{orb}} < 10^2$ yr",
              r"$10^2\leq P_{\rm{orb}} < 10^3$ yr",
              r"$10^3\leq P_{\rm{orb}} < 10^4$ yr",
              r"$10^4\leq P_{\rm{orb}} < 10^5$ yr",
              r"$P_{\rm{orb}} \geq 10^5$ yr"]
    for ax, title in zip(ax_list, titles):
        ax.text(0.5, 0.9, title, ha='center', fontsize=20,
                transform=ax.transAxes)

    fig.text(0.5, 0.04, r"$\phi_1/\phi_{1,\rm{max}}$", fontsize=30, ha='center')
    fig.text(0.07, 0.5, r"$f_{\rm{bin}}$", fontsize=30, va='center', rotation='vertical')

    # show legend only once (last axis)
    ax_list[-1].legend(bbox_to_anchor=(1.5, 0), loc='lower right')
    plt.subplots_adjust(right=0.93)
    fig.suptitle(labels[n])




def binfrac_plot_compare_sims(paths, init_displacements, labels, n_list, i_list, 
                              exclude_COs=False,
                              nbin=20,
                              countsel=50,
                              toprow_ylim=(-.01,0.1),
                              bottomrow_ylim=(-.01,0.04)
                             ):
    """
    n_list is which sims from [paths] you want to include, 
    i_list is the times at which you want to plot each of the n simulations
    (because they will have different dissolution times)
    """
    
    fig, ax_arr = plt.subplots(2, 3, figsize=[17, 10],
                               sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax_list = ax_arr.flatten()

    # ───────────────────────────── period ranges ──────────────────────────────
    period_ranges = [
        np.array([0, 10])      * u.yr,
        np.array([10, 1e2])    * u.yr,
        np.array([1e2, 1e3])   * u.yr,
        np.array([1e3, 1e4])   * u.yr,
        np.array([1e4, 1e5])   * u.yr,
        np.array([1e5, 2e9])   * u.yr
        ]
    # consistent colour per snapshot
    colors = plt.cm.copper(np.linspace(0., 1., len(i_list)))

    # ───────────────────────────── outer loop (snapshots) ────────────────────
    i_list = [[0,i] for i in i_list]
    for j, ivals in tqdm(enumerate(i_list)):
        # heavy call done once
        n=n_list[j]
        for i in ivals:
            binaries, coords, single_coords, binary_coords = load_coords(
                paths[n], i, init_displacement=init_displacements[n], use_prog=True)

            periods = calc_P(binaries.semi * u.pc,
                                 binaries.mass * u.Msun).to(u.yr)

            p1_types = binaries.p1.star.type
            p2_types = binaries.p2.star.type
            if exclude_COs==True:
                type_list = [13,14,15]
            if exclude_COs==False:
                type_list = []

            okay_p1 = (~np.isin(p1_types, type_list)) # not in type_list
            okay_p2 = (~np.isin(p2_types, type_list))# not in type_list 
            okay_type = okay_p1 & okay_p2

            # ───────────────────────── inner loop (sub-plots / period bins) ──────
            for k, (ax, period_range) in enumerate(zip(ax_list, period_ranges)):
                # pick binaries in / out of this period range
                period_sel      = (periods >= period_range[0]) & (periods < period_range[1]) & okay_type
                not_period_sel  = ~period_sel & okay_type

                # build combined arrays
                all_phi1 = np.concatenate([
                    single_coords['phi1'],
                    binary_coords['phi1'][period_sel],
                    binary_coords['phi1'][not_period_sel]
                ])
                all_phi2 = np.concatenate([
                    single_coords['phi2'],
                    binary_coords['phi2'][period_sel],
                    binary_coords['phi2'][not_period_sel]
                ])


                binary_indicator = np.concatenate([
                    np.zeros(len(single_coords['phi1'])),
                    np.ones(len(binary_coords['phi1'][period_sel])),
                    np.zeros(len(binary_coords['phi1'][not_period_sel]))
                ])
                ### first iteration around, load & find the 
                if i==0:
                    total_binfrac = np.mean(binary_indicator)
                    ax.axhline(total_binfrac, c=colors[j], linestyle="--")
                    continue


                # trim / straighten
                trimmed_phi1, trimmed_phi2, clip = trim_stream(all_phi1, all_phi2)
                binary_indicator = binary_indicator[clip]
                straight_phi1, _ = straighten_stream_polynomial(
                    trimmed_phi1, trimmed_phi2, show_plot=False)

                # plt.figure()
                # plt.scatter(trimmed_phi1, trimmed_phi2, s=.01)
                # plt.axvline(np.median(trimmed_phi1), c='k', linewidth=1)
                # plt.ylim(-2,2)

                straight_phi1-=np.median(straight_phi1)
                phimax = np.max(straight_phi1)
                phi1_bin_edges = np.linspace(-phimax, phimax, nbin + 1)
                bin_centers = 0.5 * (phi1_bin_edges[:-1] + phi1_bin_edges[1:])

                # stats
                binary_fracs, _, _ = binned_statistic(
                    straight_phi1, binary_indicator, statistic='mean',
                    bins=phi1_bin_edges)
                counts, _, _ = binned_statistic(
                    straight_phi1, straight_phi1, statistic='count',
                    bins=phi1_bin_edges)

                # plot
                good = counts > countsel           # ignore under-populated bins
                ax.plot(bin_centers[good]/max(bin_centers),
                        binary_fracs[good],
                        color=colors[j], marker='o',
                        label=labels[n] if k == 5 else None)  # legend once

    # ──────────────────────────── cosmetics per subplot ─────────────────────
    for k, ax in enumerate(ax_list):
        # paf.make_plot_nice(ax)
        ax.set_ylim(toprow_ylim if k < 3 else bottomrow_ylim)
        if k in [1, 2, 4, 5]:
            ax.set_yticklabels([])

    # titles • axis labels
    titles = [r"$P_{\rm{orb}} < 10$ yr",
              r"$10\leq P_{\rm{orb}} < 10^2$ yr",
              r"$10^2\leq P_{\rm{orb}} < 10^3$ yr",
              r"$10^3\leq P_{\rm{orb}} < 10^4$ yr",
              r"$10^4\leq P_{\rm{orb}} < 10^5$ yr",
              r"$P_{\rm{orb}} \geq 10^5$ yr"]
    for ax, title in zip(ax_list, titles):
        ax.text(0.5, 0.9, title, ha='center', fontsize=20,
                transform=ax.transAxes)

    fig.text(0.5, 0.04, r"$\phi_1/\phi_{1,\rm{max}}$", fontsize=30, ha='center')
    fig.text(0.07, 0.5, r"$f_{\rm{bin}}$", fontsize=30, va='center', rotation='vertical')

    # show legend only once (last axis)
    ax_list[-1].legend(bbox_to_anchor=(1.55, 1))
    plt.subplots_adjust(right=0.93)
    # fig.suptitle(labels[n])

    
def find_minima(arr):
    min_indeces = []
    min_values = []
    for i,a in enumerate(arr):
        # print(i)
        if i==0 or i==int(len(arr)-1):
            # print("first/last value, skipping")
            continue
        if (arr[i-1]>a) & (arr[i+1]>a):
            # print("appending a local minimum")
            min_indeces.append(i)
            min_values.append(a)

    return np.array(min_indeces), np.array(min_values)

def find_maxima(arr):
    max_indeces = []
    max_values = []
    for i, a in enumerate(arr):
        if i==0 or i==int(len(arr)-1):
            continue
        if (arr[i-1]<a) & (arr[i+1]<a):
            max_indeces.append(i)
            max_values.append(a)

    return np.array(max_indeces), np.array(max_values)

def find_dissolution_plotting_times(path, interrupt='bse'):
    # find galactocentric radius at each timestep
    core = petar.Core(interrupt_mode=interrupt, external='galpy')
    core.loadtxt(path+'data.core')
    x,y,z = core.pos.T
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # find dissolution time
    tidal = load_tidal(path)
    bound_times = tidal.time[tidal.n>=100]
    tdis = int(max(bound_times))
    
    
    # find last peri/apocenter before dissolution
    r_times_to_check = tidal.time[tdis-500:tdis].astype(int) # check last 500 Myr for a min/max r
    rs_to_check = r[r_times_to_check]
    min_inds, min_vals = find_minima(rs_to_check)
    times_at_minimum = r_times_to_check[min_inds]
    max_inds, max_vals = find_maxima(rs_to_check)
    times_at_maximum = r_times_to_check[max_inds]
    
    
    t_plot_peri = int(max(times_at_minimum))
    t_plot_apo = int(max(times_at_maximum))
    
    return tdis, t_plot_peri, t_plot_apo
    

## for uncertainties in the binary fraction. 
def Prob_of_frac(k, N, parr=np.linspace(0, 1, 1000)): # uniform prior??
    """
    return the probability mass function for p given k successes in N trials
    """
    return parr, np.array([binom.pmf(k, N, p) for p in parr]) # probability mass function

def percentile(p, Prob, percentiles=[16,50,84]):
    """
    return percentile values of the binomial dist???
    """
    cdf = np.cumsum(Prob)
    cdf /= cdf.max()
    cdf *= 100 
    values = np.interp(percentiles, cdf, p)
    return values


def make_fbin_megaplot(paths, init_displacements, apocenters, # a list of paths to simulation data
                       sim_grid,# an arrangement of the simulation indices. 1st dim goes along columns (e.g. different orbits)
                       orbit_times, colors, sim_labels,column_labels,# colors correspond to different simulations. 
                       deg_per_bin=2,
                       ylim_frac=2,
                       period_ranges = [np.array([0,1e2])*u.yr,np.array([1e2,1e4])*u.yr,np.array([1e4, 1e100000])*u.yr],
                       figsize=[20,20],
                       save=False, title=None, skipped_n=None):
    nsims_per_column = len(sim_grid[0])
    fig, axs_grid = plt.subplots(1+nsims_per_column+len(period_ranges), len(sim_grid), figsize=figsize, sharex=True)
    
    plt.subplots_adjust(wspace=0.05)
    axs_grid = np.array(axs_grid)
    if axs_grid.ndim == 1:
        # if it's (N,) that means one column; convert to (N,1) so axs_grid[:, ii] is safe
        axs_grid = axs_grid[:, None]
    
    for ii, n_list in enumerate(sim_grid):
        print("column%i"%ii)
        axs = axs_grid[:,ii]
        if ii!=0:
            for ax in axs:
                ax.set_yticklabels([])
                
        jj_counter=0
        for jj, n in enumerate(n_list):
            print("n=%i"%n)
           

            if skipped_n!=None: # check if we're skipping anything. 
                if n in skipped_n:
                    ### do not increase jj counter here so that initial binaries still get loaded for a 
                    # non-skipped simulation. 
                    continue
            path = paths[n]
            init_displacement = init_displacements[n]
            
            i = orbit_times[n] # should match the length of indexed n's
            
            apo = apocenters[n]
            all_particles, binaries, singles, coords, single_coords, binary_coords = load_coords(path, i, return_all_particles=True,
                                                                                                 return_single_particles=True,
                                                                                                 use_prog=True,
                                                                                                 tdis_estimate = i, 
                                                                                                 init_displacement=init_displacement)
            
            #### THE TRIMMING 
            ##### concatenate single/binaries ? (maybe not necessary lowkey)
            b_phi1, b_phi2, b_r, b_vr = binary_coords['phi1'],binary_coords['phi2'],binary_coords['r'],binary_coords['vr']
            s_phi1, s_phi2, s_r, s_vr = single_coords['phi1'], single_coords['phi2'], single_coords['r'], single_coords['vr']

            all_phi1 = np.concatenate([s_phi1,b_phi1])
            all_phi2 = np.concatenate([s_phi2,b_phi2])
            all_r = np.concatenate([s_r, b_r])
            all_vr = np.concatenate([s_vr, b_vr])

            ### FIRST clip based on what is still in the MW.
            in_MW = all_r < 1.5*apo
            in_MW_single = s_r < 1.5*apo
            in_MW_binary = b_r < 1.5*apo
            phi1, phi2, r, vr = all_phi1[in_MW], all_phi2[in_MW], all_r[in_MW], all_vr[in_MW]
            s_phi1, s_phi2, s_r, s_vr = s_phi1[in_MW_single], s_phi2[in_MW_single], s_r[in_MW_single], s_vr[in_MW_single]
            b_phi1, b_phi2, b_r, b_vr = b_phi1[in_MW_binary], b_phi2[in_MW_binary], b_r[in_MW_binary], b_vr[in_MW_binary]

            ### NEXT do the 3-sigma clipping in phi1, phi2, vr
            sigphi1 = calc_dispersion(phi1)
            clip_phi1 = np.abs(phi1)<=3*sigphi1
            clip_phi1_single = np.abs(s_phi1)<=3*sigphi1
            clip_phi1_binary = np.abs(b_phi1)<=3*sigphi1

            sigphi2 = calc_dispersion(phi2)
            clip_phi2 = np.abs(phi2)<=3*sigphi2
            clip_phi2_single = np.abs(s_phi2)<=3*sigphi2
            clip_phi2_binary = np.abs(b_phi2)<=3*sigphi2

            sigr = calc_dispersion(r)
            rmean = np.mean(r)
            clip_r = np.abs(r-rmean)<=3*sigr
            clip_r_single = np.abs(s_r-rmean)<=3*sigr
            clip_r_binary = np.abs(b_r-rmean)<=3*sigr

            # sigvr = calc_dispersion(vr)
            # meanvr = np.mean(vr)
            # clip_vr = np.abs(vr-meanvr)<=3*sigvr
            # clip_vr_single = np.abs(s_vr-meanvr)<=3*sigvr
            # clip_vr_binary = np.abs(b_vr-meanvr)<=3*sigvr

            final_clip = clip_phi1 & clip_phi2 & clip_r #& clip_vr
            final_clip_single = clip_phi1_single & clip_phi2_single & clip_r_single #& clip_vr_single
            final_clip_binary = clip_phi1_binary & clip_phi2_binary & clip_r_binary #& clip_vr_binary

            phi1, phi2, r, vr = phi1[final_clip], phi2[final_clip], r[final_clip], vr[final_clip]
            s_phi1, s_phi2, s_r, s_vr = s_phi1[final_clip_single], s_phi2[final_clip_single], s_r[final_clip_single], s_vr[final_clip_single]
            b_phi1, b_phi2, b_r, b_vr = b_phi1[final_clip_binary], b_phi2[final_clip_binary], b_r[final_clip_binary], b_vr[final_clip_binary]
            periods = calc_P(binaries.semi * u.pc, binaries.mass * u.Msun).to(u.yr)[in_MW_binary][final_clip_binary]

            #### plot phi1/phi2
            _, straight_phi2 = straighten_stream_polynomial(phi1, phi2, show_plot=False)
            axs[jj].scatter(phi1, straight_phi2, s=.1, c=colors[jj], rasterized=True)
            axs[jj].text(0.03, 0.75, sim_labels[n], transform=axs[jj].transAxes)
            # axs[jj].text(0.05, 0.6, r"t=%i Myr"%i, transform=axs[jj].transAxes)
            axs[jj].set_ylim(-1,1)
            # axs[jj].set_ylabel(r"$\phi_2\ (\degree)$")
            
            
            phimax = np.max(np.abs(phi1))

            # bins = np.linspace(-phimax, phimax, nbin)
            bins = np.arange(-phimax, phimax+deg_per_bin, deg_per_bin)
            bin_centers = (bins[:-1] + bins[1:])/2

            #### select which bins get plotted by how many objects are in the bin.
            counts, _,_ = binned_statistic(phi1, phi1, statistic='count', bins=bins)
            # fig, ax = plt.subplots()
            # ax.plot(bin_centers, counts)
            sel = counts>50 # only include binned data when there are >10 SYSTEMS in the bin.

            
        
            # load initial binaries and their coordinates ;-; ;-; ;-; ;-;
            if jj_counter==0: ## only once per orbit shape, since the idea is it should be the same for all (assuming low/high mass are the same)?
                binaries_0 = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR,
                                            interrupt_mode='bse', external_mode='galpy')
                binaries_0.loadtxt(path+"data.0.binary", skiprows=1)
                periods_0 = calc_P(binaries_0.semi*u.pc, binaries_0.mass*u.Msun).to(u.yr)

                singles_0 = petar.Particle(interrupt_mode='bse', external_mode='galpy')
                singles_0.loadtxt(path+'data.0.single')
            
            
            global_fbin_0 = len(binaries_0.mass) / (len(singles_0.mass)+len(binaries_0.mass)) # for later idk         

            ##### do axs[0] -- total binary fraction! 


            ### compute total binary fraction??
            phi1_concatenated = np.concatenate([s_phi1, b_phi1])
            binary_indicator = np.concatenate([
                np.zeros(len(s_phi1)),
                np.ones(len(b_phi1))
            ])
            # global_fbin, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='mean', 
            #                                     bins=bins)
            global_counts, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='count',
                                                bins=bins)
            global_nbin, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='sum', bins=bins)
            # global_binary_uncertainties = 1.96 * np.sqrt((global_fbin*(1-global_fbin))/global_counts) # Wald confidence interval
            
            global_fbin = []
            global_fbin_unc_lower = []
            global_fbin_unc_upper = []
            for k, N in zip(global_nbin, global_counts):
                p, P = Prob_of_frac(k, N)
                err_low, med, err_up = percentile(p, P)
                global_fbin.append(med)
                global_fbin_unc_lower.append(med-err_low)
                global_fbin_unc_upper.append(err_up-med)  
            global_fbin = np.array(global_fbin)
            global_fbin_unc_lower = np.array(global_fbin_unc_lower)
            global_fbin_unc_upper = np.array(global_fbin_unc_upper)
            
            # loop through binary fractions
            global_fbin_ind = nsims_per_column
            axs[global_fbin_ind].errorbar(bin_centers[sel], global_fbin[sel],
                            yerr = [global_fbin_unc_lower[sel], global_fbin_unc_upper[sel]], #yerr=global_binary_uncertainties[sel], 
                            c=colors[jj])#, label=labels[n])
            # axs[0].legend()
            if jj_counter==0: # for first n, put a horizontal line on the total fbin panel
                axs[global_fbin_ind].axhline(global_fbin_0, color='k', linestyle="--", linewidth=1)
                axs[global_fbin_ind].text(0.03, 0.75, "all binaries", fontsize=25,
                                transform=axs[global_fbin_ind].transAxes)
                
            axs[global_fbin_ind].set_ylim(-global_fbin_0/10, global_fbin_0*ylim_frac)
            # axs[global_fbin_ind].set_ylabel(r"$f_{\rm{bin}}$")


            #### PLOT IN EACH PERIOD SLICE
            period_axes = axs[int(1+nsims_per_column):]
            for k, period_range in tqdm(enumerate(period_ranges)):
                period_sel = (periods>=period_range[0]) & (periods<period_range[1])

                phi1_concatenated = np.concatenate([
                    s_phi1,
                    b_phi1[period_sel],
                    b_phi1[~period_sel]
                ])
                binary_indicator = np.concatenate([
                    np.zeros(len(s_phi1)),
                    np.ones(len(b_phi1[period_sel])),
                    np.zeros(len(b_phi1[~period_sel]))
                ])

                # binary_fracs, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='mean', 
                #                                     bins=bins)
                total_counts, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='count', 
                                                    bins=bins)
                binary_counts, _,_ = binned_statistic(phi1_concatenated, binary_indicator, statistic='sum',
                                                      bins=bins)

                binary_fracs = []
                binary_uncs_lower = []
                binary_uncs_upper = []
                for chosenK, N in zip(binary_counts, total_counts):
                    p, P = Prob_of_frac(chosenK, N)
                    err_low, med, err_up = percentile(p, P)
                    binary_fracs.append(med)
                    binary_uncs_lower.append(med-err_low)
                    binary_uncs_upper.append(err_up-med)                     
                binary_fracs = np.array(binary_fracs)
                binary_uncs_lower = np.array(binary_uncs_lower)
                binary_uncs_upper = np.array(binary_uncs_upper)
                
                
                ### compute initial binary fraction
                period_sel_0 = (periods_0>=period_range[0]) & (periods_0<period_range[1])
                binary_indicator_0 = np.concatenate([
                    np.zeros(len(singles_0.mass)),
                    np.ones(len(periods_0[period_sel_0])),
                    np.zeros(len(periods_0[~period_sel_0]))
                ])
                initial_fbin = np.mean(binary_indicator_0)

                period_axes[k].errorbar(bin_centers[sel], binary_fracs[sel],# yerr=err[sel],
                                        yerr = [binary_uncs_lower[sel], binary_uncs_upper[sel]], 
                                        c=colors[jj])
                pr = period_range.to(u.yr).value
                # if ii==0:
                
                if k==0:
                    period_axes[k].text(0.03, 0.75, r"$P< %.0e\ \rm{yr}$"%(pr[1]),
                                        fontsize=25, transform=period_axes[k].transAxes)
                elif k==len(period_axes)-1:
                    period_axes[k].text(0.03, 0.75, r"$P\geq %.0e\ \rm{yr}$"%(pr[0]),
                                        fontsize=25, transform=period_axes[k].transAxes)
                else:
                    period_axes[k].text(0.03,0.75, r"%.0e $\leq P <$ %.0e yr"%(pr[0], pr[1]),
                                        fontsize=25, transform = period_axes[k].transAxes)

                if jj_counter==0:
                    period_axes[k].axhline(initial_fbin, color='k', linestyle="--", linewidth=1)
                
                ibf = initial_fbin
                period_axes[k].set_ylim(-ibf/10, ibf*ylim_frac)
                # period_axes[k].set_ylabel(r"$f_{\rm{bin}}$")
                # if ii==0:
                #     # period_axes[k].set_ylabel(r"$f_{\rm{bin}}$")
            # for ax in axs[:-1]:
            #     ax.set_xticklabels([])
            
            jj_counter+=1

                
        

    for kk in range(len(column_labels)):
        axs_grid[0,kk].set_title(column_labels[kk])
    # axs_grid[0,0].set_title("Circular")
    # axs_grid[0,1].set_title("GD1")
    # axs_grid[0,2].set_title("Pal 5")
    
    fig.text(0.5, 0.07, r"$\phi_1$ (deg)", ha='center', fontsize=40)
    # fig.text(0.08, 0.25, r"$f_{\rm{bin}}$", va='center', rotation='vertical', fontsize=40)
    # fig.text(0.08, 0.75, r"$\phi_{2}$", va='center', rotation='vertical', fontsize=40)
    # Get positions of top 4 and bottom 4 rows
    top_axes = axs_grid[:len(sim_grid[0]), 0]  # assuming all columns aligned
    bottom_axes = axs_grid[len(sim_grid[0]):, 0]

    def get_y_center(axes):
        centers = [ax.get_position().y0 + ax.get_position().height / 2 for ax in axes]
        return sum(centers) / len(centers)

    y_phi2 = get_y_center(top_axes)
    y_fbin = get_y_center(bottom_axes)

    # Replace static values with these dynamic ones
    fig.text(0.08, y_fbin, r"$f_{\rm{bin}}$", va='center', rotation='vertical', fontsize=40)
    fig.text(0.08, y_phi2, r"$\phi_{2}$", va='center', rotation='vertical', fontsize=40)
    
    if save==True:
        # plt.savefig("/n/home02/amphillips/"+title, dpi=300, bbox_inches="tight")  
        plt.savefig(title, dpi=300, bbox_inches='tight')
    
    

def load_in_cluster_period_hists(paths, t_rh0s, sim_inds,
                                 t_trh_list=np.linspace(0,1,50), skipped_n=None):
    bins=np.linspace(-3.5, 6.5, 20)
    bin_centers = (bins[:-1]+bins[1:])/2
    # colors = plt.cm.gist_earth_r(np.linspace(0.1,1,len(t_trh_list))
    hists = []
    nbounds = []
    for k in range(len(sim_inds)):
        print("column %i"%k)
        n_list = sim_inds[k]
        hists_for_this_column = []
        nbounds_for_this_column = []
        for j, n in enumerate(n_list):
            print("sim %i"%n)
            
            path = paths[n]
            t_rh = t_rh0s[n]
            i_list = t_trh_list * t_rh
            i_list = i_list.astype(int)
            
            hists_for_this_n = []
            nbounds_for_this_n = []
            
            for i in tqdm(i_list):
                if skipped_n!=None: # check if we're skeeping this. 
                    if n in skipped_n:
                        zero_hist = np.zeros(len(bin_centers))
                        hists_for_this_n.append(zero_hist)
                        nbounds_for_this_n.append(0)
                        continue
                binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR, interrupt_mode='bse', external_mode='galpy')
                binaries.loadtxt(paths[n]+"data."+str(i)+".binary", skiprows=1)
                rrel, clip = clip_outside_rtid(paths[n], binaries, i, in_core_frame=True)

                Ps = calc_P(binaries.semi*u.pc, binaries.mass*u.Msun).to(u.yr)
                Ps = Ps[clip]
                logPs = np.log10(Ps.to(u.yr).value)
                # print(len(logPs))
                y, edges = np.histogram(logPs, bins=bins, weights = np.zeros_like(logPs)+1/logPs.size)
                hists_for_this_n.append(y)
                nbounds_for_this_n.append(len(logPs))


            hists_for_this_column.append(hists_for_this_n)
            nbounds_for_this_column.append(nbounds_for_this_n)

        hists.append(hists_for_this_column)
        nbounds.append(nbounds_for_this_column) 
        
    return bin_centers, np.array(hists), np.array(nbounds)

def plot_incluster_period_dist(paths, t_rh0s, sim_inds, labels, bin_centers, hists, nbounds, t_trh_list=np.linspace(0,1,50),
                              figsize=[20,20], save=False, title=None, cmap=plt.cm.gist_earth_r, cm_min=0, cm_max=1):
    fig, axs_grid = plt.subplots(len(sim_inds[0]), len(sim_inds), figsize=figsize,
                                 sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    colors = cmap(np.linspace(cm_min, cm_max, len(t_trh_list)))
    axs_grid = np.array(axs_grid)
    if axs_grid.ndim == 1:
        # if it's (N,) that means one column; convert to (N,1) so axs_grid[:, ii] is safe
        axs_grid = axs_grid[:, None]
        
    for k in range(len(sim_inds)):
        axs = axs_grid[:,k]
        n_list = sim_inds[k]
        hists_for_this_column = hists[k]
        nbounds_for_this_column = nbounds[k]
        j=0
        for k, n in enumerate(n_list):
            path = paths[n]
            hists_for_this_n = hists_for_this_column[j]
            nbounds_for_this_n = nbounds_for_this_column[j]
            nbound_final = nbounds_for_this_n[-1]
            ax = axs[k]
            
            c_index=0
            for hist in hists_for_this_n:
                ax.plot(bin_centers, hist, color=colors[c_index])
                c_index+=1
                
            ax.text(0.05,0.85, labels[n], transform=ax.transAxes, ha='left')
            # ax.text(0.05, 0.7, r"$t_{\rm{rh,0}}=%i$ Myr"%t_rh0s[n], transform=ax.transAxes, ha='left', fontsize=15)
            # ax.text(0.05, 0.55, r"$N_{\rm{bound}}=%i$"%nbound_final, transform=ax.transAxes, ha='left', fontsize=15)
            # ax.set_yscale('log')
            j+=1            

    
    # Make a dummy mappable with the desired 0..1 range
    norm = Normalize(vmin=0.0, vmax=max(t_trh_list))
    sm = ScalarMappable(norm=norm, cmap=cmap)  # pick your cmap
    sm.set_array([])  # required by some Matplotlib versions

    # ---- Figure-height colorbar (spans all axes vertically) ----
    # Make room on the right
    fig.subplots_adjust(right=0.85)
    fig.canvas.draw()

    # Compute the full vertical span of all visible axes in the figure
    axes = [a for a in fig.axes if a.get_visible()]
    boxes = [a.get_position() for a in axes]
    bottom = min(b.y0 for b in boxes)
    top    = max(b.y1 for b in boxes)
    right  = max(b.x1 for b in boxes)

    # Add a colorbar axes that spans the entire height
    pad, cb_width = 0.01, 0.02  # figure fraction
    cax = fig.add_axes([right + pad, bottom, cb_width, top - bottom])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r'$t/t_{rh,0}$', fontsize=30)
    # cb.set_ticks([0.0, 0.5, 1.0])

    fig.text(0.5, 0.05, r"$\log(P/\rm{yr})$", ha='center', fontsize=30)
    fig.text(0.07, 0.5, r"Fraction in bin", va='center', rotation='vertical', fontsize=30)
    

    
    if save==True:
        # plt.savefig("/n/home02/amphillips/"+title, dpi=300, bbox_inches="tight")
        plt.savefig(title, dpi=300, bbox_inches='tight')

        
def make_binary_roche_plot(paths, labels, 
                           sim_inds, figsize=[10,12], i_cutoff=10,
                           save=False, title=None, colors = ["#031926","#72a0a2"], 
                           quantity='a', xlim=None, ylim=None, 
                           xticks=None, yticks=None):
    """
    'quantity' argument can be 'a' or 'P' (default to a)
    """
    path_to_npzs = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/binary_matrix_npzs/"
    
    fig, axs_grid = plt.subplots(len(sim_inds[0]), len(sim_inds),
                                 sharex=True, sharey=True, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if axs_grid.ndim == 1:
        # if it's (N,) that means one column; convert to (N,1) so axs_grid[:, ii] is safe
        axs_grid = axs_grid[:, None]
    
    for orbit_index in range(len(sim_inds)):
        n_list = sim_inds[orbit_index]
        axs = axs_grid[:,orbit_index]
        
        j=0
        for n in tqdm(n_list):
            i=0
            ax = axs[j]
            ax.text(0.1, 0.9, labels[n], va='top', transform=ax.transAxes)
            
            dat = np.load(path_to_npzs+"%i_binary_matrices.npz"%n)
            period_matrix = dat["period_matrix"]
            a_matrix = dat["a_matrix"]
            i_list = dat["i_list"]
            m1_ID_matrix = dat["m1_ID_matrix"]
            path = paths[n]
            
            # compute what is in rtid
            binaries = petar.Binary(member_particle_type=petar.Particle,
                                    G=petar.G_MSUN_PC_MYR,
                                    interrupt_mode='bse',
                                    external_mode='galpy')
            binaries.loadtxt(path+"data.%i.binary"%i, skiprows=1)
            
            rrel, in_rtid_binary = clip_outside_rtid(path, binaries, i,
                                                         in_core_frame=True)
            binaries = binaries[in_rtid_binary]
            b_r = rrel[in_rtid_binary]
            Mtot = binaries.mass*u.Msun
            a = binaries.semi*u.pc
            P = calc_P(a, Mtot).to(u.yr)
            
            m1s = binaries.p1.mass*u.Msun
            m2s = binaries.p2.mass*u.Msun
            
            # for computing enclosed mass at each r
            all_particles = load_particle(path, i)
            all_masses = all_particles.mass * u.Msun
            ppos, pvel, p_r = xform_to_core_frame(path, all_particles, i)
            
            binary_a_disrupt = []

            for k, binary in enumerate(binaries):
                r_binary = b_r[k]
                Mtot_binary = Mtot[k]
                m1 = m1s[k]
                m2 = m2s[k]
                
                enclosed = p_r<=r_binary
                M_enc = np.sum(all_masses[enclosed])
                
                a_disrupt_with_tides = r_binary * ((m1*m2)/(M_enc*Mtot_binary*16))**(1/3)
                binary_a_disrupt.append(a_disrupt_with_tides.to(u.au).value)
                
            binary_a_disrupt = np.array(binary_a_disrupt) * u.au
            
            binary_P_disrupt = calc_P(binary_a_disrupt, Mtot).to(u.yr) # also compute the limiting period.
            
            select_binaries_at_this_i = ~np.isnan(a_matrix[:,i])
            bound_binaries = a_matrix[select_binaries_at_this_i][in_rtid_binary]
            
            # determine disruption times?
            t_disrupt_list = []
            for line in bound_binaries[:,i:]:
                len_isnan = len(line[np.isnan(line)])
                # print(len_isnan)
                if len_isnan==0:
                    t_disrupt = max(i_list)
                if len_isnan>0:
                    t_disrupt = i+np.argmax(np.isnan(line)) # first index of nan value. 
                t_disrupt_list.append(t_disrupt)
            t_disrupt_list = np.array(t_disrupt_list)
            i_criterion = i_cutoff # highlight things disrupted byth is time
            survives_selection = t_disrupt_list > i_criterion


            if quantity=='a':
                ax.scatter(a.to(u.au)[survives_selection], binary_a_disrupt.to(u.au)[survives_selection],
                           # s=5*Mtot[survives_selection],
                           c=colors[0], s=15,
                          rasterized=True)
                ax.scatter(a.to(u.au)[~survives_selection], binary_a_disrupt.to(u.au)[~survives_selection],
                           # s=5*Mtot[~survives_selection],
                           c=colors[1],
                           label=r"disrupted in $<%i$ Myr"%i_criterion, s=15,
                          rasterized=True)
                
                all_data = np.concatenate([a.to(u.au).value, 
                                           binary_a_disrupt.to(u.au).value])
                lim = [np.min(all_data), np.max(all_data)]
                ax.plot(lim,lim,ls='--', c='r')

                
                if xlim!=None:
                    ax.set_xlim(*xlim)
                # ax.set_ylim(*lim)
                if ylim!=None:
                    ax.set_ylim(*ylim)
                if xticks!=None:
                    ax.set_xticks(xticks)
                    
                if yticks!=None:
                    ax.set_yticks(yticks)

            if quantity=='P':
                ax.scatter(P.to(u.yr)[survives_selection], binary_P_disrupt.to(u.yr)[survives_selection],
                           c=colors[0], s=15, rasterized=True)
                ax.scatter(P.to(u.yr)[~survives_selection], binary_P_disrupt.to(u.yr)[~survives_selection],
                           c=colors[1], label=r"disrupted in $<%i$ Myr"%i_criterion, s=15,
                           rasterized=True)
                all_data = np.concatenate([P.to(u.yr).value,
                                           binary_P_disrupt.to(u.yr).value])
                lim = [np.min(all_data), np.max(all_data)]
                ax.plot(lim,lim, ls='--', c='r')
                
                if xlim!=None:
                    ax.set_xlim(*xlim)
                if ylim!=None:
                    ax.set_ylim(*ylim)
                    
                if xticks!=None:
                    ax.set_xticks(xticks)
                    
                if yticks!=None:
                    ax.set_yticks(yticks)
                
                
                
            ax.set_xscale('log')
            ax.set_yscale('log')



            j+=1
    if quantity=='a':
        fig.text(0.5, 0.04, r"$a$ (au)", ha='center', fontsize=30)
        fig.text(0.04, 0.5, r"$a_{\rm{disrupt\ with\ tides}}$", va='center', rotation='vertical', fontsize=30)
        
    if quantity=='P':
        fig.text(0.5, 0.04, r"$P$ (yr)", ha='center', fontsize=30)
        fig.text(0.04, 0.5, r"$P_{\rm{disrupt\ with\ tides}}$", va='center', rotation='vertical', fontsize=30)
        
    if save==True:
        plt.savefig(title, dpi=300, bbox_inches='tight')
        
        
def make_binary_ahs_plot(paths, labels, init_displacements, t_rh0s, sim_inds, i_cutoff=10,
                         figsize=[15,12], save=False, title=None,
                         colors = ["#031926","#72a0a2"],
                         quantity='a', xlim=None, ylim=None,
                        xticks=None, yticks=None):
    path_to_npzs = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/binary_matrix_npzs/"
    fig, axs_grid = plt.subplots(len(sim_inds[0]), len(sim_inds), figsize=figsize, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if axs_grid.ndim == 1:
        # if it's (N,) that means one column; convert to (N,1) so axs_grid[:, ii] is safe
        axs_grid = axs_grid[:, None]

    for orbit_index in range(len(sim_inds)): # orbit index pulls columns of the plot/shapes of cluster orbit
        n_list = sim_inds[orbit_index]
        axs = axs_grid[:,orbit_index]

        j=0
        for n in tqdm(n_list):  # j indexes rows, n indexes simulations. 
            # print(n)
            ax = axs[j]
            ax.text(0.05, 0.85, labels[n], transform=ax.transAxes)

            dat = np.load(path_to_npzs+"%i_binary_matrices.npz"%n)
            period_matrix = dat["period_matrix"]
            a_matrix = dat["a_matrix"]
            i_list = dat["i_list"]
            m1_ID_matrix = dat['m1_ID_matrix']
            path = paths[n]
            i = i_cutoff


            # compute hard/soft boundary
            ahs, b_r, a, Mtot, rh, in_rtid_binary = compute_ahs(path, i, shell_width=.5*u.pc)
            Phs = calc_P(ahs, Mtot)
            P = calc_P(a, Mtot)
            
            t_rh_i = t_rh0s[n] #t_rh[0]

            # select from period matrix
            select_binaries_at_this_i = ~np.isnan(period_matrix[:,i]) 
            bound_binaries = period_matrix[select_binaries_at_this_i][in_rtid_binary]


            t_disrupt_list = [] # compute disruption times
            for line in bound_binaries[:,i:]:
                len_isnan = len(line[np.isnan(line)])
                # print(len_isnan)
                if len_isnan==0:
                    t_disrupt = max(i_list)
                if len_isnan>0:
                    t_disrupt = i+np.argmax(np.isnan(line)) # first index of nan value. 
                t_disrupt_list.append(t_disrupt)
            t_disrupt_list = np.array(t_disrupt_list)

            i_criterion = int(t_rh_i) # do the binaries survive one relaxation timescale?
            survives_selection = t_disrupt_list > i_criterion


            ### determine whether things that aren't disrupted are _still in the cluster_ for the next i_criterion. 
            # extract primary 'id' s at this i
            binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR,
                                    interrupt_mode='bse', external_mode='galpy')
            binaries.loadtxt(paths[n]+"data."+str(i_criterion)+".binary", skiprows=1)
            primary_ids = binaries.p1.id

            all_particles = load_particle(paths[n], i_criterion)
            particle_IDs = all_particles.id
            rrel, clip = clip_outside_rtid(paths[n], all_particles, i_criterion, in_core_frame=False)
            IDs_still_bound = particle_IDs[clip]

            p1_ids_still_bound = primary_ids[np.isin(primary_ids, IDs_still_bound)]

            p1_ids_at_i = m1_ID_matrix[:,i][select_binaries_at_this_i][in_rtid_binary] # non-nan at this i, in rtid at this i

            MATRIX_ELEMENTS_STILL_BOUND = np.isin(p1_ids_at_i, p1_ids_still_bound) # ones that will still be bound after trh

            if quantity=='a':
                ax.scatter(a.to(u.au)[survives_selection & MATRIX_ELEMENTS_STILL_BOUND],
                           ahs.to(u.au)[survives_selection & MATRIX_ELEMENTS_STILL_BOUND],
                           c=colors[0], s=15,
                          rasterized=True)
                ax.scatter(a.to(u.au)[~survives_selection], ahs.to(u.au)[~survives_selection],
                           c=colors[1], label='disrupted in next %i Myr'%i_criterion, s=15,
                          rasterized=True)
                all_data = np.concatenate([a.to(u.au).value, ahs.to(u.au).value])
                lim = [np.min(all_data[~np.isnan(all_data) & np.isfinite(all_data)]), np.max(all_data[~np.isnan(all_data) & np.isfinite(all_data)])]
                print(lim)
                ax.plot(lim,lim,c='r', linestyle='--')
                
            
            if quantity=='P':
                ax.scatter(P.to(u.yr)[survives_selection & MATRIX_ELEMENTS_STILL_BOUND],
                           Phs.to(u.yr)[survives_selection & MATRIX_ELEMENTS_STILL_BOUND],
                           c=colors[0], s=15,
                          rasterized=True)
                ax.scatter(P.to(u.yr)[~survives_selection], Phs.to(u.yr)[~survives_selection],
                           c=colors[1], label='disrupted in next %i Myr'%i_criterion, s=15,
                          rasterized=True)
                all_data = np.concatenate([P.to(u.yr).value, Phs.to(u.yr).value])
                lim = [np.min(all_data[~np.isnan(all_data) & np.isfinite(all_data)]), np.max(all_data[~np.isnan(all_data) & np.isfinite(all_data)])]
                ax.plot(lim,lim,c='r', linestyle='--')
                
            if xlim!=None:
                ax.set_xlim(*xlim)
            if ylim!=None:
                ax.set_ylim(*ylim)

            ax.set_yscale("log")
            ax.set_xscale('log')

            if xticks!=None:
                ax.set_xticks(xticks)
            if yticks!=None:
                ax.set_yticks(yticks)
#             ax.set_yticks([1e0,1e2,1e4,1e6])

            j+=1

    if quantity=='a':
        fig.text(0.5, 0.05, r'$a$ (au)', ha='center', fontsize=30)
        fig.text(0.04, 0.5, r"$a_{\rm{h/s}}$ (au)", va='center', rotation='vertical', fontsize=30)  
    if quantity=='P':
        fig.text(0.5, 0.05, r'$P$ (yr)', ha='center', fontsize=30)
        fig.text(0.04, 0.5, r"$P_{\rm{h/s}}$ (yr)", va='center', rotation='vertical', fontsize=30)   

    if save==True:
        plt.savefig(title, dpi=300, bbox_inches="tight")
        
        
        
        
        
        
def load_bound_MF(paths, dissolution_times, sim_inds, t_tdis_list, skipped_n=None):
    bins = np.linspace(np.log10(0.08), np.log10(10), 30)
    bin_centers = (bins[:-1]+bins[1:])/2
    hists = []
    
    for k in range(len(sim_inds)): # index orbit shapes/columns
        print("constructing column %i")
        n_list = sim_inds[k]
        hists_for_this_orbit_shape = []
        
        for j, n in enumerate(n_list):
            print("simulation %i"%n)
            path = paths[n]
            
            tdis = dissolution_times[n]
            i_list = t_tdis_list * tdis
            i_list = i_list.astype(int)
            
            c_index = 0
            hists_for_this_n = []
            for i in tqdm(i_list):
                if skipped_n!=None: # check if we're 'skipping' this sim
                    if n in skipped_n:
                        zero_hist = np.zeros(len(bin_centers))
                        hists_for_this_n.append(zero_hist)
                        continue
                        
                particles = load_particle(path, i)
                rrel, clip = clip_outside_rtid(path, particles, i,
                                                   in_core_frame=False)
                masses = particles.mass[clip]
                logM = np.log10(masses)
                
                y, edges = np.histogram(logM, bins=bins, weights = np.zeros_like(logM)+1/logM.size)
                hists_for_this_n.append(y)
                
            hists_for_this_orbit_shape.append(hists_for_this_n)
            
        hists.append(hists_for_this_orbit_shape)
        
    hists = np.array(hists)
    
    return bin_centers, hists  


def plot_bound_MF(paths, labels, bin_centers, hists, sim_inds, t_tdis_list, figsize=[20,20],
                 cmap=plt.cm.gist_earth_r, cm_min=0, cm_max=1,
                 save=False, title=None, skipped_n=None):
    fig, axs_grid = plt.subplots(len(sim_inds[0]), len(sim_inds), figsize=figsize,
                                 sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    colors = cmap(np.linspace(cm_min, cm_max, len(t_tdis_list)))
    
    for k in range(len(sim_inds)):
        axs = axs_grid[:,k]
        n_list = sim_inds[k]
        hists_for_this_orbit_shape = hists[k]
        
        for j, n in enumerate(n_list):
            # print(n)
            path = paths[n]
            hists_for_this_n = hists_for_this_orbit_shape[j]
            ax = axs[j]
            c_index=0
            for hist in hists_for_this_n:
                ax.plot(bin_centers, hist, color=colors[c_index])
                c_index+=1
                
            ax.text(0.95,0.85, labels[n], transform=ax.transAxes, ha='right')
            # ax.text(0.9, 0.7, r"$t_{\rm{dis}}=%i$ Myr"%dissolution_times[n], transform=ax.transAxes, ha='right', fontsize=15)
            ax.set_yscale('log')
            
            
    ### add a colorbar
    norm = Normalize(vmin=0, vmax=max(t_tdis_list))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    fig.canvas.draw()
    
    axes = [a for a in fig.axes if a.get_visible()]
    boxes = [a.get_position() for a in axes]
    bottom = min(b.y0 for b in boxes)
    top = max(b.y1 for b in boxes)
    right = max(b.x1 for b in boxes)
    
    pad, cb_width = 0.01, 0.02
    cax = fig.add_axes([right+pad, bottom, cb_width, top-bottom])
    
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$t/t_{\rm{dis}}$", fontsize=30)
        
    fig.text(0.5, 0.05, r"$\log(M/M_{\odot})$", ha='center', fontsize=30)
    fig.text(0.07, 0.5, r"Fraction in bin", va='center', rotation='vertical', fontsize=30)
    
    if save==True:
        plt.savefig(title, dpi=300, bbox_inches="tight")
    
    
    
    
    
    
###### PLOT VELOCITY DISPERSIONS!!!
def plot_velocities(paths, labels, init_displacements, apocenters,
                    n, i, ax, period_cuts=np.array([0, 10, 50, 100, 1000, 10000000000]) * u.yr,
                    deg_per_bin=0.5, xlim=None,
                    cmap=plt.cm.gist_earth_r, cm_min=0, cm_max=1):
    colors = cmap(np.linspace(cm_min, cm_max, len(period_cuts)))
    for k, period_cut in enumerate(period_cuts):
        path = paths[n]
        label=labels[n]
        ax.text(0.05, 0.8, label, transform=ax.transAxes)
        init_displacement=init_displacements[n]
        apo = apocenters[n]
        
        #### compute initial cluster dispersion??
        all_particles0, binaries0, singles0, coords0, single_coords0, binary_coords0 = load_coords(path, i=0, return_all_particles=True,
                                                                                                       return_single_particles=True)
        
        bvx,bvy,bvz = binaries0.vel.T * (u.pc/u.Myr)
        bv = np.sqrt(bvx**2 + bvy**2 + bvz**2).to(u.km/u.s)
        
        svx,svy,svz = singles0.vel.T * (u.pc/u.Myr)
        sv = np.sqrt(svx**2 + svy**2 + svz**2).to(u.km/u.s)
        v = np.concatenate([sv, bv])
        sigv = calc_dispersion(v)
        sigv_1d = sigv.to(u.km/u.s).value / np.sqrt(3)
        
        ax.axhline(sigv_1d, linestyle="--", c='k', lw=1)
        
        # now load particles at this time
        all_particles, binaries, singles, coords, single_coords, binary_coords = load_coords(path, i, return_all_particles=True, return_single_particles=True, init_displacement=init_displacement)

        # compute periods
        periods = calc_P(binaries.semi*u.pc, binaries.mass*u.Msun).to(u.yr)

        # based on periods, determine which will be "detected"
        detected = periods <= period_cut

        # pull out coordinates of singles and binaries
        b_phi1, b_phi2, b_r, b_vr = binary_coords['phi1'][detected], binary_coords['phi2'][detected], binary_coords['r'][detected], binary_coords['vr'][detected]
        s_phi1, s_phi2, s_r, s_vr = single_coords['phi1'], single_coords['phi2'], single_coords['r'], single_coords['vr']

        # deal with "undetected" binaries
        undetected_p1_ids = binaries.p1.id[~detected]
        undetected_p2_ids = binaries.p2.id[~detected]

        # find the indices of these IDs in particles.id
        undetected_p1_indices = np.nonzero(np.isin(all_particles.id, undetected_p1_ids))[0]
        undetected_p2_indices = np.nonzero(np.isin(all_particles.id, undetected_p2_ids))[0]
        p1_phi1, p1_phi2, p1_r, p1_vr = coords['phi1'][undetected_p1_indices], coords['phi2'][undetected_p1_indices], coords['r'][undetected_p1_indices], coords['vr'][undetected_p1_indices]
        p2_phi1, p2_phi2, p2_r, p2_vr = coords['phi1'][undetected_p2_indices], coords['phi2'][undetected_p2_indices], coords['r'][undetected_p2_indices], coords['vr'][undetected_p2_indices]


        all_phi1 = np.concatenate([s_phi1, b_phi1, p1_phi1, p2_phi1])
        all_phi2 = np.concatenate([s_phi2, b_phi2, p1_phi2, p2_phi2])
        all_r = np.concatenate([s_r, b_r, p1_r, p2_r])
        all_vr = np.concatenate([s_vr, b_vr, p1_vr, p2_vr])

        in_MW = all_r < 1.5 * apo # FIRST: trim so that we are only looking at things still "in the MW"
        phi1, phi2, r, vr = all_phi1[in_MW], all_phi2[in_MW], all_r[in_MW], all_vr[in_MW]

        
        # NOW do the 3 sigma trimming
        sigphi1 = calc_dispersion(phi1) # 3-sigma in phi1
        phi1mean = np.mean(phi1)
        clip_phi1 = np.abs(phi1-phi1mean)<=3*sigphi1
        
        sigphi2 = calc_dispersion(phi2) # 3-sigma in phi2
        phi2mean = np.mean(phi2)
        clip_phi2 = np.abs(phi2-phi2mean)<=3*sigphi2 
        
        sigr = calc_dispersion(r) #3-sigma in r
        rmean = np.mean(r)
        clip_r = np.abs(r-rmean) <=3*sigr       
        
        final_clip = clip_phi1 & clip_phi2 & clip_r #& clip_vr

        phi1, phi2, r, vr = phi1[final_clip], phi2[final_clip], r[final_clip], vr[final_clip]
        phi1, straight_vr = straighten_stream_polynomial(phi1, vr, degree=5, show_plot=False)
        phi1, straight_phi2 = straighten_stream_polynomial(phi1, phi2, degree=5, show_plot=False)
        phimax = np.max(np.abs(phi1))
        # bins = np.linspace(-phimax, phimax, 30)
        # deg_per_bin = 0.5
        
        #### decide on bins/bin centers only once!
        if k==0:
            bins = np.arange(-phimax, phimax+deg_per_bin, deg_per_bin)
            bin_centers = (bins[1:]+bins[:-1])/2
        
        sigv, _,_ = binned_statistic(x = phi1, values = straight_vr, statistic=calc_dispersion_with_3sig_clip, bins=bins)
        counts, _, _ = binned_statistic(x=phi1, values=phi1, statistic='count', bins=bins)
        sel=counts>50

        if k==0:
        #     axs[0].scatter(phi1, straight_phi2, s=.01, c='k')
        #     axs[1].scatter(phi1, straight_vr, s=.01, c='k')
            ax.plot(bin_centers[sel], sigv[sel], color=colors[k], label=r'detect no binaries')
            continue

        if k==int(len(period_cuts)-1):
            ax.plot(bin_centers[sel], sigv[sel], color=colors[k], label=r'detect all binaries')
            continue

        ax.plot(bin_centers[sel], sigv[sel], color=colors[k], label=r'detect binaries with $P<%i$ yr'%period_cut.to(u.yr).value)
    ax.set_yscale('log')

    # axs[2].legend(bbox_to_anchor=[1,1])
    if xlim!=None:
        for ax in axs:
            # ax.set_xlim(min(bin_centers[sel])-4*deg_per_bin, max(bin_centers[sel])+4*deg_per_bin)
            ax.set_xlim(*xlim)
            

def assemble_veldisp_plot(paths, labels, init_displacements, apocenters,plot_ivals,
                          sim_inds,
                          period_cuts=np.array([0, 10, 50, 100, 1000, 10000000000]) * u.yr,
                          deg_per_bin=0.5, xlim=None,
                          cmap=plt.cm.gist_earth_r, cm_min=0, cm_max=1, figsize=[20,25],
                          save=False, title=None,
                          skipped_n=None):
    fig, axs_grid = plt.subplots(len(sim_inds[0]), len(sim_inds), figsize=figsize, sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for orbit_index in range(len(sim_inds)):
        n_list = sim_inds[orbit_index]
        axs = axs_grid[:,orbit_index]
        
        for k, n in tqdm(enumerate(n_list)):
            if skipped_n!=None:
                if n in skipped_n:
                    continue
            ax = axs[k]
            i=plot_ivals[n]
            plot_velocities(paths, labels, init_displacements, apocenters,
                            n, i, ax, period_cuts, deg_per_bin=deg_per_bin, cmap=cmap)
            
    # axs_grid[-1,-1].legend(bbox_to_anchor=[1,1])
    bb = axs_grid[0,1].get_position()
    x_center = bb.x0 + bb.width/2
    handles, labels = axs_grid[0,1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(x_center, bb.y1),
        bbox_transform=fig.transFigure
    )
    fig.text(0.06, 0.5, r"$\sigma_{v_r}\ (\rm{km\ s^{-1}})$", va='center', rotation='vertical', fontsize=30)
    fig.text(0.5, 0.04, r"$\phi_1\ (\degree)$", ha='center', fontsize=30)
    
    if save==True:
        plt.savefig(title, dpi=300, bbox_inches='tight')

    
        
################# functions for getting binary fractions.

def get_orbital_params(paths, times, n):
    """
    retrieve orbital parameters from path paths[n] at time times[n]
    returns an array of params
    [v0 (km/s), K (km/s), w, phi0, e, P (day)]
    where w, phi1 are uniformly sampled and an isotropic inclination is picked to compute k
    """
    i = times[n]
    path = paths[n]

    core = petar.Core(interrupt_mode='bse', external_mode='galpy')
    core.loadtxt(path+"data.core")
    core_pos = core.pos[i] * u.pc.to(u.kpc) # dimensionless core position in kpc
    core_vel = core.vel[i] * (u.pc/u.Myr).to(u.kpc/u.Myr) # dimensionless core velocity in kpc/myr
    core_w_jake_units = np.hstack([core_pos, core_vel])

    binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR,
                            interrupt_mode='bse', external_mode='galpy')
    binaries.loadtxt(path+"data.%i.binary"%i, skiprows=1)
    binary_pos = binaries.pos*u.pc.to(u.kpc)+core_pos
    binary_vel = binaries.vel*(u.pc/u.Myr).to(u.kpc/u.Myr)+core_vel
    binary_w = np.hstack([binary_pos, binary_vel])



    binary_stream_obj = StreamFrame(sim_coords=binary_w, prog_sim_coord=core_w_jake_units)
    binary_coords = binary_stream_obj.GetStreamFrame()

    ###### collect everything we need
    a_vals = binaries.semi*u.pc
    Mtot_vals = binaries.mass*u.Msun
    P_vals = calc_P(a_vals, Mtot_vals)

    inclinations = draw_inclinations(len(a_vals))

    K_vals = np.array([calc_amplitude(binary, i) for binary, i in zip(binaries, inclinations)]) * u.m/u.s

    e_vals = binaries.ecc

    phi1_vals = binary_coords['phi1'] # degrees
    v0_vals = (binary_coords['vr']*u.km/u.s).to(u.m/u.s) # extract systemic velocities.



    ### draw arguments of periapsis and mean anomaly phase offsets ?? 
    w_vals = np.random.uniform(low=0, high=2*np.pi, size=len(a_vals))
    phi0_vals = np.random.uniform(low=0, high=1, size=len(a_vals))
    
    params_all = np.array([v0_vals.to(u.km/u.s).value, 
                           K_vals.to(u.km/u.s).value,
                           w_vals,
                           phi0_vals,
                           e_vals,
                           P_vals.to(u.day).value]
                          ).T
    
    return params_all, phi1_vals

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Vectorized Kepler solver using Newton-Raphson with safety.
    """
    M = np.asarray(M)
    e = np.asarray(e)

    # Ensure shapes are compatible
    if M.shape != e.shape:
        e = np.full_like(M, e)

    E = M.copy()  # initial guess

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime

        # Protect against divide-by-zero or overflow
        delta = np.where(np.isfinite(delta), delta, 0.0)
        E_new = E - delta

        # Convergence mask
        if np.all(np.abs(delta) < tol):
            break
        E = E_new

    # Final sanity check
    E = np.where(np.isfinite(E), E, np.nan)
    return E

def true_anomaly(E, e):
    """
    Convert eccentric anomaly E to true anomaly nu safely.
    """
    sqrt_1_plus_e = np.sqrt(np.clip(1 + e, 0, None))
    sqrt_1_minus_e = np.sqrt(np.clip(1 - e, 0, None))

    sin_E2 = np.sin(E / 2)
    cos_E2 = np.cos(E / 2)

    # Replace NaNs with 0 to avoid propagating errors
    sin_E2 = np.nan_to_num(sin_E2)
    cos_E2 = np.nan_to_num(cos_E2)

    return 2 * np.arctan2(
        sqrt_1_plus_e * sin_E2,
        sqrt_1_minus_e * cos_E2
    )

def radial_velocity(t, params):
    """
    t : times at which to compute RV
    v0 : systemic velocity (not a quantity, but in km/s)
    K : RV semiamplitude (not a quantity, but in km/s)
    w : arg of periapsis
    e : eccentricity
    phi_0 : mean anomaly phase offset
    P : period (in same units as t)
    """
    v0, K, w, phi_0, e, P = params


    M = 2 * np.pi * t / P - phi_0
    E = solve_kepler(M % (2 * np.pi), e)
    nu = true_anomaly(E, e)
    vr = v0 + K * (np.cos(nu + w) + e * np.cos(w))
    ### in another version i might add gaussian noise with sigma=.1 km/s to this. 
    ### returns radial velocity in whatever units vr and v0 are in!
    return vr

def get_obstimes(N_obsm, N):
    dt_min = 30
    dt_max = 3*360
    rng = np.random.default_rng(seed=42)

    base = np.repeat(0, N)
    gap1 = rng.uniform(30,90, size=N)
    gap2 = rng.uniform(3*365, 5*365, size=N)

    deltaTs = np.vstack((base, gap1, gap2)).T

    obstimes_all = np.cumsum(deltaTs, axis=1) ##### THIS IS WHAT WILL GO INTO THE RV GENERATION FUNCTION!   
    return obstimes_all

def get_rvs(params_all, obstimes_all):

    RVs_all = []
    N=len(params_all[:,0])

    for j in tqdm(range(N)):
        params = params_all[j]
        obstimes = obstimes_all[j]

        rvs = radial_velocity(obstimes, params)
        RVs_all.append(rvs)

    rvs_all = np.array(RVs_all)
    return rvs_all


def get_detections(e_rv, rvs_all, v0_vals, bool_arr="undet"): #return "undet" of "det"
    """
    returns a bool array; v0_vals should be dimensionless and in km/s
    """
    mu_rvs = np.mean(rvs_all, axis=1) # vsys
    delta_vsys = np.abs(mu_rvs-v0_vals.to(u.km/u.s).value)
    resid_rvs = rvs_all - mu_rvs[:, None]

    chi2_rvs = np.sum((resid_rvs/e_rv)**2, axis=1)
    Nobs=3
    P_chi2 = stats.chi2.sf(chi2_rvs, df=Nobs-1)
    zscore = np.max(np.abs(resid_rvs/e_rv), axis = 1) # idk what this is,

    undet = P_chi2>0.1 # detect binaires when the P value of the chi square test is <0.1 
    
    if bool_arr=='undet':
        return undet, delta_vsys
    if bool_arr=='detet':
        return ~undet, delta_vsys
    
    