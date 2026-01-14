import petar
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.table import Table
from scipy.stats import binned_statistic_2d

import galpy
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit

from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

sys.path.append('/n/home02/amphillips/scripts/')
from analyze_petar import analyze_petar
print("loaded packages")

def calc_P(a, Mtot):
    """
    period from semimajor axis
    """
    P = 2*np.pi * np.sqrt((a**3)/(const.G * Mtot))
    return P


################ specify which simulation to read and where to save frames
path="/n/holystore01/LABS/conroy_lab/Lab/amphillips/denser_mcluster_test/"
savedir="/n/holystore01/LABS/conroy_lab/Lab/amphillips/stream_movies/dense_test/binaries_scatter_equal_aspect/"
len_sim = 3000 # length of simulation
equal_aspect=True # whether to adjust plot aspect ratios
################################


for i in range(len_sim+1):
    obj = analyze_petar(path, file_n=i, add_CM=False, r_criterion=500000,
                        plot_rs=False, read_sb=True)    

    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=[17,5])
    sps = obj.single_particles
    spos = sps.pos

    bps = obj.binary_particles
    bpos = bps.pos


    semi_au = bps.semi * 206265

    Periods = calc_P(bps.semi*u.pc, bps.mass*u.Msun)
    P_yr = Periods.to(u.yr).value

    ax1.scatter(spos[:,0], spos[:,1], s=0.1, c='0.7')
    # ax1.scatter(bpos[:,0], bpos[:,1], s=1, c=np.log10(semi_au), cmap='plasma')
    ax1.scatter(bpos[:,0], bpos[:,1], s=1, c=np.log10(P_yr), cmap='magma')


    ax2.scatter(spos[:,0], spos[:,2], s=0.1, c='0.7')
    # ax2.scatter(bpos[:,0], bpos[:,2], s=1, c=np.log10(semi_au), cmap='plasma')
    ax2.scatter(bpos[:,0], bpos[:,2], s=1, c=np.log10(P_yr), cmap='magma')


    ax3.scatter(spos[:,1], spos[:,2], s=0.1, c='0.7')
    # ob = ax3.scatter(bpos[:,1], bpos[:,2], s=1, c=np.log10(semi_au), cmap='plasma')
    ob = ax3.scatter(bpos[:,1], bpos[:,2], s=1, c=np.log10(P_yr), cmap='magma')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0)
    # colorbar = fig.colorbar(ob, cax=cax, orientation='vertical', label=r"$\log{a/\rm{AU}}$")
    colorbar = fig.colorbar(ob, cax=cax, orientation='vertical', label=r"$\log{P/\rm{yr}}$")

    ax1.set_xlabel("x [pc]")
    ax1.set_ylabel("y [pc]")
    
    ax2.set_xlabel("x [pc]")
    ax2.set_ylabel("z [pc]")
    
    ax3.set_xlabel("y [pc]")
    ax3.set_ylabel("z [pc]")
    
    if equal_aspect==True:
        for ax in (ax1,ax2,ax3):
            ax.set_aspect("equal")
    
    fig.suptitle("%i Myr"%i)
    
    filename=f'frame_{i:05d}.png'
    plt.savefig(savedir+filename)#, bbox_inches="tight")
    
    # if i%10==0:
    #     print("saved file "+str(file_i))
    
    plt.close()
