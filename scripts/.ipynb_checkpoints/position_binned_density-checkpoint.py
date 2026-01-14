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
import matplotlib.colors as mcolors


import sys

sys.path.append('/n/home02/amphillips/scripts/')
from analyze_petar import analyze_petar


path="/n/holystore01/LABS/conroy_lab/Lab/amphillips/denser_mcluster_test/"
savedir="/n/holystore01/LABS/conroy_lab/Lab/amphillips/stream_movies/dense_test/pos_hist_and_density/"
start = 0 # nominally, 0
len_sim = 3000 # length of simulation
equal_aspect=False # whether to adjust plot aspect ratios
nbins = 31

for i in range(start, len_sim+1):
    obj = analyze_petar(path, file_n=i, add_CM=False, r_criterion=50000, plot_rs=False, read_sb=True)
    
    filename=f'frame_{i:05d}.png'
    
    # for plotting just the position density hists:
    # obj.plot_position_density_hists(nbins=nbins, save=True, savepath=savedir+filename)
    
    # for plotting the scatter as well:
    obj.plot_pos_scatter_and_density(nbins, save=True, savepath=savedir+filename)