script_path = "/n/home02/amphillips/binaries_in_streams/scripts" # for cannon

print("loading packages")
import argparse
import petar
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms
from gala.units import galactic

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from tqdm import tqdm

import sys

sys.path.append(script_path)
from streamframe import StreamFrame
import PETAR_ANALYSIS_FUNCTIONS as paf
import astropy.constants as const

print("assigning paths")
ph = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/"

path0 = ph+"optimized_dense_sims/0_circ_rvir0.75_lm/"
path1 = ph+"optimized_dense_sims/1_circ_rvir0.75_hm/"
path2 = ph+"2_circ_rvir1.5_lm/"
path3 = ph+"3_circ_rvir1.5_hm/"
path4 = ph+"4_circ_rvir3_lm/"
path5 = ph+"5_circ_rvir3_hm/"
path6 = ph+"6_circ_rvir6_lm/"
path7 = ph+"7_circ_rvir6_hm/"

path8 = ph+"optimized_dense_sims/8_gd1_rvir0.75_lm/"
# path8 = ph+"8_reprocessed_REAL/" #<- this would be in scratch, not storage.
path9 = ph+"optimized_dense_sims/9_gd1_rvir0.75_hm/"
path10 = ph+"10_gd1_rvir1.5_lm/"
path11 = ph+"11_gd1_rvir1.5_hm/"
path12 = ph+"12_gd1_rvir3_lm/"
path13 = ph+"13_gd1_rvir3_hm/"
path14 = ph+"14_gd1_rvir6_lm/"
path15 = ph+"15_gd1_rvir6_hm/"

path16 = ph+"optimized_dense_sims/16_pal5_rvir0.75_lm/"
path17 = ph+"optimized_dense_sims/17_pal5_rvir0.75_hm/"
path18 = ph+"18_pal5_rvir1.5_lm/"
path19 = ph+"19_pal5_rvir1.5_hm/"
path20 = ph+"20_pal5_rvir3_lm/"
path21 = ph+"21_pal5_rvir3_hm/"
path22 = ph+"22_pal5_rvir6_lm/"
path23 = ph+"23_pal5_rvir6_hm/"

labels=["circ_rvir0.75_lm", "circ_rvir0.75_hm", "circ_rvir1.5_lm", "circ_rvir1.5_hm","circ_rvir3_lm","circ_rvir3_hm","circ_rvir6_lm","circ_rvir6_hm",
        "gd1_rvir0.75_lm", "gd1_rvir0.75_hm", "gd1_rvir1.5_lm","gd1_rvir1.5_hm","gd1_rvir3_lm","gd1_rvir3_hm","gd1_rvir6_lm","gd1_rvir6_hm",
        "pal5_rvir0.75_lm", "pal5_rvir0.75_hm", "pal5_rvir1.5_lm","pal5_rvir1.5_hm","pal5_rvir3_lm","pal5_rvir3_hm","pal5_rvir6_lm","pal5_rvir6_hm"]

paths = [path0,path1,path2,path3,path4,path5,path6,path7,
         path8,path9,path10,path11,path12,path13,path14,path15,
         path16,path17,path18,path19,path20,path21,path22,path23]

id_circ=np.array([20.0,0.0,0.0,0.0,197.61111175113643,0.0])

init_pos = np.array([-14680.175269960177, -18049.00342055592, -3355.258112269282]) * u.pc.to(u.kpc)
init_vel = np.array([126.94017915235732, -33.56076783776999, 112.93136486685516]) # km/s
id_gd1 = np.concatenate([init_pos, init_vel])


init_pos_pal5 = np.array([13536.852506889449, -3328.304203161837, 5478.0923546972035]) * u.pc.to(u.kpc)
init_vel_pal5 = np.array([119.95920838380918, -89.0209982268969, -88.31131556786855]) # km/s
id_pal5 = np.concatenate([init_pos_pal5, init_vel_pal5])


init_displacements = [id_circ]*8 + [id_gd1]*8 + [id_pal5]*8





##########################
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_n", type=int, default=0)


    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--save_to", type=str,
                        default="/n/netscratch/conroy_lab/Lab/amphillips/small_grid/binary_matrix_npzs/")
    
    args = parser.parse_args()
    
    n = args.sim_n
    path = paths[n]
    
    
    # init_displacement=init_displacements[n]
    
    i_list = np.arange(args.start, args.stop+args.step, args.step)

    period_matrix, a_matrix, e_matrix, m1_matrix, m2_matrix, m1_ID_matrix, m2_ID_matrix, array_of_bIDs = paf.create_binary_matrix(path, i_list)
    
    savedir = args.save_to
    np.savez(savedir+str(n)+"_binary_matrices", # save the npz file to the same directory as the simulation data. 
             i_list=i_list,
             period_matrix = period_matrix,
             a_matrix = a_matrix,
             e_matrix = e_matrix,
             m1_matrix = m1_matrix,
             m2_matrix = m2_matrix,
             m1_ID_matrix = m1_ID_matrix,
             m2_ID_matrix = m2_ID_matrix,
             binary_IDs = array_of_bIDs
            )
    
