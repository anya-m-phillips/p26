##### i would like the bound mass and half mass radius (of stuff within 100pc of the core) over time. 

print("loading packages...")
import argparse
import petar
import numpy as np
import astropy.units as u
import astropy.constants as const


# import galpy
# from galpy.potential import MWPotential2014
# from galpy.orbit import Orbit

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms

from gala.units import galactic
from tqdm import tqdm

# import sys
script_path = "/n/home02/amphillips/binaries_in_streams/scripts" # for cannon
import sys
sys.path.append(script_path)
from streamframe import StreamFrame

import PETAR_ANALYSIS_FUNCTIONS as paf

print("assigning paths")
ph = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/"

path0 = ph+"optimized_dense_sims/0_circ_rvir0.75_lm/" # unfinished
# path0 = "/n/netscratch/conroy_lab/Lab/amphillips/small_grid/optimize_dense_sims_testing/0_circ_rvir0.75_lm/"
path1 = ph+"optimized_dense_sims/1_circ_rvir0.75_hm/"  # done
path2 = ph+"2_circ_rvir1.5_lm/"
path3 = ph+"3_circ_rvir1.5_hm/"
path4 = ph+"4_circ_rvir3_lm/"
path5 = ph+"5_circ_rvir3_hm/"
path6 = ph+"6_circ_rvir6_lm/"
path7 = ph+"7_circ_rvir6_hm/"

# path8 = ph+"optimized_dense_sims/8_gd1_rvir0.75_lm/"  # done
path8 = ph+"8_copies/0/"
path9 = ph+"optimized_dense_sims/9_gd1_rvir0.75_hm/" # done
# path10 = ph+"10_gd1_rvir1.5_lm/" # some stars got deleted ! 
path10 = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/10_copies_successful/0/" # slightly more rigorous. 


path11 = ph+"11_gd1_rvir1.5_hm/"
path12 = ph+"12_gd1_rvir3_lm/"
path13 = ph+"13_gd1_rvir3_hm/"
# path14 = ph+"14_gd1_rvir6_lm/"
path14 = ph+"14_copies/0/"
# path15 = ph+"15_gd1_rvir6_hm/"
path15 = ph+"15_copies/0/"

path16 = ph+"optimized_dense_sims/16_pal5_rvir0.75_lm/" # done. 
path17 = ph+"optimized_dense_sims/17_pal5_rvir0.75_hm/" # done. 
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



def structure_over_time(n, i_list): # ADD FUNCTION SO THAT WHEN IT "IS DISSOLVED" IT SETS RH TO ZERO AND MOVES ON.
    """
    compute tha half mass radius of a cluster over time based on
    what is within 100 pc of the core. 
    also compute velocity dispersion within 100 pc of the core.
    """
    rhs_rtid = []
    rhs_100 = []
    rhs_energy = []

    dispersions_rtid = []
    dispersions_100 = []
    dispersions_energy = []

    path=paths[n]
    tidal = paf.load_tidal(path)
    for i in tqdm(i_list):
        rtid = tidal.rtid[i] * u.pc
        # all_particles = paf.load_particle(path, i=i)
        singles = petar.Particle(interrupt_mode='bse', external_mode='galpy')
        singles.loadtxt(path+"data.%i.single"%i)
        vx, vy, vz = singles.vel.T * u.pc/u.Myr
        x,y,z = singles.pos.T * u.pc
        s_r = np.sqrt(x**2 + y**2 + z**2)
        s_v = np.sqrt(vx**2 + vy**2 + vz**2)

        pot_singles = (singles.pot - singles.pot_ext) * u.pc**2 / u.Myr**2

        binaries = petar.Binary(member_particle_type=petar.Particle,
                                interrupt_mode='bse', external_mode='galpy')
        binaries.loadtxt(path+"data.%i.binary"%i)
        vx,vy,vz = binaries.vel.T * u.pc/u.Myr
        x,y,z = binaries.pos.T * u.pc
        b_r = np.sqrt(x**2 + y**2 + z**2)
        b_v = np.sqrt(vx**2 + vy**2 + vz**2)

        pot_binaries = (binaries.p1.pot - binaries.p1.pot_ext) * u.pc**2 / u.Myr**2

        r = np.concatenate([s_r, b_r])
        v = np.concatenate([s_v, b_v])

        pots = np.concatenate([pot_singles, pot_binaries])
        KEs = 0.5 * v**2
        Etots = KEs + pots

        masses = np.concatenate([singles.mass, binaries.mass]) * u.Msun

        selection_rtid = r<rtid
        selection_100 = r<100*u.pc # pc
        selection_energy = Etots <= 0* u.pc**2 / u.Myr**2




        r_sel_rtid = r[selection_rtid]
        v_sel_rtid = v[selection_rtid]
        m_sel_rtid = masses[selection_rtid]

        r_sel_100 = r[selection_100]
        v_sel_100 = v[selection_100]
        m_sel_100 = masses[selection_100]

        r_sel_energy = r[selection_energy]
        v_sel_energy = v[selection_energy]
        m_sel_energy = masses[selection_energy]

        if len(r_sel_rtid) < 100: # handling for if the cluster has dissolved. 
            rh_rtid=0
            rh_100=0
            rh_energy=0
            dispersion_rtid=0
            dispersion_100=0
            dispersion_energy=0

        else:
            rh_rtid = paf.calculate_half_mass_radius(ms=m_sel_rtid.to(u.Msun).value,
                                                     rs=r_sel_rtid.to(u.pc).value)
            rh_100 = paf.calculate_half_mass_radius(ms=m_sel_100.to(u.Msun).value,
                                                     rs=r_sel_100.to(u.pc).value)
            rh_energy = paf.calculate_half_mass_radius(ms=m_sel_energy.to(u.Msun).value,
                                                        rs=r_sel_energy.to(u.pc).value)

            dispersion_rtid = paf.calc_dispersion(v_sel_rtid.to(u.km/u.s).value)
            dispersion_100 = paf.calc_dispersion(v_sel_100.to(u.km/u.s).value)
            dispersion_energy = paf.calc_dispersion(v_sel_energy.to(u.km/u.s).value)


        rhs_rtid.append(rh_rtid)
        rhs_100.append(rh_100)
        rhs_energy.append(rh_energy)
        dispersions_rtid.append(dispersion_rtid)
        dispersions_100.append(dispersion_100)
        dispersions_energy.append(dispersion_energy)


    rhs_rtid = np.array(rhs_rtid)
    rhs_100 = np.array(rhs_100)
    rhs_energy = np.array(rhs_energy)
    dispersions_rtid = np.array(dispersions_rtid)
    dispersions_100 = np.array(dispersions_100)
    dispersions_energy = np.array(dispersions_energy)


    return rhs_rtid, rhs_100, rhs_energy, dispersions_rtid, dispersions_100, dispersions_energy

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_n", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save_to", type=str,
                        default="/n/netscratch/conroy_lab/Lab/amphillips/small_grid/structure_npzs_REAL/")
    args = parser.parse_args()

    n=args.sim_n
    i_list = np.arange(args.start,args.stop+args.step, args.step)
        

    print("computing rh and velocity dispersion over time...")
    rhs_rtid, rhs_100, rhs_energy, dispersions_rtid, dispersions_100, dispersions_energy = structure_over_time(n, i_list)

    print("grabbing data from the tidal file")
    tidal = paf.load_tidal(paths[n])
    select_tidal_data = np.isin(tidal.time.astype(int), i_list)

    rtid = tidal.rtid[select_tidal_data]
    nbound = tidal.n[select_tidal_data]
    mbound = tidal.mass[select_tidal_data]

    print("saving")
    savedir = args.save_to
    np.savez(savedir+str(n)+'_structure',
             i_list = i_list,
             rh_rtid = rhs_rtid,
             rh_100 = rhs_100,
             rh_energy = rhs_energy,
             dispersion_rtid = dispersions_rtid,
             dispersion_100 = dispersions_100,
             dispersion_energy = dispersions_energy,
             nbound = nbound,
             mbound = mbound,
             rtid = rtid,
             allow_pickle=True)
