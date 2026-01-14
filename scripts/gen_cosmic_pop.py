#-------------------------------------------------#
# STEP 1 OF GENERATING INITIAL CONDITIONS!        #     
# returns a .npz file of                          #
#        primary masses (Msun wo units)           #
#        secondary masses (Msun wo units)         #
#        Ps (days wo units)                       #
#        es                                       #
#        Mtots (of binaries)                      #
#        all_CMs (binary Mtots + single masses)   #
#-------------------------------------------------#

# requirements for the environment
#   astropy
#   numpy
#   scipy
#   COSMIC (with adjusted multidim)

from scipy.optimize import root_scalar
import numpy as np
import astropy.units as u
import pandas as pd

from cosmic.sample.initialcmctable import InitialCMCTable
from cosmic.sample.sampler import cmc
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import multidim


import argparse

# the king profile script
import make_king_profile as king

def run_COSMIC(N_stars_desired, n_samples=int(1e6)):
    """
    generate binary properties, single stars
    N_stars_desired is desired cluster size
    n_samples is # of samples to generate with COSMIC. i think ideally this would be larger than N_stars_desired...
    """
    # sample ~one million binaries with COSMIC
    print("...generating InitialBinaryTable")
    InitialBinaries, mass_singles, mass_binaries, n_singles, n_binaries, singles_mass_list = InitialBinaryTable.sampler('multidim',final_kstar1=[6], final_kstar2=[6],rand_seed=2, nproc=1, SF_start=13700.0, SF_duration=0.0, met=0.0002, size=n_samples) # omitted rand_seed=2

    print("estimating scale for correct # of stars")
    # scale COSMIC outputs to desired cluster size
    def estimate_scale(scale, N_target, n_singles, n_binaries):
        """
        scale = a fraction by which to multiply the number of stars from COSMIC to reach N_target stars total
        returns difference between scaled N and desired N
        """
        N = int(n_singles*scale)+2*(n_binaries*scale)
        diff = N-N_target
        return diff
    
    result = root_scalar(f=estimate_scale, x0=0.5, method='bisect',
                         args=(N_stars_desired, n_singles, n_binaries), bracket=[0,1])
    if result.converged==False:
        print("WARNING! SCIPY SCALE FACTOR FAILED!")
    if result.converged==True:
        print("scipy.optimize.root_scalar successfully converged on a scale factor")
    
    # print("scaled")
    scale = result.root
    N_stars = int(n_singles * scale) + 2*int(n_binaries*scale)
    print("scaled COSMIC outputs to generate %i stars"%N_stars)

    # scale and select binaries from table
    print("sampling InitialBinaryTable and single masses")
    sampled_binaries = InitialBinaries.sample(int(scale*n_binaries), replace=True)
    sampled_binaries["Mtot"] = sampled_binaries["mass_1"] + sampled_binaries["mass_2"]

    # scale and select single masses??
    sel_single_ind = np.random.randint(low=0, high=len(singles_mass_list), size=int(scale*n_singles))
    single_masses = singles_mass_list[sel_single_ind]

    # just for a sanity check
    all_masses = np.concatenate([sampled_binaries['mass_1'], sampled_binaries['mass_2'], single_masses])
    print("sanity check: %i masses generated"%len(all_masses))

    # collect binary properties
    print("concatenating binary properties")
    binary_properties = sampled_binaries[["mass_1", "mass_2", "porb", "ecc", "Mtot"]].values
    primary_masses, secondary_masses, Ps, es, Mtots = binary_properties.T

    # concatenate masses of binary CMs, single stars:
    print("gathering CMs")
    all_CMs = np.concatenate([Mtots, single_masses])

    # add units to everything
    print("adding astropy units")
    primary_masses*=u.Msun
    secondary_masses*=u.Msun
    Mtots*=u.Msun
    all_CMs*=u.Msun

    Ps*=u.day

    return primary_masses, secondary_masses, Ps, es, Mtots, all_CMs #Mtots is binary masses, all_CMs is binary Mtots+single masses (for king profile)


##### MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stars", type=int, default=int(1e4))
    parser.add_argument("--cosmic_samples", type=int, default=1e4) # length of InitialBinaryTable
    parser.add_argument("--f", type=str, default="mass_binary_info") # name of output .npz and .txt files
    args = parser.parse_args()

    N = args.n_stars
    cosmic_samples = args.cosmic_samples
    filename = args.f

    print("running COSMIC function")
    primary_masses, secondary_masses, Ps, es, Mtots, all_CMs = run_COSMIC(N_stars_desired=N, n_samples=cosmic_samples)
    print("done; compiling info")


    m1s = primary_masses.to(u.Msun).value
    m2s = secondary_masses.to(u.Msun).value
    mtots = Mtots.to(u.Msun).value
    all_cms = all_CMs.to(u.Msun).value


    #### # of masses generated
    nbin = len(mtots)
    # print("# of binaries", nbin)
    n_singles = len(all_cms) - len(mtots)
    # print("# of singles", n_singles)
    n_stars_achieved = 2*nbin + n_singles
    # print("resulting # of stars", n_stars_achieved)

    single_masses = all_cms[nbin:]

    all_masses = np.concatenate([m1s, m2s, single_masses])
    high_masses = all_masses[all_masses>=8.0]
    M_highmass = np.sum(high_masses)
    M_cluster = np.sum(all_masses)

    ps = Ps.to(u.day).value


    print(f"saving cosmic population to {filename}.npz")
    np.savez(filename,
             primary_masses=m1s,
             secondary_masses=m2s,
             Mtots=mtots,
             periods=ps,
             es=es,
             all_CMs=all_cms,
             allow_pickle=True)
    
    print(f"saving summary info to {filename}.txt")
    lines = [
        "nstars           : "+str(n_stars_achieved),
        "n_binaries       : "+str(nbin), # for petar, eventually.
        "M_cluster        : "+str(M_cluster)+" Msun",
        "M_>8             : "+str(M_highmass)+" Msun"
    ]


    output_file = filename+".txt"
    with open(output_file, "w") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"info printed to {output_file}")
