ph = "/n/netscratch/conroy_lab/Lab/amphillips/small_grid/"

path0 = ph+"0_circ_rvir1.5_lm/"
path1 = ph+"1_circ_rvir1.5_hm/"
path2 = ph+"2_circ_rvir3_lm/"
path3 = ph+"3_circ_rvir3_hm/"
path4 = ph+"4_circ_rvir6_lm/"
path5 = ph+"5_circ_rvir6_hm/"

path6 = ph+"6_gd1_rvir1.5_lm/"
path7 = ph+"7_gd1_rvir1.5_hm/"
path8 = ph+"8_gd1_rvir3_lm/"
path9 = ph+"9_gd1_rvir3_hm/"
path10 = ph+"10_gd1_rvir6_lm/"
path11 = ph+"11_gd1_rvir6_hm/"

labels=["circ_rvir1.5_lm", "circ_rvir1.5_hm","circ_rvir3_lm","circ_rvir3_hm","circ_rvir6_lm","circ_rvir6_hm",
        "gd1_rvir1.5_lm","gd1_rvir1.5_hm","gd1_rvir3_lm","gd1_rvir3_hm","gd1_rvir6_lm","gd1_rvir6_hm"]

paths = [path0,path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11]

circ_indices = [0,1,2,3,4,5]#,8,10] # only including finished sims.
gd1_indices = [6,7,8,9,10,11] # UNTIL SIM2 finishes.

script_path = "/n/home02/amphillips/binaries_in_streams/scripts" # for cannon


import petar
import numpy as np
from tqdm import tqdm

new_path = "/n/netscratch/conroy_lab/Lab/amphillips/small_grid/6c_gd1_rvir1.5_lm_removals/"
dat_file = "data.3550"

problem_IDs = np.array([567,568, 819,2971])
N_problems = len(problem_IDs)

input_file = new_path+dat_file
output_file = input_file ### this script replaces!!

print("updating "+dat_file)

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    # Copy the header
    header = fin.readline()
    header_parts = header.strip().split()

    # Ensure there are at least 2 tokens
    if len(header_parts) < 2:
        raise ValueError("Header does not contain enough elements to update particle count.")

    try:
        N_particles_original = int(header_parts[1])
    except ValueError:
        raise ValueError(f"Second element in header is not an integer: {header_parts[1]}")

    N_particles_new = N_particles_original - N_problems
    header_parts[1] = str(N_particles_new)
    updated_header = ' '.join(header_parts) + '\n'
    
    fout.write(updated_header)
    for line in tqdm(fin):
        # Split line into values
        parts = line.split()
        if not parts:
            continue  # skip empty lines
        
        # Parse column 24 (index 23)
        try:
            id_val = int(float(parts[23]))  # sometimes values are formatted as floats
        except (IndexError, ValueError):
            continue  # skip malformed lines

        # Only write lines not in problem_IDs
        if id_val not in problem_IDs:
            fout.write(line)

            
input_par_path = new_path + "input.par"
output_par_path = input_par_path

print("modifying input.par")
with open(input_par_path, 'r') as f:
    lines = f.readlines()

# Ensure the file has at least 26 lines
if len(lines) < 26:
    raise ValueError("input.par does not have at least 26 lines.")

# Replace line 26 (index 25)
parts = lines[25].strip().split()
if parts[0] != 'I' or parts[1] != 'n':
    raise ValueError(f"Line 26 does not contain expected 'I n' format: {lines[25]}")

# Update the value
parts[2] = str(N_particles_new)
lines[25] = ' '.join(parts) + '\n'

# Write back to file
with open(output_par_path, 'w') as f:
    f.writelines(lines)

print(f"Updated line 26 in input.par with N = {N_particles_new}")