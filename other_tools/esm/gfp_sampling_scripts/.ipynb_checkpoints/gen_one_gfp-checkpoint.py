# script to sample from ESM-IF for certain numbers of mutated positions in GFP across temperatures

import esm

import pickle
import numpy as np
import importlib
import pandas as pd
import time
import sys
import torch
import random

random.seed=3

models_dir = "models"
device = "cuda" if torch.cuda.is_available() else None
model_id = float(time.time())
print(device)

t = float(sys.argv[1])  # temperature to sample at
n_pos_mutate = int(sys.argv[2])  # how many positions to mutate
n_seqs = 400  # number of sequences to sample

datapath = "/n/groups/marks/users/david/esm_if/data/gen_seqs/gfp/"
cifpath = '/n/groups/marks/users/david/esm_if/data/2wur.cif' # .pdb format is also acceptable
gfp_din = '/n/groups/marks/users/david/ex62/final_data/DMS_data_from_ada/gfp_data_linear_lr.csv'

# set the output file
pout = datapath + "esm_t{}_n{}_n_pos_mutate{}gen_seq_gfp.csv".format(t, n_seqs, n_pos_mutate)
print("writing to {}".format(pout))


################### setup model
# load model
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

# to get rid of random dropout
model= model.eval()

################## load input structure
structure = esm.inverse_folding.util.load_structure(cifpath, 'A')
coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

################## open dms file to find positions to mutate
# get positions to choose from that are mutated in the experiment
df_dms = pd.read_csv(gfp_din)
#mut_pos_seen = set([m[:-1] for mk in df_dms.mutant for m in mk.split(':')]) # in M1 indexing
# need to limit the mutation range to the ones that are found in PDB
mut_pos_seen=set([m[:-1] for mk in df_dms.mutant for m in mk.split(':') if int(m[1:-1])>=3 and int(m[1:-1])<230])
n_sample=n_seqs

offset = 3 # to subtract from the df_dms indexing to get to ESM mask_pattern indexing that starts with K3
sampled_seqs = []

for i in range(n_sample):
    # choose a n random positions to mutate
    mutpos_to_mutate = random.sample(mut_pos_seen, n_pos_mutate) # without replacement from the positions that are seen
    pos_to_mutate = [int(m[1:])-offset for m in mutpos_to_mutate]
    mask_pattern = list(native_seq)
    for p in pos_to_mutate:
        mask_pattern[p] = '<mask>'
    sampled_seq = 'MS'+model.sample(coords, temperature=t, partial_seq = mask_pattern, device=None) + 'MDELYK' # adding these wt residues
    #write to file
    sampled_seqs.append(sampled_seq)
    # closing and opening because IO issues on cluster.
    fout = open(pout, "w")
    fout.write("\n".join(sampled_seqs))
    fout.close()
    print(sampled_seq)
