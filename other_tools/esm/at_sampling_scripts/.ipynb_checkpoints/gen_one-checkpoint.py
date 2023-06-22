# script to sample from ESM-IF for particular mutated positions in antitoxin only
# based on the octameric complex structure.

import esm

import pickle
import numpy as np
import importlib
import pandas as pd
import time
import sys
import torch

models_dir = "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = float(time.time())
print(device)


# reading arguments
datapath = "/n/groups/marks/users/david/esm_if/data/gen_seqs/"
num_pos_mut = int(sys.argv[1])  # number of positions to mutate
t = float(sys.argv[2])  # temperature to sample at
n_seqs = 100  # number of sequences to sample

# set the output file
pout = datapath + "esm_t{}_pos{}_n{}_gen_seq.csv".format(t, num_pos_mut, n_seqs)
print("writing to {}".format(pout))


print("loading model in")
# load model
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
# to get rid of random dropout
model = model.eval()

print("reading structure in")
# read structure in
cifpath = (
    "/n/groups/marks/users/david/esm_if/data/bio_all_rm_non_chain.cif"
)  # .pdb format is also acceptable
coords, seqs = esm.inverse_folding.multichain_util.load_complex_coords(
    cifpath, ["A", "B", "C", "D", "E", "F", "G", "H"]
)
################################################################################
# masking only the mutated positions for designing sequences at these positions
def generate_mask(wt_seq, mut_str_m1, offset=1):
    # makes a list of positions to mask based on a mustring
    mask_list_chC = list(wt_seq)

    for m in mut_str_m1.split(":"):
        wt_aa = m[0]
        aa_pos = int(m[1:-1])
        aa_pos_off = aa_pos - offset

        assert mask_list_chC[aa_pos_off] == wt_aa  # check indexing was right
        mask_list_chC[aa_pos_off] = "<mask>"
    return mask_list_chC


ch_c_mask_3_pos = generate_mask(seqs["C"], "D61A:K64A:E80A", offset=2)
ch_c_mask_4_pos = generate_mask(seqs["C"], "L59A:W60L:D61A:K64L", offset=2)

ch_c_mask_10_pos = generate_mask(
    seqs["C"], "L48L:D52D:I53I:R55R:L56L:F74F:R78R:E80E:A81A:R82R", offset=2
)

if num_pos_mut == 3:
    ch_c_mask = ch_c_mask_3_pos
elif num_pos_mut == 4:
    ch_c_mask = ch_c_mask_4_pos
elif num_pos_mut == 10:
    ch_c_mask = ch_c_mask_10_pos
else:
    print("wrong input given for num positions to mutate")


##############################################
##### sampling ###############################

def _concatenate_coords(coords, target_chain_id, padding_length=10):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chain_id]]
    for chain_id in coords:
        if chain_id == target_chain_id:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated


def sample_sequence_in_complex(
    model,
    coords,
    target_chain_id,
    temperature=1.0,
    padding_length=10,
    mask_pattern=None,
):
    """
    Samples sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: padding length in between chains
    Returns:
        Sampled sequence for the target chain
    """
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = _concatenate_coords(
        coords, target_chain_id
    )  # puts the target chain first

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ["<pad>"] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = "<mask>"

    if mask_pattern != None:
        # make sure the supplied mask pattern is the correct length for the sequence

        assert len(mask_pattern) == target_chain_len
        for i in range(len(mask_pattern)):
            padding_pattern[i] = mask_pattern[i]

    sampled = model.sample(
        all_coords, partial_seq=padding_pattern, temperature=temperature
    )
    sampled = sampled[:target_chain_len]
    return sampled


# actual sampling
sampled_seqs = []
for i in range(n_seqs):
    with torch.no_grad():
        sampled_seq = sample_sequence_in_complex(
            model, coords, "C", temperature=t, mask_pattern=ch_c_mask
        )
        sampled_seqs.append(sampled_seq)
        # closing and opening because IO issues on cluster.
        fout = open(pout, "w")
        fout.write("\n".join(sampled_seqs))
        fout.close()
        print(sampled_seq)
