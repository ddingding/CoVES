
# open the file and close file each iteration, clanky, but had some system file issues with keeping file open.

import esm

import pickle
import numpy as np
import importlib
import pandas as pd
import time
import sys
import torch

models_dir = 'models'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())
print(device)


# batch score
datapath = '/n/groups/marks/users/david/esm_if/data/seq_to_score/'
#print(sys.argv)
f_name = sys.argv[1]
#f_name = 'df_mut_all_no_stop.csv'

pout = datapath + f_name.rstrip('.csv') + '_scores.csv'
print('writing to {}'.format(pout))
df_to_score = pd.read_csv(datapath+ f_name)

#df_to_score = df_to_score[:10]

print('loading model in')
# load model
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
# to get rid of random dropout
model= model.eval()

print('reading structure in')
# read structure in 
cifpath = '/n/groups/marks/users/david/esm_if/data/bio_all_rm_non_chain.cif' # .pdb format is also acceptable
coords, seqs = esm.inverse_folding.multichain_util.load_complex_coords(
    cifpath, 
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
)


# reading in scores that are already done
list_write = []
muts_done = {}
# read in all the lines that have already been done
f_done = open(pout, 'r')
for l in f_done:
    list_write.append(l.rstrip('\n'))
    mut_m1 = l.split(',')[0]
    muts_done[mut_m1] = 1
#print(list_write)
f_done.close()


print('starting to score')
c = 0
for n,r in df_to_score.iterrows():
    mut_str = r.muts_m1
    
    if mut_str not in muts_done:
        print('scoring one...')
        seq_to_score = r.mut_seq_chC

        start = time.time()
        print('start scoring')
        ll_fullseq, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
            model, 
            alphabet,
            coords,
            'C',
            seq_to_score
        )
        write_line = ','.join([mut_str, seq_to_score, str(ll_fullseq), str(ll_withcoord)]) #+ '\n'
        list_write.append(write_line)
        end = time.time()
        it_time = end- start
        print('one it of scoring took {} seconds, total hrs expected to complete:{}'.format(it_time, it_time * (len(df_to_score)-c)/60/60))
        print('just scored {}'.format(mut_str))

        fout = open(pout, 'w')
        fout.write('\n'.join(list_write))
        fout.close()
    c+=1