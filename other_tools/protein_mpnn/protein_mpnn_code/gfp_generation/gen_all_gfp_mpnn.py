# submit all 

import os 

#for n_pos_mutate in [5,10]:
    #for t in [0.0001, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]: 
    #for t in [0.7, 1.0, 1.5,1.75, 2,2.25, 2.5,3,4,5,10,20,40,70,100]: 
# to scan max pos range at low temps
#for n_pos_mutate in list(range(5,21,2)):
#    for t in [0.7, 1, 1.5]:

# to scan temps for max_pos_mutate 15
for n_pos_mutate in [15]:
    for t in [0.0001, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0,1.5,1.75, 2,2.25, 2.5,5,10]:         
        os.system('sbatch gpujob_gen_gfp_mpnn.sh {} {}'.format(t, n_pos_mutate))
