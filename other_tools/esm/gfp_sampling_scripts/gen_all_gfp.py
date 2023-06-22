# submit all 

import os 

#for n_pos_mutate in [5,10]:
    #for t in [1e-3,1e-2,1e-1,0.5,0.75,1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 10]: 
    #for t in [20,30,40,70,80,100,200,1000]: 
    
# max pos mutate range scan at low temperatures
#for n_pos_mutate in list(range(5,21,2)):
#    for t in [0.5, 0.75, 1.0]:
        
#temperature scan at max mut 15
for n_pos_mutate in [15]:
    for t in [1e-2,1e-1,0.5,0.75,1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 10]: 
        os.system('sbatch gpujob_gen_gfp.sh {} {}'.format(t, n_pos_mutate))
