# submit all 

import os 
for pos in [3,4,10]:
    #for t in [1e-3, 1e-2, 1e-1,0.5,1, 10]: # first run
    for t in [0.75, 1.25, 1.5, 1.75, 2, 3, 4, 5]: # second run aug 28 2022
        os.system('sbatch gpujob_gen.sh {} {}'.format(pos, t))
