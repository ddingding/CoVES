# submit all 

import os 

os.system('sbatch gpujob_score.sh df_test.csv')
os.system('sbatch gpujob_score.sh df_704_10x_exp_no_stop.csv')
os.system('sbatch gpujob_score.sh df_mut_chris_no_stop.csv')
os.system('sbatch gpujob_score.sh df_mut_all_no_stop.csv')


