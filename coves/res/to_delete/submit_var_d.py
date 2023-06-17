import os


for d_cutoff in [4.5,5,6,8,10,15,20,30]:
    my_cmd = 'sbatch gpujob_gvp_vanilla_var_d.sh {}'.format(d_cutoff)
    print(my_cmd)
    os.system(my_cmd)