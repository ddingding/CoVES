# submit all

import os

#for ds in ['gb1']:
#for ds in ["at_3p", "at_10p", "pabp", "aav", "gfp", "gb1", "grb2"]:
for ds in ['gb1', 'aav']:
    os.system("sbatch /n/groups/marks/users/david/github/pareSingleLibrary2/codebase/pairedEnd/ex62_gvp/subsampling_all/subsample_job.sh {}".format(ds))
