import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


sys.path.append('/n/groups/marks/users/david/github/pareSingleLibrary2/codebase/pairedEnd/ex62_gvp/')
import importlib
import dataTools as dt
importlib.reload(dt)
import regressionTools as rt
importlib.reload(rt)

from regressionTools import log_layer, log_layer_scale_shift

ds = str(sys.argv[1])
print(f'using dataset {ds}')

dout = '/n/groups/marks/users/david/ex62/subsampling_o2/'
n_reps = 3
fold_obs_test = [1,2,3,4,5,6,10, 1000]


if ds == 'at_3p':
    x,y, df_at_3p = dt.get_x_y_at_3p()
    rt.run_subsampling('at_3p', 
                       df_at_3p, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code',
                       fit_val = 'E3',
                       lr = 0.1,
                       epochs = 300,
                       dout = dout,
                       layer = log_layer(1))
if ds == 'at_10p': 
    x,y, df_at_10p = dt.get_x_y_at_10p()

    rt.run_subsampling('at_10p', 
                       df_at_10p, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code',
                       fit_val = 'mean_lrr_norm',
                       lr = 0.1,
                      epochs = 300,
                      dout = dout,
                      layer = log_layer(1))
if ds == 'pabp': 
    df_pabp = dt.load_pabp()
    rt.run_subsampling('pabp', 
                       df_pabp, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code_subset',
                       fit_val = 'log_DMS_score',
                       lr = 5e-4,
                       epochs = 1000,
                       dout = dout, 
                       layer = log_layer_scale_shift(1))
if ds == 'aav': 
    df_aav = dt.load_df_aav()

    rt.run_subsampling('aav', 
                       df_aav, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code_subset',
                       fit_val = 'viral_selection',
                       lr = 5e-4,
                      epochs = 10000,
                      dout = dout,
                      layer = log_layer_scale_shift(1))
if ds == 'gfp': 
    df_gfp = dt.load_df_gfp()

    rt.run_subsampling('gfp', 
                       df_gfp, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code_subset',
                       fit_val = 'DMS_score',
                       lr = 0.005,
                      epochs = 1000,
                      dout = dout,
                      layer = log_layer_scale_shift(1))
if ds == 'gb1':
    x,y,df_gb1 = dt.get_x_y_gb1()

    rt.run_subsampling('gb1', 
                       df_gb1, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test, 
                       oh_col_name = 'oh_code',
                       fit_val = 'DMS_score',
                       lr = 0.0005,
                      epochs = 8000,
                      dout = dout,
                      layer = log_layer_scale_shift(1))
if ds == 'grb2':
    x,y,df_grb2 = dt.get_x_y_grb2()

    rt.run_subsampling('grb2', 
                       df_grb2, 
                       n_reps = n_reps,
                       fold_obs_test = fold_obs_test,
                       oh_col_name = 'oh_code',
                       fit_val = 'DMS_score',
                       lr = 5e-4,
                      epochs = 4000,
                      dout = dout,
                      layer = log_layer_scale_shift(1))