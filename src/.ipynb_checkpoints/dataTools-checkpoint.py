import pandas as pd
import numpy as np
from regressionTools import aa_to_idx,aa_to_idx_stop
import regressionTools as rt


dms_orig_path = '/n/groups/marks/users/david/ex62/final_data/original/' # for pabp
dms_path = '/n/groups/marks/users/david/ex62/final_data/DMS_data_from_ada/'
dms_at_path = '/n/groups/marks/users/david/github/CoVES/data/DMS_data/'

def get_x_y_at_3p():
    fin = dms_at_path+'df_mut_all_norm.csv'
    df_mut_all = pd.read_csv(fin, index_col = 0)
    df_mut_all['muts'] = df_mut_all.index
    df_mut_all['hamming'] = df_mut_all.apply(lambda r: np.sum([m[0]!= m[-1] for m in r.muts.split(':')]), axis=1)

    # getting rid of stop codon ones
    df_mut_all = df_mut_all.loc[~((df_mut_all.muts.str[-1] == '_') | (df_mut_all.muts.str[-6] == '_')| (df_mut_all.muts.str[3] == '_'))]

    # make one hot encoding, not considering whether each mutation at each position occurs.
    df_mut_all['mut_seq'] = df_mut_all.apply(lambda r: rt.get_mut_seq_from_mutkey(r.muts), axis=1)
    df_mut_all['oh_code'] = df_mut_all.apply(lambda r: rt.seq_to_oh(r.mut_seq, aa_to_idx), axis=1)
    
    X = np.stack(list(df_mut_all['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])] # adding a bias term

    Y = df_mut_all['E3'].values.flatten() 
    return X, Y, df_mut_all


def get_x_y_at_10p():
    fin = dms_at_path+'df_at_10pos.csv'
    df_10p = pd.read_csv(fin)
    df_10p= df_10p[['pre1', 'post1', 'rr1', 'lrr1', 'pre2','post2', 'rr2', 'lrr2', 'muts', 'stop','mean_lrr','mean_lrr_norm', 'lrr1_norm', 'lrr2_norm']]
    df_10p['hamming'] = df_10p.apply(lambda r: np.sum([m[0]!= m[-1] for m in r.muts.split(':')]), axis=1)

    # get rid of stop codons
    df_10p = df_10p.loc[~df_10p.stop]

    df_10p['mut_seq'] = df_10p.apply(lambda r: rt.get_mut_seq_from_mutkey(r.muts), axis=1)
    # make one hot encoding, not considering whether each mutation at each position occurs.
    df_10p['oh_code'] = df_10p.apply(lambda r: rt.seq_to_oh(r.mut_seq, aa_to_idx), axis=1)
    
    X = np.stack(list(df_10p['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])] # add bias term
    Y = df_10p['mean_lrr_norm'].values
    return X,Y, df_10p

def load_pabp():
    # loading pabp data
    #/content/gdrive/MyDrive/data/pabp/pabp_data_melamed_2013.csv
    df_pabp = pd.read_csv(dms_orig_path+'pabp_data_melamed_2013.csv', index_col = 0)
    # to reindex mutkeys with the minimum position that is mutated
    df_pabp['m1'] = df_pabp.apply(lambda r: r.mutant.split(':')[0], axis=1)
    df_pabp['m2'] = df_pabp.apply(lambda r: r.mutant.split(':')[1], axis=1)
    df_pabp['wt_aa1'] = df_pabp.m1.str[0]
    df_pabp['aa_pos1'] = df_pabp.m1.str[1:-1].astype(int)
    df_pabp['mut_aa1'] = df_pabp.m1.str[-1]
    df_pabp['wt_aa2'] = df_pabp.m2.str[0]
    df_pabp['aa_pos2'] = df_pabp.m2.str[1:-1].astype(int)
    df_pabp['mut_aa2'] = df_pabp.m2.str[-1]
    min_pos = min([*list(df_pabp['aa_pos1'].values), *list(df_pabp['aa_pos2'].values)])
    df_pabp['mutant_reindexed'] = df_pabp.apply(lambda r: rt.reindex_mut_str(r.mutant, min_pos), axis=1)
    print('reindexed using min pos:',min_pos)

    df_pabp['m1_reindexed'] = df_pabp.apply(lambda r: r.mutant_reindexed.split(':')[0], axis=1)
    df_pabp['m2_reindexed'] = df_pabp.apply(lambda r: r.mutant_reindexed.split(':')[1], axis=1)

    df_pabp = df_pabp.rename(columns= {'XY_Enrichment_score': 'DMS_score'})
    
    # to make one hot encoded vectors, using one hot encoding of only mutations that are seen at each position, 
    # and log-enrichment values
    wt_seq = 'GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP'
    
    # make dictionary of position to dictionary of mutation at that position to one-hot encoded amino acid mutation
    unique_single_muts = rt.get_unique_single_muts(df_pabp.mutant_reindexed)
    
    # make dictionary of sequence position to list of mutants at that position
    dic_p_to_list_muts, dic_p_to_wt_aa = rt.get_dic_p_to_list_muts(unique_single_muts)
    # create dictionary of sequence position to dictionary of amino acid and it's associated one hot encoded vector
    dic_p_to_dic_mut_to_oh = rt.create_dic_p_to_dic_mut_to_oh_pos(dic_p_to_list_muts)

    df_pabp['mut_seq'] = df_pabp.apply(
        lambda r: rt.make_mut_seq_from_mut_key_and_wt_seq(r.mutant_reindexed, wt_seq), 
        axis=1)
    
    df_pabp['oh_code_subset'] = df_pabp.apply(
        lambda r: rt.make_oh_mut_pos_only(r.mut_seq, dic_p_to_dic_mut_to_oh), axis=1)
    df_pabp['log_DMS_score'] = np.log(df_pabp['DMS_score'])
    return df_pabp

def get_x_y_pabp(df_pabp):
    X = np.stack(list(df_pabp['oh_code_subset'].values), axis=0)
    # concatenate a column with 1s
    X= np.c_[X, np.ones(X.shape[0])]
    print(X.shape)
    Y = df_pabp.log_DMS_score.values
    return X,Y

def load_df_aav():
    df_aav = pd.read_csv(dms_path+'aav_data_linear_lr_fitted_dd.csv', index_col = 0)
    wt_seq = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    df_aav['mutkeys'] = df_aav.apply(lambda r: rt.make_mut_str_from_2_seqs(wt_seq, r.mutation_sequence), axis=1)

    # fetching the unique single mutants observed in combinations
    unique_single_muts = rt.get_unique_single_muts(df_aav.mutkeys) 
    print('{} individual mutants seen out of possible {}'.format(len(unique_single_muts), 28*20))

    # make dictionary of sequence position to list of mutants at that position
    dic_p_to_list_muts, dic_p_to_wt_aa = rt.get_dic_p_to_list_muts(unique_single_muts)
    # create dictionary of sequence position to dictionary of amino acid and it's associated one hot encoded vector
    dic_p_to_dic_mut_to_oh = rt.create_dic_p_to_dic_mut_to_oh_pos(dic_p_to_list_muts)

    df_aav['mut_seq'] = df_aav.apply(
        lambda r: rt.make_mut_seq_from_mut_key_and_wt_seq(r.mutkeys, wt_seq), axis=1)
    df_aav['oh_code_subset'] = df_aav.apply(
        lambda r: rt.make_oh_mut_pos_only(r.mut_seq, dic_p_to_dic_mut_to_oh), axis=1)

    
    return df_aav

def get_x_y_aav(df_aav):   
    X = np.stack(list(df_aav['oh_code_subset'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)

    Y = (df_aav.viral_selection.values)
    return X,Y

def load_df_gfp():
    df_dms = pd.read_csv(dms_path+'gfp_data_linear_lr.csv', index_col = 0)
    # need to reindex the mutkeys to subtract 1
    df_dms['mutant_old'] = df_dms.mutant
    df_dms['mutant'] = df_dms.apply(lambda r: rt.reindex_mut_str(r.mutant_old, 1), axis=1)
    
    # collect a set of unique single mutants, to set which features to make
    
    unique_single_muts = rt.get_unique_single_muts(df_dms.mutant)

    # make dictionary of sequence position to list of mutants at that position
    dic_p_to_list_muts, dic_p_to_wt_aa = rt.get_dic_p_to_list_muts(unique_single_muts)
    # create dictionary of sequence position to dictionary of amino acid and it's associated one hot encoded vector
    dic_p_to_dic_mut_to_oh = rt.create_dic_p_to_dic_mut_to_oh_pos(dic_p_to_list_muts)

    #### create wt sequence
    # temporary wild-type seq with mutations in it, it does not match all the unique single mutkeys
    wt_seq_original = list(df_dms.iloc[0].mutated_sequence)
    # make a list of of position to wt_aa from the seen unique mutkeys
    list_wt_aa = [dic_p_to_wt_aa[p]+str(p) for p in sorted(dic_p_to_wt_aa.keys())]

    # not all mutants are matching the wt_seq
    #for wt_aa_pos in list_wt_aa:
    #  print(wt_aa_pos, wt_seq_original[int(wt_aa_pos[1:])])

    # create a new wt_seq, based on unique single mutkeys
    wt_seq_new =''
    for i, old_wt_aa in enumerate(wt_seq_original):
        if i in dic_p_to_wt_aa.keys():
            wt_seq_new += dic_p_to_wt_aa[i]
        else:
            wt_seq_new += old_wt_aa # this one is only here to fill the position with something, will not be selected later

    wt_seq = wt_seq_new


    #dic_p_to_dic_mut_to_oh_reindexed = reindex_dic_p_to_dic_mut_to_oh_pos(dic_p_to_dic_mut_to_oh)
    print(min(dic_p_to_dic_mut_to_oh.keys()))
    print(max(dic_p_to_dic_mut_to_oh.keys()))
    print(len(wt_seq))
    df_dms['mut_seq'] = df_dms.apply(
        lambda r: rt.make_mut_seq_from_mut_key_and_wt_seq(r.mutant, wt_seq), axis=1)

    df_dms['oh_code_subset'] = df_dms.apply(
        lambda r: rt.make_oh_mut_pos_only(r.mut_seq, dic_p_to_dic_mut_to_oh), axis=1)
    
    return df_dms

def get_x_y_gfp(df_dms):
    X = np.stack(list(df_dms['oh_code_subset'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)

    Y = df_dms.DMS_score.values
    return X,Y

def get_x_y_gb1():
    # fetching the full one hot encoding of sequences irrespective of whether individual variants are seen
    df_gb1 = pd.read_csv(dms_path+ 'gb1_data_linear_lr_fitted_dd.csv', index_col = 0)
    df_gb1['oh_code'] = df_gb1.apply(lambda r: rt.seq_to_oh(r.sequence, aa_to_idx_stop), axis=1)
    X = np.stack(list(df_gb1['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)
    Y = df_gb1.fitness.values
    return X,Y, df_gb1

def get_x_y_grb2():
    # fetching the full one hot encoding of sequences irrespective of whether individual variants are seen
    df_dms = pd.read_csv(dms_path + 'GRB2_HUMAN_dms_scores_fitted_dd.csv', index_col = 0)
    df_dms['oh_code'] = df_dms.apply(lambda r: rt.seq_to_oh(r.mutated_region, aa_to_idx_stop), axis=1)
    X = np.stack(list(df_dms['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)
    Y = df_dms.DMS_score.values
    return X,Y,df_dms