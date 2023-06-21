import pandas as pd
import numpy as np
from regressionTools import aa_to_idx,aa_to_idx_stop
import regressionTools as rt
from os import listdir

dms_path = '../data/DMS_data/'

def get_x_y_at_3p():
    fin = dms_path+'df_mut_all_norm.csv'
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
    fin = dms_path+'df_at_10pos.csv'
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
    df_pabp = pd.read_csv(dms_path+'pabp_data_melamed_2013.csv', index_col = 0)
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
    df_aav = pd.read_csv(dms_path+'aav_data.csv', index_col = 0)
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
    # creates a df_gfp with the one hot encoding only for positions and mutations at those positions that are individually observed in experiment. Performance is the same if doing standard all amino acids at each position one hot encoding.
    # note: the PDB file 2WUR has 3 mutations (R80Q:T167I:N238K) compared to the wild-type sequence used in experiment
    df_dms = pd.read_csv(dms_path+'gfp_data.csv', index_col = 0)
    # need to reindex the mutkeys to subtract 1
    df_dms['mutant_old'] = df_dms.mutant
    df_dms['mutant'] = df_dms.apply(lambda r: rt.reindex_mut_str(r.mutant_old, 1), axis=1)
    
    # collect a set of unique single mutants, to set which features to make
    
    unique_single_muts = rt.get_unique_single_muts(df_dms.mutant)

    # make dictionary of sequence position to list of mutants at that position
    dic_p_to_list_muts, dic_p_to_wt_aa = rt.get_dic_p_to_list_muts(unique_single_muts)
    # create dictionary of sequence position to dictionary of amino acid and it's associated one hot encoded vector
    dic_p_to_dic_mut_to_oh = rt.create_dic_p_to_dic_mut_to_oh_pos(dic_p_to_list_muts)

    wt_gfp_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

    df_dms['mut_seq'] = df_dms.apply(
        lambda r: rt.make_mut_seq_from_mut_key_and_wt_seq(r.mutant, wt_gfp_seq), axis=1)

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
    df_gb1 = pd.read_csv(dms_path+ 'gb1_data.csv', index_col = 0)
    df_gb1['oh_code'] = df_gb1.apply(lambda r: rt.seq_to_oh(r.sequence, aa_to_idx_stop), axis=1)
    X = np.stack(list(df_gb1['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)
    Y = df_gb1.fitness.values
    return X,Y, df_gb1

def get_x_y_grb2():
    # fetching the full one hot encoding of sequences irrespective of whether individual variants are seen
    df_dms = pd.read_csv(dms_path + 'GRB2_data.csv', index_col = 0)
    df_dms['oh_code'] = df_dms.apply(lambda r: rt.seq_to_oh(r.mutated_region, aa_to_idx_stop), axis=1)
    X = np.stack(list(df_dms['oh_code'].values), axis=0)
    X= np.c_[X,np.ones(X.shape[0])]
    print(X.shape)
    Y = df_dms.DMS_score.values
    return X,Y,df_dms


############################################ reading sampled sequences for GFP 
def read_gfp_sampling_file_coves(fin, wt_gfp_seq_dms):
    # get the mutated keys
    mutkeys = rt.read_sampled_mut_key(fin)
    print(len(mutkeys))
    unique_mutkeys_gen = list(set(mutkeys))
    # filter out wt mutkeys
    unique_mutkeys_gen_no_wt = [':'.join([m for m in mks.split(':') if m[0]!=m[-1]]) for mks in unique_mutkeys_gen]

    # filter out wt mutkeys that are empty
    unique_mutkeys_gen_no_wt = [mk for mk in unique_mutkeys_gen_no_wt if mk != '']

    # reconstruct the mutated sequence from these mutkeys
    unique_mutseq_gen = [rt.make_mut_seq_from_mut_key_and_wt_seq(mk, wt_gfp_seq_dms, offset=-1) for mk in unique_mutkeys_gen_no_wt]

    df_gen_seqs = pd.DataFrame({'gen_seq': unique_mutseq_gen, 'mutkey': unique_mutkeys_gen_no_wt})

    # filter out the gen seqs that don't have any mutants
    df_gen_seqs = df_gen_seqs.loc[df_gen_seqs.mutkey != '']
    df_gen_seqs['oh'] = df_gen_seqs.apply(lambda r: rt.seq_to_oh(r.gen_seq, rt.aa_to_idx), axis=1)

    return df_gen_seqs

def read_gfp_sampling_file_esm(fin):
    df_esm = pd.read_csv(fin, header=None)
    # sampled sequences from ESM lack the 3 fluoropore residues, so need to insert a 'SYG' at position 64 for ESM sampled sequences. 
    unique_mutseq_gen = [s[:64] +'SYG'+s[64:] for s in list(set(df_esm[0].values))]
    # have to make sure the reconstructed sequence has the wt_gfp_seq_dms as background, before making oh
    unique_mutseq_gen = [convert_to_dms_wt_gfp(s) for s in unique_mutseq_gen]
    # get rid of sequence with funny symbols
    unique_mutseq_gen = [s for s in unique_mutseq_gen if rt.seq_with_valid_alphabet(s,rt.AA_ALPHABET)]
    df_gen_seqs = pd.DataFrame({'gen_seq': unique_mutseq_gen})
    df_gen_seqs['oh'] = df_gen_seqs.apply(lambda r: rt.seq_to_oh(r.gen_seq, rt.aa_to_idx), axis=1)
    return df_gen_seqs

def read_gfp_sampling_file_mpnn(fin):
    df_gen = pd.read_csv(fin, header=None)
    # need to replace the 'XXX' a 'SYG' at position 64
    unique_mutseq_gen = [s[:64] +'SYG'+s[67:] for s in list(set(df_gen[0].values))]
    # have to make sure the reconstructed sequence has the wt_gfp_seq_dms as background, before making oh
    unique_mutseq_gen = [convert_to_dms_wt_gfp(s) for s in unique_mutseq_gen]
    df_gen_seqs = pd.DataFrame({'gen_seq': unique_mutseq_gen})
    df_gen_seqs['oh'] = df_gen_seqs.apply(lambda r: rt.seq_to_oh(r.gen_seq, rt.aa_to_idx), axis=1)
    return df_gen_seqs

def get_gfp_model_predictions(model, df_gen_seqs):
    X_gen = np.stack(list(df_gen_seqs['oh'].values), axis=0)
    X_gen= np.c_[X_gen,np.ones(X_gen.shape[0])]
    yhat_non_gen = model.predict(X_gen).flatten()
    return X_gen, yhat_non_gen

def convert_to_dms_wt_gfp(mut_seq, list_to_change = ['R80Q','T167I','N238K']):
    # oracle is trained on the dms-wild-type sequence, rather than the PDB wt sequence.
    # used the wrong end of the wild type sequence to reconstruct the wt sequence in the sampling code.
    #wt_gfp_seq_dms  # has these mutations compared to the wt_pdb sequence: 'R80Q:T167I:N238K'
    #wt_gfp_seq      # has these mutations compared to wt_dms_seq: 'Q80R:I167T:K238N'

    #conv_seq_1 = convert_to_dms_wt(df_gen_seqs.iloc[0].gen_seq)
    #hamming(wt_gfp_seq_dms, conv_seq_1)
    # takes a sequence and converts any potential mutaitons that are PDB wild-type, rather than DMS wild-type reference
    list_mut_seq = list(mut_seq)
    for mk in list_to_change:
        true_ref_aa = mk[-1]
        pos = int(mk[1:-1])-1
        wrong_ref_aa = mk[0]
        if list_mut_seq[pos] == wrong_ref_aa:
            list_mut_seq[pos] = true_ref_aa
    return ''.join(list_mut_seq)

def add_gfp_result_row(model,
                       df_result, 
                        df_gen_seqs, 
                        wt_gfp_seq_dms,
                        muts_seen_dms, 
                        threshold,
                        t,
                        n_pos_mutate,
                        n_sample
                  ):
        # takes one df of generated sequences, filters for valid sequences, and then adds a row to df_result
        # filtering read in files for whether oracle can make predictions about.
        # filter generated sequences by whether they have less than 15 mutations (maximum of oracle trained data)
        df_gen_seqs['hamming'] = df_gen_seqs['gen_seq'].apply(lambda s: rt.hamming(wt_gfp_seq_dms,s))
        df_gen_seqs = df_gen_seqs.loc[df_gen_seqs.hamming <15]
        # filter generated sequences by whether they have all been seen in experiment.
        df_gen_seqs['all_seen'] = df_gen_seqs.mutkey.apply(lambda mk: not bool(sum([1 for m in mk.split(':') if m not in muts_seen_dms])))
        df_gen_seqs = df_gen_seqs.loc[df_gen_seqs.all_seen]

        if len(df_gen_seqs) > 5:
            ###### make predicions 
            # get oracle predictions on the generated sequences.
            X_gen, yhat_non_gen = get_gfp_model_predictions(model, df_gen_seqs)

            # calculate summary statistics
            frac_above_thresh = sum(yhat_non_gen>threshold)/len(yhat_non_gen)
            ham_dists_wt = [rt.hamming(wt_gfp_seq_dms,mutseq) for mutseq in df_gen_seqs['gen_seq']]
            average_wt_hamming = np.mean(ham_dists_wt)

            df_result.loc[len(df_result.index)] = [t, n_pos_mutate, average_wt_hamming, frac_above_thresh, len(df_gen_seqs), n_sample]

def add_mutkey_to_df_gen(df_gen_seqs, wt_gfp_seq_dms, list_muts_dms_exclude):
    # add mutkey wrt to the dms sequence
    df_gen_seqs['mutkey_wrt_dms'] = df_gen_seqs.gen_seq.apply(
        lambda s: rt.make_mut_str_from_2_seqs(wt_gfp_seq_dms,s, muts_only = True, offset=1))

    # these are only the mutants without the reference mutant changes.
    df_gen_seqs['mutkey'] = df_gen_seqs['mutkey_wrt_dms'].apply(
        lambda mks:':'.join([m for m in mks.split(':') if m not in list_muts_dms_exclude]))

def collate_gfp_sampling_results_coves(din, 
                                   model, 
                                   muts_seen_dms, 
                                   wt_gfp_seq_dms, 
                                   threshold,
                                   suffix_filter='230528.csv', 
                                   suffix_exclude = '.txt'):
    # to collate the results of model samples across temperatures, n_pos_mutate and sample number

    df_coves = pd.DataFrame(columns=['t', 'max_mut', 'average_wt_hamming', 'frac_above_thresh', 'n_muts_seen', 'n_muts_sampled'])

    fs = [f for f in listdir(din) if f.endswith(suffix_filter) and not f.endswith(suffix_exclude)]
    print(len(fs))
    for f in fs:
        # reading the params from the file name
        t = float(f.split('_')[10][1:])
        n_sample = int(f.split('_')[11][1:])
        n_pos_mutate = int(float(f.split('_')[12][10:]))
        print(f'reading f with t{t}, n_sample{n_sample}, n_pos_mutate{n_pos_mutate}')
        
        # read the res file
        df_gen_seqs = read_gfp_sampling_file_coves(din + f, wt_gfp_seq_dms)
        add_gfp_result_row(model, df_coves, df_gen_seqs, wt_gfp_seq_dms, muts_seen_dms, threshold, t, n_pos_mutate, n_sample)
    return df_coves

def collate_gfp_sampling_results_esm(din, 
                                   model, 
                                   muts_seen_dms, 
                                   wt_gfp_seq_dms,
                                   list_muts_dms_exclude, # list of mutations
                                   threshold,
                                   suffix_filter='.csv'
                                ):
    # to collate the results of model samples across temperatures, n_pos_mutate and sample number
    
    
    df_esm = pd.DataFrame(columns=['t', 'max_mut', 'average_wt_hamming', 'frac_above_thresh', 'n_muts_seen', 'n_muts_sampled'])

    fs = [f for f in listdir(din) if f.endswith(suffix_filter)]
    for f in fs:
        # reading the params from the file name
        t = float(f.split('_')[1][1:])
        n_sample = int(f.split('_')[2][1:])
        n_pos_mutate = int(float(f.split('_')[5][6:-3]))
        print(f'reading f with t{t}, n_sample{n_sample}, n_pos_mutate{n_pos_mutate}')
        
        # read the file
        df_gen_seqs = read_gfp_sampling_file_esm(din + f)
        # 
        add_mutkey_to_df_gen(df_gen_seqs, wt_gfp_seq_dms, list_muts_dms_exclude)
        # add statistic results to df
        add_gfp_result_row(model, df_esm, df_gen_seqs, wt_gfp_seq_dms, muts_seen_dms, threshold, t, n_pos_mutate, n_sample)
    return df_esm

def collate_gfp_sampling_results_mpnn(din, 
                                   model, 
                                   muts_seen_dms, 
                                   wt_gfp_seq_dms,
                                   list_muts_dms_exclude, # list of mutations
                                   threshold,
                                   suffix_filter='.csv'
                                ):
    # to collate the results of model samples across temperatures, n_pos_mutate and sample number
    
    df_mpnn = pd.DataFrame(columns=['t', 'max_mut', 'average_wt_hamming', 'frac_above_thresh', 'n_muts_seen', 'n_muts_sampled'])

    fs = [f for f in listdir(din) if f.endswith(suffix_filter)]
    for f in fs:
        # reading the params from the file name
        t = float(f.split('_')[2][1:])
        n_pos_mutate = int(float(f.split('_')[5][7:]))
        n_sample = int(f.split('_')[6][1:-4])

        print(f'reading f with t{t}, n_sample{n_sample}, n_pos_mutate{n_pos_mutate}')
        
        # read the file
        df_gen_seqs = read_gfp_sampling_file_mpnn(din + f)
        
        # add mutkey
        add_mutkey_to_df_gen(df_gen_seqs, wt_gfp_seq_dms, list_muts_dms_exclude)
        
        # get the results
        add_gfp_result_row(model, df_mpnn, df_gen_seqs, wt_gfp_seq_dms, muts_seen_dms, threshold, t, n_pos_mutate, n_sample)
        
    return df_mpnn