import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
from Bio.Align import substitution_matrices


def get_norm_prob_log(scores, t=1):
    # getting boltzmann probabilities using scores as energies.
    # assumes that all scores are negative, with the highest number (least negative) being the best.
    # this version gets rid of underflow issues by normalizing the likelihoods by the smalles value
    
    # convert scores to log-probs
    ll = -np.abs(scores)/t
    
    # subtract the largest one so they are not all nan if dealing with small numbers
    ll_norm = np.exp(ll-max(ll))

    p_norm = ll_norm/np.sum(ll_norm)
    return p_norm

# functions for scoring using RES model
def get_norm_probability_log(df, mutant_score, pos, t=1):
    # for a particular mutant score from RES model, get it's site-wise normalized probability
    # subtracting max log_p to not have underflow issues
    df_pos = df.loc[df.pos == pos]
    
    log_p_mut = -np.abs(mutant_score)/t # scalar
    log_p_all = -np.abs(df_pos.mean_x)/t # array

    max_log_p_all = max(log_p_all)
    p_mut_norm_max = np.exp(log_p_mut - max_log_p_all)

    p_all_norm_max = np.exp(log_p_all - max_log_p_all)

    # normalize probabilities to sum to one
    p_norm = p_mut_norm_max/np.sum(p_all_norm_max)
    return p_norm

def test_fn():
    print('import test successful')

def get_joint_log_prob_mutants(df, muts, p_col = 'log_p_t0.1'):
    # assuming independence between positions, give a score based on RES predictions
    # expects muts as a concatenation of 'D5L:R6K'
    # beware of difference in number of elements, cannot compare
    log_prob = 0
    try:
        for m in muts.split(':'):
            log_prob += df.loc[df.mut == m][p_col].values[0]
    except IndexError:
        print('index_error with mut:{}'.format(muts))
    return log_prob

def read_res_pred(fin):
    df_gvp_pred = pd.read_csv(fin)
    df_gvp_pred['pos'] = df_gvp_pred.mut.str[1:-1].astype(int) # m1_indexed
    df_gvp_pred['mut_aa'] = df_gvp_pred.mut.str[-1].astype(str)
    return df_gvp_pred

def add_p_col_df_gvp_log(df_gvp_pred, t=1):
    df_gvp_pred[f'p_t{t}'] = df_gvp_pred.apply(lambda r: get_norm_probability_log(df_gvp_pred,r.mean_x, r.pos, t=t), axis=1)
    df_gvp_pred[f'log_p_t{t}'] = np.log(df_gvp_pred[f'p_t{t}'])
    return df_gvp_pred



###################################################################################################
### sampling functions


def sample_aa_gvp_pos_boltz(aas, scores, t=1, n_sample = 1):
    # takes a 20 list score, and outputs a nmber of samples
    # doing this by assuming probability is proportional to the energy pi prop. exp(e/t)

    #p_norm = get_norm_prob(scores, t=t) # this can give underflow issues with small numbers
    p_norm = get_norm_prob_log(scores, t=t)
    #sample amino acids based on these probabilities
    #print(scores.values, p_norm)
    samples = np.random.choice(aas, n_sample, p=p_norm)
    
    return samples


def sample_gvp_pos_no_w(mut_pos_m1, n_sample = 10, t=1):
    # unweighted sampling of different positions based on RES model   
    # ie. every position considered independently. 
    samples_per_pos = [] # create a list of list of mutation per position
    
    for wt_aa_pos in mut_pos_m1:
        p = int(wt_aa_pos[1:])
        #print(p)
        df_gvp_pred_pos = df_gvp_pred.loc[df_gvp_pred.pos == p]
        muts = df_gvp_pred_pos.mut.values

        # sanity check that I'm getting the right positions and indexing
        assert muts[0][:-1] == wt_aa_pos

        # sample mutations for this position and append to a list
        sampled_aas = sample_aa_gvp_pos_boltz(
            df_gvp_pred_pos.mut_aa, df_gvp_pred_pos.mean_x, 
            t=t, n_sample = n_sample)
        sampled_muts = [wt_aa_pos + mut_aa for mut_aa in sampled_aas]
        samples_per_pos.append(sampled_muts)

    #convert these to sampled mut_str like 'L48R:D52D:I53L:R55L:L56M:F74Y:R78R:E80E:A81A:R82K'
    sampled_mutkeys = []
    for s in range(n_sample):
        samples_muts = []
        for i in range(len(samples_per_pos)):
            samples_muts.append(samples_per_pos[i][s])
        sample_mutkey = ':'.join(samples_muts)
        sampled_mutkeys.append(sample_mutkey)
    return sampled_mutkeys



############################### blosum scores
def blosum_score(str1, str2, subs_matrix = 'BLOSUM45'):
    # choosing subs matrix from names = substitution_matrices.load() 

    blosum = substitution_matrices.load(subs_matrix)

    score = 0
    for i1, i2 in zip(str1, str2):
        try:
            score += blosum.get((i1, i2))
        except TypeError:
            continue
            #print(i1, i2)
    return score

## misc
'''
# can give underflow issues
def get_norm_probability(df, mutant_score, pos, t=1):
    # for a particular mutant score from RES model, get it's site-wise normalized probability
    df_pos = df.loc[df.pos == pos]
    
    p_mut = np.exp(-np.abs(mutant_score)/t) # scalar
    p_all = np.exp(-np.abs(df_pos.mean_x)/t) # array

    # normalize probabilities to sum to one
    p_norm = p_mut/np.sum(p_all)
    return p_norm
'''

'''
# old version gives underflow error with small temperatures.
def get_norm_prob(scores,t=1):
    # takes some scores and returns probabilities
    # get boltzman relative probabilities
    #scores = np.array(scores, dtype='float64')
    #print('scores in get_norm_prob', scores.values)
    p = np.array(np.exp(-np.abs(scores)/t), dtype='float64')
    #print('p in get_norm_prob',p)

    # normalize probabilities to sum to one
    p_norm = p/np.sum(p)
    #print('p_norm in get_norm_prob',p_norm)
    return p_norm
'''

### scoring functions

'''
def get_joint_prob_mutants(df, muts, p_col = 'log_p_t1'):
    # assuming independence between positions, give a score based on RES predictions
    # expects muts as a concatenation of 'D5L:R6K'
    # beware of difference in number of elements, cannot compare
    log_prob = 0
    for m in muts.split(':'):
        log_prob += df.loc[df.mut == m][p_col].values[0]
    return log_prob
'''

'''
def add_p_col_df_gvp(df_gvp_pred, t=1):
    df_gvp_pred[f'p_t{t}'] = df_gvp_pred.apply(lambda r: get_norm_probability(df_gvp_pred,r.mean_x, r.pos, t=t), axis=1)
    df_gvp_pred[f'log_p_t{t}'] = np.log(df_gvp_pred[f'p_t{t}'])
    return df_gvp_pred
'''