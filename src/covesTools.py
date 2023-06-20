import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
from Bio.Align import substitution_matrices


def get_norm_prob(scores, t=1):
    # getting boltzmann probabilities using scores as energies.
    # assumes that all scores are negative, with the highest number (least negative) being the best.
    # this version gets rid of underflow issues by shifting the likelihoods by the smallest value

    # convert scores to log-probs
    ll = -np.abs(scores) / t
    # subtract the largest one so they are not all nan if dealing with small numbers
    ll_shift = ll - max(ll)
    # get likelihood
    l = np.exp(ll_shift)
    # normalize
    p_norm = l / np.sum(l)
    return p_norm


def get_norm_probability_df(df, mutant_score, pos, t=1):
    # normalizes the probabilities of a given score in a dataframe
    # for a particular mutant score from RES model, get it's site-wise normalized probability
    # subtracting max log_p to not have underflow issues
    df_pos = df.loc[df.pos == pos]

    log_p_mut = -np.abs(mutant_score) / t  # scalar
    log_p_all = -np.abs(df_pos.mean_x) / t  # array

    max_log_p_all = max(log_p_all)
    p_mut_norm_max = np.exp(log_p_mut - max_log_p_all)
    p_all_norm_max = np.exp(log_p_all - max_log_p_all)

    # normalize probabilities to sum to one
    p_norm = p_mut_norm_max / np.sum(p_all_norm_max)
    return p_norm


def get_joint_log_prob_mutants(df, muts, p_col="log_p_t0.1"):
    # assuming independence between positions, give a score based on RES predictions
    # expects muts as a concatenation of 'D5L:R6K'
    # beware of difference in number of elements, cannot compare
    log_prob = 0
    try:
        for m in muts.split(":"):
            log_prob += df.loc[df.mut == m][p_col].values[0]
    except IndexError:
        print("index_error with mut:{}".format(muts))
    return log_prob


def read_res_pred(fin):
    df_gvp_pred = pd.read_csv(fin)
    df_gvp_pred["pos"] = df_gvp_pred.mut.str[1:-1].astype(int)  # m1_indexed
    df_gvp_pred["mut_aa"] = df_gvp_pred.mut.str[-1].astype(str)
    return df_gvp_pred


def add_p_col_df_gvp_log(df_gvp_pred, t=1):
    df_gvp_pred[f"p_t{t}"] = df_gvp_pred.apply(
        lambda r: get_norm_probability_df(df_gvp_pred, r.mean_x, r.pos, t=t), axis=1
    )
    df_gvp_pred[f"log_p_t{t}"] = np.log(df_gvp_pred[f"p_t{t}"])
    return df_gvp_pred


###################################################################################################
### sampling functions


def sample_aa_gvp_pos_boltz(aas, scores, t=1, n_sample=1, rand_seed=41):
    # takes a 20 list score, and outputs a nmber of samples
    # doing this by assuming probability is proportional to the energy pi prop. exp(e/t)

    np.random.seed(rand_seed)

    # normalize scores
    p_norm = get_norm_prob(scores, t=t)
    # sample amino acids based on these probabilities
    samples = np.random.choice(aas, n_sample, p=p_norm)

    return samples


def sample_coves(df_gvp_pred, mut_pos_m1, n_sample=10, t=1):
    # unweighted sampling of different positions based on RES model

    # create dictionary of position to wt_aa
    dic_pos_to_wt_aa = dict([(int(m[1:]), m[0]) for m in mut_pos_m1])

    # make a list of sampled mutkeys
    sampled_mutkeys = []
    for i in range(n_sample):
        pos_to_sampled_mut = {}

        for wt_aa_pos in mut_pos_m1:
            p = int(wt_aa_pos[1:])
            # print(p)
            df_gvp_pred_pos = df_gvp_pred.loc[df_gvp_pred.pos == p]
            muts = df_gvp_pred_pos.mut.values

            # sanity check indexing
            assert muts[0][:-1] == wt_aa_pos

            # sample mutations for this position and append to a list
            sampled_aa = sample_aa_gvp_pos_boltz(
                df_gvp_pred_pos.mut_aa, df_gvp_pred_pos.mean_x, t=t, n_sample=1
            )
            # print(sampled_aa)
            pos_to_sampled_mut[p] = sampled_aa

        # for this one sample, generate the whole mutkey
        sampled_mutkey_list = []
        for p in sorted(pos_to_sampled_mut.keys()):
            ind_mutkey = dic_pos_to_wt_aa[p] + str(p) + pos_to_sampled_mut[p]
            sampled_mutkey_list.append(ind_mutkey[0])
        sampled_mutkeys.append(":".join(sampled_mutkey_list))

    return sampled_mutkeys


############################### blosum scores
def blosum_score(str1, str2, subs_matrix="BLOSUM45"):
    # choosing subs matrix from names = substitution_matrices.load()

    blosum = substitution_matrices.load(subs_matrix)

    score = 0
    for i1, i2 in zip(str1, str2):
        try:
            score += blosum.get((i1, i2))
        except TypeError:
            continue
            # print(i1, i2)
    return score


"""
def sample_coves(df_gvp_pred, mut_pos_m1, n_sample=10, t=1):
    # unweighted sampling of different positions based on structure scores
    # ie. every position normalized and considered independently.
    samples_per_pos = []  # create a list of list of mutation per position

    for wt_aa_pos in mut_pos_m1:
        p = int(wt_aa_pos[1:])
        df_gvp_pred_pos = df_gvp_pred.loc[df_gvp_pred.pos == p]
        muts = df_gvp_pred_pos.mut.values

        # sanity check right position and indexing
        assert muts[0][:-1] == wt_aa_pos

        # sample mutations for this position and append to a list
        sampled_aas = sample_aa_gvp_pos_boltz(
            df_gvp_pred_pos.mut_aa, df_gvp_pred_pos.mean_x, t=t, n_sample=n_sample
        )
        sampled_muts = [wt_aa_pos + mut_aa for mut_aa in sampled_aas]
        samples_per_pos.append(sampled_muts)

    # convert these to sampled mut_str like 'L48R:D52D:I53L:R55L:L56M:F74Y:R78R:E80E:A81A:R82K'
    sampled_mutkeys = []
    for s in range(n_sample):
        samples_muts = []
        for i in range(len(samples_per_pos)):
            samples_muts.append(samples_per_pos[i][s])
        sample_mutkey = ":".join(samples_muts)
        sampled_mutkeys.append(sample_mutkey)
    return sampled_mutkeys
"""
