import tensorflow as tf
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

import scipy
import seaborn as sns
import random
import itertools

import keras
from scipy.sparse.linalg import lsmr
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import time

from os.path import isfile

# creating OH encodings

AA_ALPHABET_STOP = list("ACDEFGHIKLMNPQRSTVWY_")
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

aa_to_idx_stop = dict(zip(AA_ALPHABET_STOP, range(len(AA_ALPHABET_STOP))))
idx_to_aa_stop = dict(zip(aa_to_idx_stop.values(), aa_to_idx_stop.keys()))

aa_to_idx = dict(zip(AA_ALPHABET, range(len(AA_ALPHABET))))
idx_to_aa = dict(zip(aa_to_idx.values(), aa_to_idx.keys()))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


########################### one hot encoding functions ###########################
def aa_to_oh(aa, dic_aa_to_oh_pos=aa_to_idx_stop):
    oh = np.zeros(len(dic_aa_to_oh_pos))
    oh[dic_aa_to_oh_pos[aa]] = 1
    return oh


def oh_to_aa(oh, dic_idx_to_aa=idx_to_aa_stop):
    # take a one-hot encoding and return the amino acid
    pos = np.where(oh == 1)[0]
    return dic_idx_to_aa[pos]


def seq_to_oh(aa_seq, d_aa_to_oh_pos):
    # amino acid sequence to one hot encoding
    oh_all = np.array([])
    for aa in aa_seq:
        oh = aa_to_oh(aa, d_aa_to_oh_pos)
        oh_all = np.concatenate([oh_all, oh])
    return oh_all


##### to make one hot encoded vectors of just the individual variants that are observed during in a dataset
def make_oh_mut_pos_only(mut_seq, dic_p_to_dic_mut_to_oh):
    # takes a mutated sequence, and fetch the one hot encoding for each position,
    # and returns a truncated OH sequence with only mutants and positions of mutants
    # observed in dataset

    # convert full mut_str to oh
    list_full_oh = []
    for full_seq_pos in sorted(dic_p_to_dic_mut_to_oh.keys()):
        aa = mut_seq[full_seq_pos]
        try:  # catching mutants not seen in the dic_p_to_dic_mut_to_oh
            list_full_oh.append(dic_p_to_dic_mut_to_oh[full_seq_pos][aa])
        except KeyError:
            print(full_seq_pos, aa, dic_p_to_dic_mut_to_oh[full_seq_pos], mut_seq)
    return np.concatenate(list_full_oh)


def create_dic_p_to_dic_mut_to_oh_pos(dic_p_to_list_muts):
    # create a dictionary of sequence position to dictionary of amino acid mutant to index in the one-hot vector
    dic_p_to_dic_mut_to_oh_pos = {}
    for p, l_muts in dic_p_to_list_muts.items():
        dic_p_to_dic_mut_to_oh_pos[p] = dict(zip(l_muts, list(range(len(l_muts)))))

    # create dictionary of position to dictionary of mutations to one hot encoding
    dic_p_to_dic_mut_to_oh = {}
    for p, dic_mut_to_oh_pos in dic_p_to_dic_mut_to_oh_pos.items():
        dic_mut_to_oh = dict(
            zip(
                list(dic_mut_to_oh_pos.keys()),
                [aa_to_oh(aa, dic_mut_to_oh_pos) for aa in dic_mut_to_oh_pos.keys()],
            )
        )
        dic_p_to_dic_mut_to_oh[p] = dic_mut_to_oh

    # print(dic_p_to_dic_mut_to_oh[175])
    return dic_p_to_dic_mut_to_oh


def get_dic_p_to_list_muts(unique_single_muts):
    # unique_single_muts is a set of all the unique single mutants seen for which one hot encoding should be constructed.
    # returns a dictionary of sequence position to possible mutations
    dic_p_to_list_muts = {}
    dic_p_to_wt_aa = {}
    print("# unique single muts", len(unique_single_muts))

    for m in unique_single_muts:
        p = int(m[1:-1])
        mut_aa = m[-1]
        wt_aa = m[0]
        dic_p_to_wt_aa[p] = wt_aa

        if p not in dic_p_to_list_muts:
            dic_p_to_list_muts[p] = [m[-1]]
        else:
            dic_p_to_list_muts[p].append(m[-1])

        # add the wt_aa to the possible mutations at a given position
        if wt_aa not in dic_p_to_list_muts[p]:
            dic_p_to_list_muts[p].append(wt_aa)

    # sort these for better one hot encoding later
    for p, list_muts in dic_p_to_list_muts.items():
        dic_p_to_list_muts[p] = sorted(list_muts)

    # print(dic_p_to_list_muts)
    return dic_p_to_list_muts, dic_p_to_wt_aa


########################### manipulating mutkeys (e.g. 'L48K:E59D')  ###########################


def get_mut_seq_from_mutkey(mutkey):
    # get mutations from a mutkey 'L48K:E59D' --> 'KD'
    return "".join([m[-1] for m in mutkey.split(":")])


def get_muts_from_mutkeys(mutkeys):
    # get mutation sequence list for a list of mutkeys ['L48K:E59D'] --> ['KD']
    muts = [get_mut_seq_from_mutkey(mutkey) for mutkey in mutkeys]
    return muts


def read_sampled_mut_key(fin):
    # just reads a
    muts = []
    for l in open(fin, "r"):
        # some ESM sequences have <eos> in them
        if "<" not in l:
            muts.append(l.rstrip())
        else:
            print("""found '<' in file: """, fin)
    return muts


def reindex_mut_str(mut_str, index_to_subtr):
    # subtract a number from each mutkey position
    list_new_mut_str = []
    for m in mut_str.split(":"):
        list_new_mut_str.append(m[0] + str(int(m[1:-1]) - index_to_subtr) + m[-1])
    reidx_mut_str = ":".join(list_new_mut_str)
    return reidx_mut_str


def make_mut_str_from_2_seqs(wt_seq, mut_seq):
    # expect wild-type sequence and mut seq where every non-mutated position is lower case
    list_muts = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if mut_aa != "_":
            mut = wt_aa + str(i) + mut_aa
            list_muts.append(mut)
    return ":".join(list_muts)


def make_mut_seq_from_mut_key_and_wt_seq(mut_key, wt_seq):
    # take a mut_keys and wild-type sequence, and return a sequence with the correct mutations in it

    muts = mut_key.split(":")

    list_seq = list(wt_seq)
    for m in muts:
        wt_aa = m[0]
        aa_pos = int(m[1:-1])
        mut_aa = m[-1]
        try:
            list_seq[aa_pos] = mut_aa
            try:
                assert wt_seq[aa_pos] == wt_aa  # check indexing correction went ok
            except AssertionError:
                print(aa_pos, wt_seq[aa_pos], wt_aa, m, muts)
        except IndexError:
            print(aa_pos, m)

    final_mut_seq = "".join(list_seq)

    assert hamming(final_mut_seq, wt_seq) == len(
        muts
    )  # check only desired mutations introduced
    assert len(final_mut_seq) == len(
        wt_seq
    )  # check no error in making the full mutated sequence

    return final_mut_seq


def get_unique_single_muts(list_mutkeys):
    # takes a list of mutkeys ie. ['L47L:D63A', ...], and returns a list of unique individual variants
    tot_list_muts = []
    for mutkey in list_mutkeys:
        for m in str(mutkey).split(":"):
            tot_list_muts.append(m)
    unique_single_muts = set(tot_list_muts)
    return unique_single_muts


############################################
def hamming(str1, str2):
    assert len(str1) == len(str2)
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def get_pairwise_hammings(muts, ref_seq=None):
    # take a list of mutations as string, like ['ACADK', 'ACADD']
    # and either return all pairwise hamming distances, or hamming distances to a ref_seq if given
    if ref_seq == None:
        pair_order_list = list(itertools.combinations(muts, 2))
        hamming_dists = [hamming(s1, s2) for (s1, s2) in pair_order_list]

    else:
        hamming_dists = [hamming(s, ref_seq) for s in muts]

    return hamming_dists


def fasta_iter_py3(fasta_name):
    """
	given a fasta file,  yield tuples of header, sequence
	https://github.com/GuyAllard/fasta_iterator/blob/master/fasta_iterator/__init__.py
    """
    rec = None
    for line in open(fasta_name, "r"):
        if line[0] == ">":
            if rec:
                yield rec
            rec = FastaRecord(line.strip()[1:])
        else:
            rec.sequence += line.strip()

    if rec:
        yield rec


class FastaRecord:
    """
    Fasta Record, contains a header and a sequence
	# from https://github.com/GuyAllard/fasta_iterator/blob/master/fasta_iterator/__init__.py
    """

    def __init__(self, header="", sequence=""):
        self.header = header
        self.sequence = sequence

    def __str__(self):
        return ">{}\n{}".format(self.header, self.sequence)


##########################linear regression############################


def calc_lin_reg(X, Y, fout_n=None):
    print(X.shape, Y.shape)
    mod = sm.OLS(Y, X)
    res = mod.fit()
    yhat = res.predict(X)

    return yhat


##########################logistic regression functions and layers############################
# layers used
class log_layer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, reg_strength=0):
        super(log_layer, self).__init__()
        self.num_outputs = num_outputs
        self.reg_strength = reg_strength

    def build(self, input_shape):
        # linear regression weights
        self.w = self.add_weight(
            "w",
            shape=(int(input_shape[-1]), self.num_outputs),
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(self.reg_strength),
        )

    def call(self, input):
        return tf.sigmoid(tf.matmul(input, self.w))


class log_layer_scale_shift(tf.keras.layers.Layer):
    def __init__(self, num_outputs, reg_strength=0):
        super(log_layer_scale_shift, self).__init__()
        self.num_outputs = num_outputs
        self.reg_strength = reg_strength

        # how much to scale by
        scale_init = tf.ones_initializer()
        self.scale = tf.Variable(
            name="scale",
            initial_value=scale_init(shape=(1,), dtype="float32"),
            trainable=True,
        )

        # how much to shift by
        shift_init = tf.ones_initializer()
        self.shift = tf.Variable(
            name="shift",
            initial_value=shift_init(shape=(1,), dtype="float32"),
            trainable=True,
        )

    def build(self, input_shape):
        # linear regression weights
        self.w = self.add_weight(
            "w",
            shape=(int(input_shape[-1]), self.num_outputs),
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(self.reg_strength),
        )

    def call(self, input):
        return (
            tf.multiply(tf.sigmoid(tf.matmul(input, self.w)), self.scale) - self.shift
        )


def fit_log_model(
    Xb,
    Y,
    rand_seed=3,
    my_layer=log_layer(1),  # default layer is log_layer_scale,
    init_linear=True,
    init_list=[],
    adam_lr=1,
    checkpoint_filepath="./tmp/checkpoint",
    val_data=None,
    epochs=200,
    batch_size=1000,
    verbose=True,
):
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)

    # set up the model
    my_log_layer = my_layer
    my_log_layer.build(Xb.shape)
    model = tf.keras.Sequential([my_log_layer])

    # Compile model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr),
    )

    # initialize with linear regression
    beta_hat = lsmr(Xb, Y, show=False)[0]
    # print('beta hat',beta_hat)
    layer0_weights = model.layers[0].get_weights()
    n_weights = len(layer0_weights)
    weights_init = [np.array(8).reshape(1)] * n_weights
    weights_init[-1] = np.array(beta_hat).reshape(-1, 1)
    model.layers[0].set_weights(weights_init)

    if verbose:
        print(weights_init[-1].shape)
        print("layer0 weights set to: ", model.layers[0].get_weights()[-1][:5])
        print("beta_hat", beta_hat[:5])

    # initialize with init_list
    if init_list != []:
        model.layers[0].set_weights(init_list)

    # ah the batch loss is just over the batch. need to take the full epoch loss, that is over the full data.
    batch_weight_history = []
    batch_loss_history = []
    epoch_weight_history = []
    epoch_loss_history = []

    class MyCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            weights = model.get_weights()
            batch_weight_history.append(weights)
            loss = logs["loss"]
            batch_loss_history.append(logs["loss"])

        def on_epoch_end(self, epoch, logs=None):
            weights = model.get_weights()
            epoch_weight_history.append(weights)
            loss = logs["loss"]
            epoch_loss_history.append(logs["loss"])

    callback = MyCallback()

    if val_data == None:
        var_monitor = "loss"
    else:
        var_monitor = "val_loss"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=var_monitor,
        mode="min",
        save_best_only=True,
    )

    # Train the model
    history = model.fit(
        Xb,
        Y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
        callbacks=[callback, model_checkpoint_callback],
        validation_data=val_data,
    )

    if verbose:
        # plot the training loss
        plt.figure(figsize=(2, 2))
        plt.title(
            "history of loss by epoch, min: {}".format(
                min(np.array(history.history["loss"]))
            )
        )
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Magnitude")
        plt.plot(np.array(history.history["loss"]))
        plt.show()

    model.load_weights(checkpoint_filepath)
    return (
        model,
        history,
        [
            batch_weight_history,
            batch_loss_history,
            epoch_weight_history,
            epoch_loss_history,
        ],
    )


def get_logreg_w(model, alphabet, dic_p_list_to_mut_p):
    # model is the log_reg model that is fit from tf
    # alphabet is the alphabet used for choosing as a list or str
    # dic_p_list_to_mut_p maps the position in the list of weighs from 1-num_mutated_pos
    #                       to the actual positions that are mutated
    #                       ie. for the 3 pos AT library: {1:60, 2:63, 3: 79}
    # returns: dictionary of position to

    len_oh = len(alphabet)
    aa_weights_learned = np.array(model.weights[-1])[:-1]  # get rid of bias term
    p_to_mut_to_w_lr = {}
    for p in range(1, len(dic_p_list_to_mut_p) + 1):
        pos_w = aa_weights_learned[((p - 1) * len_oh) : (p) * len_oh].flatten()
        p_to_mut_to_w_lr[dic_p_list_to_mut_p[p]] = dict(zip(alphabet, list(pos_w)))
    return p_to_mut_to_w_lr


################################ subsampling functions ######################################


def get_df_train_test(df_data, train_frac=0.9, rand_seed=3):
    # memory efficient partitioning into df_train and test
    np.random.seed(rand_seed)

    n_train = int(len(df_data) * train_frac)
    n_test = len(df_data) - n_train

    df_data_shuffled = df_data.reindex(np.random.permutation(df_data.index))
    df_data_shuffled.index = list(range(len(df_data_shuffled)))

    df_data_train = df_data_shuffled.iloc[list(range(n_train))]
    df_data_test = df_data_shuffled.iloc[list(range(n_train, n_train + n_test))]
    print(df_data_train.shape)
    print(df_data_test.shape)

    return df_data_train, df_data_test


"""
def get_subset_data(x,y, subset=0.9):
    # taking a n x feature array of X, and target values Y
    # and splitting it randomly into training X and test X,

    n = x.shape[0]
    train_n = int(n*subset)
    test_n = n - train_n
    train_rand_idx = random.sample(list(range(n)), train_n)
    test_rand_idx = [i for i in list(range(n)) if i not in train_rand_idx]

    x_train = x[train_rand_idx,:]
    x_test = x[test_rand_idx,:]

    y_train = y[train_rand_idx]
    y_test = y[test_rand_idx]
    # taking 2
    return x_train, y_train, x_test, y_test
"""


def fit_plot_train_test(
    df_train,
    df_test,
    fit_val="fitness",
    xlim=None,
    ylim=None,
    lr=0.005,
    epochs=200,
    batch_size=1000,
    oh_col_name="oh_code",
    take_log_of_y=False,
    layer=log_layer(1),
    chpt_id=str(int(time.time())),
):

    X = np.stack(list(df_train[oh_col_name].values), axis=0)
    X = np.c_[X, np.ones(X.shape[0])]

    Y = df_train[fit_val].values

    model, history, callbacks = fit_log_model(
        X,
        Y,
        my_layer=layer,
        adam_lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_filepath=f"./tmp/tmp{chpt_id}/checkpoint",
    )

    yhat_non_train = model.predict(X).flatten()
    df_train["yhat_non"] = yhat_non_train

    X_test = np.stack(list(df_test[oh_col_name].values), axis=0)
    X_test = np.c_[X_test, np.ones(X_test.shape[0])]
    print(X_test.shape)
    Y_test = df_test[fit_val].values

    Yhat_non = model.predict(X_test).flatten()
    df_test["yhat_non"] = Yhat_non

    mets = get_corr_metrics(Yhat_non, Y_test)
    print(mets)
    return df_train, df_test, model


def run_subsampling(
    prot_n,
    df_dms,
    n_reps=3,
    fold_obs_test=[1, 2, 5, 10, 1000],
    oh_col_name="oh_code_subset",
    fit_val="DMS_score",
    lr=0.0005,
    epochs=8000,
    batch_size=1000,
    dout="./",
    layer=log_layer(1),
):
    # hack to write file if cell crashes
    if not isfile(dout + "subsampling_" + prot_n + ".csv"):
        fout = open(dout + "subsampling_" + prot_n + ".csv", "w")
        fout.close()

    for fold in fold_obs_test:
        fout = open(dout + "subsampling_" + prot_n + ".csv", "a")

        n_prefs = df_dms[oh_col_name].iloc[0].shape[0]
        fraction_test = float(n_prefs * fold) / float(len(df_dms))
        if fraction_test > 0.98:
            fraction_test = 0.98
        print("fraction of data to test on:", fraction_test, n_prefs * fold)
        for i in range(n_reps):
            df_train, df_test = get_df_train_test(
                df_dms, train_frac=fraction_test, rand_seed=i
            )
            df_train, df_test, model_subset = fit_plot_train_test(
                df_train,
                df_test,
                fit_val=fit_val,
                oh_col_name=oh_col_name,
                xlim=None,
                ylim=None,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                layer=layer,
                chpt_id=str(
                    prot_n + "_" + str(i) + "_" + str(int(time.time()))
                ),  # to make unique checkpoint
            )
            # get test set prediction metrics
            ve_test = get_var_explain(df_test[fit_val], df_test.yhat_non)
            pc_test = pearsonr(df_test[fit_val], df_test.yhat_non)[0]

            # get full dataset prediction metrics
            X_all = np.stack(list(df_dms[oh_col_name].values), axis=0)
            X_all = np.c_[X_all, np.ones(X_all.shape[0])]
            Y_all = df_dms[fit_val].values
            Yhat_all = model_subset.predict(X_all).flatten()

            ve_all = get_var_explain(Yhat_all, Y_all)
            pc_all = pearsonr(Yhat_all, Y_all)[0]

            fout.write(
                "\t".join(
                    [
                        prot_n,
                        str(fraction_test),
                        str(n_prefs * fold),
                        str(n_prefs),
                        str(ve_test),
                        str(pc_test),
                        str(ve_all),
                        str(pc_all),
                        str(i),
                        str(lr),
                        str(epochs),
                    ]
                )
                + "\n"
            )
        fout.close()
        print(f"done: {prot_n}, {fold}, {i}")


########################### analytics of samples ##################################


def get_mse(yhat, y):
    return np.mean((yhat - y) ** 2)


def get_var_explain(yhat, y):
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((yhat - y) ** 2)

    r2 = 1 - ss_res / ss_tot
    # print(r2)
    return r2


def get_corr_metrics(yhat, y):
    ve = get_var_explain(yhat, y)
    pe = pearsonr(yhat, y)[0]
    sp = spearmanr(yhat, y)[0]
    return ve, pe, sp


def plot_corr_marginal(
    x,
    y,
    plot_margin=0.04,
    figsize=(2, 2),
    s=0.3,
    c="black",
    alpha=1,
    ticksize=7,
    fout=None,
    plot_log_hist=False,
    plot_diag=True,
    diag_alpha=0.5,
    diag_col="orange",
    xticks=[0, 1],
    yticks=[0, 1],
    xlim=None,
    ylim=None,
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])

    ax2 = fig.add_subplot(gs[2])
    ax2.scatter(x, y, s=s, alpha=alpha, c="black")
    if plot_diag:
        ax2.plot(
            [min(xticks) - plot_margin, max(xticks) + plot_margin],
            [min(xticks) - plot_margin, max(xticks) + plot_margin],
            color=diag_col,
            alpha=diag_alpha,
        )
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.set_xticklabels(xticks, fontsize=ticksize)
    ax2.set_yticklabels(yticks, fontsize=ticksize)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    ax4 = fig.add_subplot(gs[0], sharex=ax2)
    ax4.hist(x, color=c, density=True, alpha=0.5, log=plot_log_hist)
    ax4.axis("off")

    # plot the distance marginal on the right
    ax1 = fig.add_subplot(gs[3], sharey=ax2)
    ax1.hist(
        y, color=c, density=True, alpha=0.5, orientation="horizontal", log=plot_log_hist
    )
    ax1.axis("off")

    fig.patch.set_visible(False)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)
    ax4.patch.set_visible(False)

    # try not to cutoff xticklabels
    fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)  # has to come after fig.tight_layout

    if fout != None:
        plt.savefig(fout + ".svg", format="svg")
        plt.savefig(fout + ".pdf", format="pdf")
        plt.savefig(fout + ".png", format="png", dpi=800)
    print("var explained {}".format(get_var_explain(x, y)))
    print("mse {}".format(get_mse(x, y)))
    print("pearsonr {}".format(pearsonr(x, y)[0]))
    print("spearmanr {}".format(spearmanr(x, y)[0]))
    plt.show()


def plot_roc(
    df_yhats_y, fname="ROC_10p", n_thresh=20, true_thresh=0.5, fig_size=(1.5, 1.5)
):
    # takes a df with columns 'yobs' and 'yhat_non' and plots a ROC curve for the given true threshold

    print(true_thresh)

    list_tpr = []
    list_fpr = []

    for t_norm in np.array(list(range(n_thresh)) + [n_thresh]) / n_thresh:
        # scale the treshold by the range of the observations
        t = (t_norm * (max(df_yhats_y.yhat_non) - min(df_yhats_y.yhat_non))) + min(
            df_yhats_y.yhat_non
        )
        # TPR: P(true positive| true positive + FN), or sensitivity
        df_ground_truth_positive = df_yhats_y.loc[df_yhats_y.yobs > true_thresh]
        df_TP = df_ground_truth_positive.loc[df_ground_truth_positive.yhat_non > t]
        tpr = len(df_TP) / len(df_ground_truth_positive)
        list_tpr.append(tpr)

        # Specificity or true negative rate: TN| (FP+TN), or
        df_ground_truth_negatives = df_yhats_y.loc[df_yhats_y.yobs < true_thresh]
        df_tn = df_ground_truth_negatives.loc[df_ground_truth_negatives.yhat_non < t]
        tnr = len(df_tn) / len(df_ground_truth_negatives)
        list_fpr.append(1 - tnr)  # since FPR =1-TNR
        # print(f'using t{t}, TPR:{tpr}, FPR:{1-tnr}')

    plt.figure(figsize=fig_size)
    plt.plot(list_fpr, list_tpr, c="black", lw=0.7)
    plt.plot([0, 1], [0, 1], "--", lw=0.5, c="black")
    plt.xticks([0, 0.5, 1], fontsize=7)
    plt.yticks([0, 0.5, 1], fontsize=7)
    plt.ylabel("TPR (Sensitivity)", fontsize=8)
    plt.xlabel("FPR (1-Specificity)", fontsize=8)
    plt.tight_layout()
    plt.savefig(fname + ".svg", format="svg")

    plt.show()

    roc_auc = 1 + np.trapz(list_fpr, list_tpr)
    print("AUC", roc_auc)
