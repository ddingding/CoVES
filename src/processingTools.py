def get_read_cut_data(df, read_cut=5, post=False):
    # takes a df_data, and returns data that has been filtered for read counts
    # either just in the pre selection, or also in the post-selection (post=True)
    if post == False:
        df_read = df[(df.pre1>read_cut) &(df.pre2 > read_cut)]
    else:
        df_read = df[(df.pre1>read_cut) &(df.pre2 > read_cut) & (df.post1>read_cut) &(df.post2 > read_cut)]
    X_cut = np.stack(list(df_read['oh1'].values), axis=0)
    X_cut= np.c_[X_cut,np.ones(X_cut.shape[0])] # add bias term
    Y_cut_norm = df_read['mean_lrr_norm'].values
    Y_cut_unnorm = df_read['mean_lrr'].values
    return df_read, X_cut, Y_cut_norm, Y_cut_unnorm