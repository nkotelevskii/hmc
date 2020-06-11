
import os
import sys
import glob
import fileinput
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_rows(tp, min_uc=5, min_sc=5):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def preprocess(RAW_DATA_FILE, dataset, clicks_number=20000000):
    clean_dir = '../data/{}/'.format(dataset)
    if not os.path.exists(os.path.dirname(clean_dir)):
        os.makedirs(os.path.dirname(clean_dir))


    min_uc=5
    min_sc=5
    if dataset == 'foursquare':
        df = pd.read_csv(RAW_DATA_FILE, error_bad_lines=False, nrows=clicks_number, sep='\t', usecols=[0,1], names=['user', 'item'])
        n_heldout_users = 10000
        min_uc=38
        min_sc=80
    if dataset == 'gowalla':
        df = pd.read_csv(RAW_DATA_FILE, error_bad_lines=False, nrows=clicks_number, sep='\t', usecols=[0,1], names=['user', 'item'])
        n_heldout_users = 10000
        min_uc=15
        min_sc=95
    if dataset == 'ml20':
        ml20_ratings = pd.read_csv(RAW_DATA_FILE, error_bad_lines=False, header=0, nrows=clicks_number, usecols=[0, 1, 2], names=['user', 'item', 'rating'])
        ml20_ratings = ml20_ratings[ml20_ratings['rating'] > 3.5]
        n_heldout_users = 10000
        min_uc=55
        min_sc=10

    raw_data, user_activity, item_popularity = filter_rows(df, min_uc=min_uc, min_sc=min_sc)
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print(f"In {dataset}, after filtering, there are %d watching events from %d users and %d items (sparsity: %.3f%%)" %(raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity.index

    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    print(f"tr_users: {tr_users.shape[0]},\n vd_users: {vd_users.shape[0]},\n te_users: {te_users.shape[0]}")

    assert tr_users.shape[0] > 0
    assert vd_users.shape[0] > 0
    assert te_users.shape[0] > 0

    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    def numerize(tp):
        uid = list(map(lambda x: profile2id[x], tp['user']))
        sid = list(map(lambda x: show2id[x], tp['item']))
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    with open(clean_dir + 'unique_sid.txt', 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(train_plays)
    train_data.to_csv(clean_dir + 'train.csv', index=False)

    vad_data_tr = numerize(vad_plays_tr)
    vad_data_tr.to_csv(clean_dir + 'validation_tr.csv', index=False)

    vad_data_te = numerize(vad_plays_te)
    vad_data_te.to_csv(clean_dir + 'validation_te.csv', index=False)

    test_data_tr = numerize(test_plays_tr)
    test_data_tr.to_csv(clean_dir + 'test_tr.csv', index=False)

    test_data_te = numerize(test_plays_te)
    test_data_te.to_csv(clean_dir + 'test_te.csv', index=False)

    sys.exit("preprocess finished")
