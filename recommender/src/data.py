import os

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from functools import reduce

def get_data_file_names(dataset):
    datasets_paths = {
        'foursquare': ['../data/foursquare/train.tsv', '../data/foursquare/tune.tsv', '../data/foursquare/test.tsv'],
        'gowalla': ['../data/gowalla/train.tsv', '../data/gowalla/tune.tsv', '../data/gowalla/test.tsv'],
        'ml25m': ['../data/ml25m/train.tsv', '../data/ml25m/tune.tsv', '../data/ml25m/test.tsv'],
        'ml100k': ['../data/ml100k/train.tsv', '../data/ml100k/tune.tsv', '../data/ml100k/test.tsv']
    }
    return datasets_paths.get(dataset, "Invalid dataset name")


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


class Dataset():
    def __init__(self, args, data_dir=None):
        self.device = args.device
        self.data = args.data
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        if data_dir:
            DATA_DIR = data_dir + str(self.data)
        else:
            DATA_DIR = '../data/{}'.format(self.data)
        if args.data == 'ml20m':
            pro_dir = os.path.join(DATA_DIR, 'pro_sg')
            unique_sid = list()
            with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
                for line in f:
                    unique_sid.append(line.strip())

            self.n_items = len(unique_sid)
            n_items = self.n_items
            self.train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)
            self.N = self.train_data.shape[0]

            self.vad_data_tr, self.vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                                                 os.path.join(pro_dir, 'validation_te.csv'), n_items)
            self.N_vad = self.vad_data_tr.shape[0]
        elif args.data in {'foursquare', 'gowalla', 'ml25m', 'ml100k'}:
            paths = get_data_file_names(args.data)

            train_file = paths[0]
            tune_file = paths[1]
            test_file = paths[2]

            train_data = pd.read_csv(train_file, sep='\t', names=['u', 'i'])
            val_data = pd.read_csv(tune_file, sep='\t', names=['u', 'i'])
            test_data = pd.read_csv(test_file, sep='\t', names=['u', 'i'])

            all_items = reduce(np.union1d, (train_data.i.unique(), val_data.i.unique(), test_data.i.unique()))

            self.n_items = int(all_items.max())+1
            self.n_users = train_data.u.unique().shape[0]
            print("Loaded:")
            print(f"    - {self.n_users} users")
            print(f"    - {self.n_items} items")
            assert train_data['u'].max() + 1 == self.n_users

            rows, cols = train_data['u'], train_data['i']
            self.train_data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(self.n_users, self.n_items))

            self.N = self.train_data.shape[0]

            joined_df = pd.merge(train_data.groupby('u')['i'].apply(list).reset_index(name='listTr'),
                                 val_data.groupby('u')['i'].apply(list).reset_index(name='listTv'), how='right',
                                 on=['u'])

            vad_data_tr_list = []
            vad_data_te_list = []
            n_val_users = 0
            min_u = 1000000
            max_u = 0
            for _, row in joined_df.iterrows():
                n_val_users += 1
                if row['u'] < min_u:
                    min_u = row['u']
                if row['u'] > max_u:
                    max_u = row['u']
                tr = row["listTr"]
                for item in tr:
                    v = [
                        row['u'],
                        item
                    ]
                    vad_data_tr_list.append(v)
                te = row["listTv"]
                for item in te:
                    v = [
                        row['u'],
                        item
                    ]
                    vad_data_te_list.append(v)
            vad_data_tr_list = np.array(vad_data_tr_list)
            vad_data_te_list = np.array(vad_data_te_list)

            rows_tr = vad_data_tr_list[:,0] - min_u
            cols_tr = vad_data_tr_list[:,1]
            data_tr = np.ones(rows_tr.shape, dtype=int)
            self.vad_data_tr = sparse.coo_matrix((data_tr, (rows_tr, cols_tr)), shape=(max_u - min_u + 1, self.n_items)).tocsr()

            rows_te = vad_data_te_list[:,0] - min_u
            cols_te = vad_data_te_list[:,1]
            data_te = np.ones(rows_te.shape, dtype=int)
            self.vad_data_te = sparse.coo_matrix((data_te, (rows_te, cols_te)), shape=(max_u - min_u + 1, self.n_items)).tocsr()

            self.N_vad = self.vad_data_tr.shape[0]
            assert self.N_vad == self.vad_data_te.shape[0]
            assert self.vad_data_te.shape == self.vad_data_tr.shape
        else:
            raise ModuleNotFoundError

    def next_train_batch(self):
        """
        Training batches will reshuffle every epoch and involve dynamic
        binarization
        """
        idxlist = np.arange(self.N)
        np.random.shuffle(idxlist)
        for bnum, st_idx in enumerate(range(0, self.N, self.train_batch_size)):
            end_idx = min(st_idx + self.train_batch_size, self.N)
            X = self.train_data[idxlist[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            print(X.shape)
            yield X

    def next_val_batch(self):
        """
        Validation batches will be used for ELBO estimates without importance
        sampling (could change)
        """
        idxlist_vad = np.arange(self.N_vad)
        for bnum, st_idx in enumerate(range(0, self.N_vad, self.val_batch_size)):
            end_idx = min(st_idx + self.val_batch_size, self.N_vad)
            X = self.vad_data_tr[idxlist_vad[st_idx:end_idx]]
            X_ = self.vad_data_te[idxlist_vad[st_idx:end_idx]]
            print(X.shape)
            print(X_.shape)
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            yield X, X_

    def next_test_batch(self):
        """
        Test batches are same as validation but with added binarization
        """
        for test_batch in self.test_dataloader:
            yield test_batch
