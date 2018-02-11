# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:55:41 2016

@author: raon
"""

import numpy as np
import scipy.sparse as ss
from tempfile import TemporaryFile
import sys
from pathos.multiprocessing import ProcessingPool as Pool

import utils as uh
import metrics as mh

outfile = TemporaryFile()


# all udpate functions to invoke later
def update_allcols(ids, U):
    a = uh.colsample(Acsc, ids, T)
    v, biasv = uh.colupdate(a, U, regularizer, cgiter)
    return (v, biasv, ids)


def update_allrows(ids, V):
    a = uh.rowsample(Acsr, ids, T)
    u, biasu = uh.rowupdate(a, V, regularizer, cgiter)
    return (u, biasu, ids)


# Main function
if __name__ == '__main__':
    k = 10                  # RANK
    regularizer = 0.1               # REGULARIZER
    T = 4                   # LENGTH OF WALK
    cgiter = 10             # ITERATIONS OF CONJUGATE GRADIENT
    max_iter = 10           # ITERATIONS OF COORDINATE DESCENT (EPOCHS)
    srow, scol = None, None   # LOCATION OF ROW AND COLUMN GRAPHS
    alpha = 1               # TRADEOFF BETWEEN GRAPH AND RATINGS
    ptype = 'linear'        # TRANSITION PROBABILITY FUNCTION
    thresh = 5              # THRESHOLD TO DETERMINE SUCCESS
    evalmetrics_5, evalmetrics_10 = [], []

    foo = sys.argv
    for i in range(1, len(foo)):
        if foo[i] == '-k':       k = int(float(foo[i+1]))
        if foo[i] == '-train':   train = foo[i+1]
        if foo[i] == '-val':     val = foo[i+1]
        if foo[i] == '-siderow': srow = foo[i+1]
        if foo[i] == '-sidecol': scol = foo[i+1]
        if foo[i] == '-maxit':   max_iter = int(float(foo[i+1]))
        if foo[i] == '-T':       T = int(float(foo[i+1]))
        if foo[i] == '-cg':      cgiter = int(float(foo[i+1]))
        if foo[i] == '-l':       regularizer = float(foo[i+1])
        if foo[i] == '-ptype':   ptype = foo[i+1]
        if foo[i] == '-alpha':   alpha = float(foo[i+1])
        if foo[i] == '-thr':     thresh = float(foo[i+1])
        if foo[i] == '-frac':    frac = float(foo[i+1])

    print('Loading training data ...')
    Ratings = uh.load_data(train)
    numuser = Ratings.shape[0]
    print('Transforming: %s' % (ptype))
    Ratings = uh.function_transform(Ratings, ptype=ptype)
    print('Creating Transition Probability matrix')
    if srow is not None or scol is not None:
        A = uh.make_A_si(Ratings, alpha=alpha, rowlink=srow, collink=scol)
    else:
        A = uh.make_A_si(Ratings)

    p = A.shape[0]
    print('A has {} rows'.format(p))
    Acsr = ss.csr_matrix(A)
    Acsc = ss.csc_matrix(A)

    print('Loading validation data')
    Rv = uh.load_data(val)

    print('Initializing')
    U, V = uh.initvars(p, k, np.sqrt(k))
    bias_u, bias_v = np.zeros((p,)), np.zeros((p,))

    print('Starting HOMF with')
    print('cyclic CD for %d iterations' % (max_iter))
    idset = range(p)
    P = Pool()
    preds = {}
    for t in range(max_iter):
        print('Iter %d' % (t+1))
        Vlist = P.map(update_allcols, idset, [U for i in range(p)])
        for i in range(len(Vlist)):
            V[Vlist[i][2], :] = Vlist[i][0]
            bias_v[Vlist[i][2]] = Vlist[i][1]
        Ulist = P.map(update_allrows, idset, [V for i in range(p)])
        for i in range(len(Ulist)):
            U[Ulist[i][2], :] = Ulist[i][0]
            bias_u[Ulist[i][2]] = Ulist[i][1]

        tmp = mh.predict(U, bias_u, Rv, numuser)
        print("After %d iterations, (Precision@5, Recall@5, MAP@5, NDCG@5)=" % (t), mh.Calculate(tmp, n=5, thr=thresh))
        print("After %d iterations, (Precision@10, Recall@10, MAP@10, NDCG@10)=" % (t), mh.Calculate(tmp, n=10, thr=thresh))

    print('Au revoir, World!')
