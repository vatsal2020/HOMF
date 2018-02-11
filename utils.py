#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:08:57 2016

@author: raon
"""

import scipy.sparse as ss
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
import numpy.random as nr


# THE FUNCTIONS BELOW RETURN VECTOR OF THE FORM
# a + A*a + A^2*a ... for columns and rows
def colsample(A, colinds, T):
    a1 = A[:, colinds]
    v = A[:, colinds]
    for t in range(T-1):
        v = a1 + A * v
    return v.toarray()


def rowsample(A, rowinds, T):
    a1 = A[rowinds, :]
    v = A[rowinds, :]
    for t in range(T-1):
        v = a1 + v*A
    return v.toarray()


# The functions below perform ridge update of column and row variables
def colupdate(y, U, regularizer, cgiter=10):
    y = np.ravel(y)
    ids = np.ravel(np.argwhere(y != 0))
    if len(ids) > 0:
        clf = Ridge(alpha=regularizer, max_iter=cgiter, solver='sparse_cg',
                    fit_intercept=True)
        clf = clf.fit(U[ids, :], y[ids])
        vhat = clf.coef_
        bias = clf.intercept_
    else:
        bias = 0
        vhat = np.zeros((U.shape[1],))
    return vhat, bias


def rowupdate(y, V, regularizer, cgiter=10):
    y = np.ravel(y)
    ids = np.ravel(np.argwhere(y != 0))
    if len(ids) > 0:
        clf = Ridge(alpha=regularizer, max_iter=cgiter, solver='sparse_cg',
                    fit_intercept=True)
        clf = clf.fit(V[ids, :], y[ids])
        uhat = clf.coef_
        bias = clf.intercept_
    else:
        bias = 0
        uhat = np.zeros((V.shape[1],))
    return uhat, bias


# The following function converts the data into a scipy.sparse matrix
def load_data(fname):
    c = 0
    with open(fname) as f:
        row, col, data = [], [], []
        for line in f:
            if c == 0:
                vals = line.strip('\r').split(',')
                num_rows = int(vals[0])
                num_cols = int(vals[1])
                c += 1
            else:
                vals = line.strip('\n').split(',')
                rowval = int(float(vals[0]))
                colval = int(float(vals[1]))
                row.append(rowval)
                col.append(colval)
                data.append(float(vals[2]))
    X = ss.coo_matrix((data, (row, col)), shape=(num_rows, num_cols))
    return X


# Create the transition probability matrix in absence of any side
# information graphs
def make_A_nosi(X):
    from sklearn.preprocessing import normalize
    X = ss.csr_matrix(X)
    X1 = normalize(X, norm='l1', axis=1)
    X = ss.csc_matrix(X)
    X2 = normalize(X, norm='l1', axis=0)
    A = ss.bmat([[None, X1], [X2.T, None]])
    return A


# Create the transition probability matrix when either or both side
# information graphs may be present
def make_A_si(X, alpha=1, rowlink=None, collink=None):
    if rowlink is None and collink is None:
        A = make_A_nosi(X)
        return A
    RL, RC = None, None
    if rowlink is not None:
        c = 0
        with open(rowlink) as f:
            row, col, data = [], [], []
            for line in f:
                if c == 0:
                    vals = line.strip('\n').split(',')
                    p = int(vals[0])
                    c += 1
                else:
                    vals = line.strip('\n').split(',')
                    rowval = int(float(vals[0]))
                    colval = int(float(vals[1]))
                    row.append(rowval)
                    col.append(colval)
                    data.append(float(vals[2]))
                    row.append(colval)
                    col.append(rowval)
                    data.append(float(vals[2]))
        RL = ss.coo_matrix((data, (row, col)), shape=(p, p))
        RL = RL*(1-alpha)
    if collink is not None:
        c = 0
        with open(collink) as f:
            row, col, data = [], [], []
            for line in f:
                if c == 0:
                    vals = line.strip('\n').split(',')
                    p = int(vals[0])
                    c += 1
                else:
                    vals = line.strip('\n').split(',')
                    rowval = int(float(vals[0]))
                    colval = int(float(vals[1]))
                    row.append(rowval)
                    col.append(colval)
                    data.append(float(vals[2]))
                    row.append(colval)
                    col.append(rowval)
                    data.append(float(vals[2]))
        RC = ss.coo_matrix((data, (row, col)), shape=(p, p))
        RC = RC*(1-alpha)
    A = ss.bmat([[RL, X*alpha], [X.T*alpha, RC]])
    A = normalize(A, norm='l1', axis=1)
    return A


# THE FUNCTIONS BELOW CREATE THE "f(X)" matrices
def function_transform(R, ptype='linear'):
    if ptype == 'linear':
        return R
    elif ptype == 'exp':
        d = R.data
        d = np.exp(d)
        R.data = d
        return R
    elif ptype == 'step':
        d = np.ones(R.data().shape)
        R.data = d
        return R


# Initialize embedding matrix using scaled normal distribution
def initvars(p, k, rho=0.01):
    U = nr.randn(p, k)/rho
    V = nr.randn(p, k)/rho
    return U, V


# Precision is basically the average of total number of relevant
# recommendations by the top n recommendations for each user.
def cal_precision(dicTopn, n, thr):
    def getkey(tp):
        return tp[1]
    num_good_user = 0.0
    Prec = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        x_sorted = sorted(x, key=getkey, reverse=True)
        sumP = 0.0
        num_good_user += 1.0
        for i in range(n):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
        Prec += sumP/n
    if num_good_user < 1.0:
        print('no valid users, ERROR metric')
        return 0.0
    Prec = Prec/num_good_user
    return Prec


# Recall is the number of relevant items in the top n recommendations divided
# by the total number of relevant items (which can be maximum of n)
def cal_recall(dicTopn, n, thr):
    def getkey(tp):
        return tp[1]
    num_good_user = 0.0
    Rec = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        act_tot = 0.0
        for i in range(len(x)):
            if x[i][0] >= thr:
                act_tot += 1.0
        if act_tot < 1.0:
            continue  # skip users without '1''s in ground truth
        x_sorted = sorted(x, key=getkey, reverse=True)
        sumP = 0.0
        num_good_user += 1.0
        for i in range(n):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
        Rec += float(sumP)/act_tot
    if num_good_user < 1.0:
        print('no valid users, ERROR metric')
        return 0.0
    Rec = Rec/num_good_user
    return Rec


# Average Precision is the average of precision at which relevant items are
# recorded among the top n recommendations.
# MAP is the mean of the average precision over all the users.
def cal_map(dicTopn, n, thr):
    def getkey(tp):
        return tp[1]
    MAP = 0.0
    num_good_user = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        act_tot = 0.0
        for i in range(len(x)):
            if x[i][0] >= thr:
                act_tot += 1.0
        if act_tot < 1.0:
            continue  # skip users without '1''s in ground truth
        x_sorted = sorted(x, key=getkey, reverse=True)
        sumP = 0.0
        ap = 0.0
        num_good_user += 1.0
        upper = min(n, len(x))
        for i in range(upper):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
                ap += sumP/float(i+1.0)
        MAP += ap/min(upper, act_tot)
    if num_good_user < 1.0:
        print('no valid users, ERROR metric')
        return 0.0
    MAP = MAP/num_good_user
    return MAP


# Normalized Discounted Cumulative Gain (NDCG) is normal discounted
# cumulative gain. IDCG is calculated based on the actual top N
# recommendations while DCG is calculated based on the predicted top N.
# NDCG = DCG/IDCG. NDCG@N applies to 2**x - 1 function on each rating before
# multiplying top ith item by 1/log2(i+1)
def cal_ndcg(dicTopn, n, thr):
    def getkeydcg(tp):
        return tp[1]  # Predicted

    def getkeyidcg(tp):
        return tp[0]  # True
    NDCG = 0.0
    num_good_user = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        dcg = 0.0
        idcg = 0.0
        num_good_user += 1.0
        sorted_x1 = sorted(x, key=getkeydcg, reverse=True)
        for i in range(n):
            dcg += (2**sorted_x1[i][0]-1)/np.log2(i+2.0)
        sorted_x2 = sorted(x, key=getkeyidcg, reverse=True)
        for i in range(n):
            idcg += (2**sorted_x2[i][0] - 1)/np.log2(i+2.0)
        NDCG += dcg/idcg
    if num_good_user < 1.0:
        print('no valid users, ERROR metric')
        return 0.0
    NDCG = NDCG/num_good_user
    return NDCG


# Assuming that we are reading results from saved prediction score file
# each line: userId, movieId, actual_rating, predicted_score
def parsetuples(tuple):
    dic = {}
    for c in tuple:
        uid = c[0]
        mid = c[1]
        entry = {}
        entry['t'] = float(c[2])  # Actual rating
        entry['p'] = float(c[3])  # Predicted score
        if uid not in dic:
            dic[uid] = {}
        dic[uid][mid] = entry
    return dic


# Returns the outputs of evaluation metrics
def Calculate(tuple, n=10, thr=5):
    dicTopn = parsetuples(tuple)
    OutPrec = cal_precision(dicTopn, n, thr)
    OutRec = cal_recall(dicTopn, n, thr)
    OutMAP = cal_map(dicTopn, n, thr)
    OutNDCG = cal_ndcg(dicTopn, n, thr)
    return (OutPrec, OutRec, OutMAP, OutNDCG)
