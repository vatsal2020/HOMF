import numpy as np


# Computes the predicted rating using either U or V embeddings
def predict(U, bias_u, Test, nrows):
    data = Test.data
    rows = Test.row
    cols = Test.col
    tuple = []
    for c in range(len(data)):
        s = sum(U[rows[c], :]*U[cols[c]+nrows, :]) + bias_u[rows[c]] \
            + bias_u[cols[c] + nrows]
        tuple.append((rows[c], cols[c], data[c], s))
    return tuple


# Computes the predicted rating using both U and V embeddings
def predictuv(U, V, bias_u, bias_v, Test, nrows):
    data = Test.data
    rows = Test.row
    cols = Test.col
    tuple = []
    for c in range(len(data)):
        s = sum(U[rows[c], :]*V[cols[c]+nrows, :]) + bias_u[rows[c]] \
            + bias_v[cols[c] + nrows]
        tuple.append((rows[c], cols[c], data[c], s))
    return tuple


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