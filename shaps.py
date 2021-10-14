import itertools

import numpy as np
import scipy.special


def shap_exact(j, example, model, y_i):
    def powerset(iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

    n_features = example.shape[0]
    mask = np.zeros(n_features, dtype=np.bool)
    res = 0
    for s in powerset(set(range(n_features)).difference([j])):
        weight = scipy.special.comb(n_features - 1, len(s)) * n_features
        mask[:] = 0
        mask[list(s)] = 1
        f_s = model((example * mask).reshape((1, -1)))[:, y_i]
        mask[j] = 1
        f_s_with_i = model((example * mask).reshape((1, -1)))[:, y_i]
        res += (f_s_with_i - f_s) / weight

    return res


def owen_sam_shap(example, model, y_i, runs=2, q_splits=100):
    n_features = example.shape[0]
    phi = np.zeros(n_features)

    s = []
    for _ in range(runs):
        for q_num in range(q_splits + 1):
            q = q_num / q_splits
            s.append(np.array(np.random.binomial(1, q, n_features)))
    s = np.array(s)

    for j in range(n_features):
        mu_2 = example.copy()
        mu_2[j] = 0
        s1 = s * mu_2
        s2 = s1.copy()
        s2[:, j] = example[j]
        item = model(s2) - model(s1)
        phi[j] = np.mean(item[:, y_i])

    return phi


def owen_sam_shap_halved(example, model, y_i, runs=2, q_splits=100):
    n_features = example.shape[0]
    phi = np.zeros(n_features)

    s = []
    for _ in range(runs):
        for q_num in range(q_splits // 2 + 1):
            q = q_num / q_splits
            b = np.array(np.random.binomial(1, q, n_features))
            s.append(b)
            if q != 0.5:
                s.append(1 - b)
    s = np.array(s)

    for j in range(n_features):
        mu_2 = example.copy()
        mu_2[j] = 0
        s1 = s * mu_2
        s2 = s1.copy()
        s2[:, j] = example[j]
        item = model(s2) - model(s1)
        phi[j] = np.mean(item[:, y_i])

    return phi


def castro_sam_shap(example, model, y_i, runs=1):
    n_features = example.shape[0]
    phi = np.zeros(n_features)

    s = {}
    for i in range(n_features):
        s[i] = {0: [], 1: []}

    for r in range(runs):
        p = np.random.permutation(n_features)
        tmp_example = example.copy()

        for i in p:
            s[i][0].append(tmp_example)
            tmp_example = tmp_example.copy()
            tmp_example[i] = 0
            s[i][1].append(tmp_example)

    for i in range(n_features):
        s[i][0] = np.array(s[i][0])
        s[i][1] = np.array(s[i][1])
        phi[i] = np.mean(model(s[i][0])[:, y_i] - model(s[i][1])[:, y_i])

    return phi

def owen_control_sam_shap(example, model, y_i, runs=2, q_splits=100):
    n_features = example.shape[0]
    phi = np.zeros(n_features)
    s = []
    gq = []
    gqm = [0,0]

    for k in range(runs):
        for q_num in range(q_splits + 1):
            q = q_num / q_splits
            s.append(np.array(np.random.binomial(1, q, n_features)))
            gq.append(q)
    s = np.array(s)

    gqm[0] = gq
    gqm[1] = gq
    gq = np.array(gq)
    gqm = np.array(gqm)

    v = np.mean(gq)

    for j in range(n_features):
        mu_2 = example.copy()
        mu_2[j] = 0
        s1 = s * mu_2
        s2 = s1.copy()
        s2[:, j] = example[j]
        item = model(s2) - model(s1)

        a = item.shape[0]
        b = item.shape[1]
        gqm = gqm.reshape(a,b)

        item1 = item + 0.1*(v - gqm)
        phi[j] = np.mean(item1[:, y_i])
    return phi 

def owen_control_sam_shap_halved(example, model, y_i, runs=2, q_splits=100):
    n_features = example.shape[0]
    phi = np.zeros(n_features)
    s = []

    gq = []
    gqm = [0,0]

    for _ in range(runs):
        for q_num in range(q_splits // 2 + 1):
            q = q_num / q_splits
            b = np.array(np.random.binomial(1, q, n_features))
            s.append(b)
            gq.append(q)
            if q != 0.5:
                s.append(1 - b)
                gq.append(1-q)
    s = np.array(s)

    gqm[0] = gq
    gqm[1] = gq
    gq = np.array(gq)
    gqm = np.array(gqm)

    v = np.mean(gq)

    for j in range(n_features):
        mu_2 = example.copy()
        mu_2[j] = 0
        s1 = s * mu_2
        s2 = s1.copy()
        s2[:, j] = example[j]
        item = model(s2) - model(s1)

        a = item.shape[0]
        b = item.shape[1]
        gqm = gqm.reshape(a,b)

        item1 = item + 0.001 * (v - gqm)
        phi[j] = np.mean(item1[:, y_i])

    return phi
