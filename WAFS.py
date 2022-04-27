import statistics
from numba import jit
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from statistics import mean
from WhiteBox_WAFS import whitebox
import threading
from threading import Thread
import time


def estimate_s(x, y, vectorizer, text, s_method=False):
    SEED = 2000
    x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=.2, random_state=SEED)
    x_train_features, x_test_features, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=SEED)
    try:
        d, s = whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test, x.columns, vectorizer,
                 PGDonly=s_method)
        result = mean(d)
    # except:
    #     try:
    #         result = mean(
    #             whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test, x.columns, vectorizer,
    #                      PGDonly=s_method))
    except statistics.StatisticsError:
        result = -1
        s = -1
    # print("mean distance: ", result)
    return result, s


def wafs(x_feature_tr, x_feature_ts, y_tr, y_ts, k, vectorizer, text, s_method, thread_num, lamda=0.5):
    initial_features = x_feature_tr.columns.tolist()
    best_features = []
    while len(initial_features) > 0 and len(best_features) < k:
        start_mian = time.time()
        remaining_features = list(set(initial_features)-set(best_features))
        new_g = pd.Series(index=remaining_features, dtype='float64')
        new_s = pd.Series(index=remaining_features, dtype='float64')
        new_gs = pd.Series(index=remaining_features, dtype='float64')
        # feature_count = 0
        # for new_column in remaining_features:
        #     start = time.time()
        #     print("Gone through ",  feature_count, "/", len(remaining_features))
        #     feature_count = feature_count + 1
        #     model = svm.SVC(kernel='linear')
        #     print(best_features + [new_column])
        #     new_g[new_column] = mean(cross_val_score(model, x_feature_tr[best_features + [new_column]], y_tr, cv=5))
        #     # Revise bellow line when security scoring is finished
        #     new_s[new_column], s = estimate_s(x_feature_ts[best_features + [new_column]], y_ts, vectorizer, text,
        #                                    s_method=s_method)
        #     new_gs[new_column] = new_g[new_column] + lamda * new_s[new_column]
        #     print("Time used: ", time.time() - start)

        threads = []
        remaining = np.array_split(remaining_features, thread_num)

        for i in range(thread_num):
            th = Thread(target=best_feat, args=(i, x_feature_tr, x_feature_ts, y_tr, y_ts, vectorizer, text, s_method, lamda, best_features,
                          remaining[i], new_g, new_s, new_gs))
            threads.append(th)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lamda = lamda * (new_s.max() ** -1)
        best_features.append(new_gs.idxmax())
        print("current lambda ", lamda)
        print("current features ", best_features)
        print("current len ", len(best_features))
        print("Total time for selecting this feature: ", start_mian - time.time())
    return best_features


def best_feat(name, x_feature_tr, x_feature_ts, y_tr, y_ts, vectorizer, text, s_method, lamda, best_features, remaining_features, new_g, new_s, new_gs):
    feature_count = 0
    for new_column in remaining_features:
        start = time.time()
        print(name, " gone through ", feature_count, "/", len(remaining_features))
        feature_count += 1
        # print("Thread %i" % name)
        # print(x)
        model = svm.SVC(kernel='linear')
        # model.fit(data[best_features+[new_column]], target)
        print(best_features + [new_column])
        new_g[new_column] = mean(cross_val_score(model, x_feature_tr[best_features + [new_column]], y_tr, cv=5))
        # Revise bellow line when security scoring is finished
        new_s[new_column], s = estimate_s(x_feature_ts[best_features + [new_column]], y_ts, vectorizer, text, s_method=s_method)
        new_gs[new_column] = new_g[new_column] + lamda * new_s[new_column]
        print("Time used: ", time.time() - start)
    return new_s, new_gs




