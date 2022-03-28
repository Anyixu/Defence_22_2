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
    # try:
    result = mean(whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test, x.columns, vectorizer,
                           PGDonly=s_method))
    # except:
    #     try:
    #         result = mean(
    #             whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test, x.columns, vectorizer,
    #                      PGDonly=s_method))
    #     except:
    #
    #         result = 0
    print("mean distance: ", result)
    return result


def wafs(x_feature_tr, x_feature_ts, y_tr, y_ts, k, vectorizer, text, s_method, thread_num, lamda=0.5):
    initial_features = x_feature_tr.columns.tolist()
    best_features = []
    while len(initial_features) > 0 and len(best_features) < k:
        remaining_features = list(set(initial_features)-set(best_features))
        new_g = pd.Series(index=remaining_features, dtype='float64')
        new_s = pd.Series(index=remaining_features, dtype='float64')
        new_gs = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            # print("Thread %i" % name)
            # print(x)
            model = svm.SVC(kernel='linear')
            # model.fit(data[best_features+[new_column]], target)
            print(best_features + [new_column])
            new_g[new_column] = mean(cross_val_score(model, x_feature_tr[best_features + [new_column]], y_tr, cv=5))
            # Revise bellow line when security scoring is finished
            new_s[new_column] = estimate_s(x_feature_ts[best_features + [new_column]], y_ts, vectorizer, text,
                                           s_method=s_method)
            new_gs[new_column] = new_g[new_column] + lamda * new_s[new_column]
        # threads = []
        # remaining = np.array_split(remaining_features, thread_num)
        # # thread1 = Thread("Thread-1", data, target, vectorizer, text, s_method, lamda, best_features,
        # #                  remaining[0], new_g, new_s, new_gs)
        # # thread2 = Thread("Thread-2", data, target, vectorizer, text, s_method, lamda, best_features,
        # #                  remaining[1], new_g, new_s, new_gs)
        # for i in range(thread_num):
        #     th = Thread(target=best_feat, args=(i, data, target, vectorizer, text, s_method, lamda, best_features,
        #                   remaining[i], new_g, new_s, new_gs))
        #     threads.append(th)
        #
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()

        lamda = lamda * (new_s.max() ** -1)
        best_features.append(new_gs.idxmax())
        print("current lambda ", lamda)
        print("current features ", best_features)
        print("current len ", len(best_features))
    return best_features


def best_feat(name, data, target, vectorizer, text, s_method, lamda, best_features, remaining_features, new_g, new_s, new_gs):
    x = 0
    for new_column in remaining_features:
        x += 1
        # print("Thread %i" % name)
        # print(x)
        model = svm.SVC(kernel='linear')
        # model.fit(data[best_features+[new_column]], target)
        print(best_features + [new_column])
        new_g[new_column] = mean(cross_val_score(model, data[best_features + [new_column]], target, cv=5))
        # Revise bellow line when security scoring is finished
        new_s[new_column] = estimate_s(data[best_features + [new_column]], target, vectorizer, text, s_method=s_method)
        new_gs[new_column] = new_g[new_column] + lamda * new_s[new_column]
    return new_s, new_gs


# class Thread(threading.Thread):
#     def __init__(self, name, data, target, vectorizer, text, s_method, lamda, best_features, remaining_features, new_g, new_s, new_gs):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.data = data
#         self.target = target
#         self.vectorizer = vectorizer
#         self.text = text
#         self.s_method = s_method
#         self.lamda = lamda
#         self.best_features = best_features
#         self.remaining_features = remaining_features
#         self.new_g = new_g
#         self.new_s = new_s
#         self.new_gs = new_gs
#         self._lock = threading.Lock()
#
#     def run(self):
#         best_feat(self.name, self.data, self.target, self.vectorizer, self.text, self.s_method, self.lamda, self.best_features, self.remaining_features, self.new_g, self.new_s, self.new_gs)










