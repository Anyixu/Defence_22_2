from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM
from secml.adv.attacks.evasion import CAttackEvasionPGD, CAttackEvasionPGDLS, CAttackEvasionPGDExp
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_fgm_attack import CFoolboxFGM
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import threading
import time
from scipy.spatial.distance import cityblock


ham_unique = list(pd.read_csv("ham_unique.csv").columns)


def single_transform(message, feature_names, vectorizer):
    d = vectorizer.transform(message).toarray()
    result = pd.DataFrame(data=d, columns=vectorizer.get_feature_names_out())
    result = result[feature_names]
    # result = pd.DataFrame(vectorizer.transform(message), columns=vectorizer.get_feature_names_out())[feature_names]
    return result


def train_test_SVM(x_train_features, x_test_features, y_train, y_test):
    start = time.time()
    tr_set = CDataset(x_train_features, y_train)
    # Train the SVM
    clf_lin = CClassifierSVM()
    # xval_splitter = CDataSplitterKFold()
    # xval_lin_params = {'C': [1]}
    # best_lin_params = clf_lin.estimate_parameters(
    #     dataset=tr_set,
    #     parameters=xval_lin_params,
    #     splitter=xval_splitter,
    #     metric='accuracy',
    #     perf_evaluator='xval'
    # )
    clf_lin.fit(tr_set.X, tr_set.Y)
    ts_set = CDataset(x_test_features, y_test)
    print("SVM training time: ", time.time() - start)
    return tr_set, ts_set, clf_lin


def pdg_attack(clf_lin, tr_set, ts_set, y_test, feature_names, nb_attack, dmax, lb, ub, just_pgd):
    # print("Starting PGD")
    class_to_attack = 1
    idx_candidates = np.where(y_test == class_to_attack)
    ori_examples2_x = ts_set.X.tondarray()[idx_candidates]
    ori_examples2_y = ts_set.Y.tondarray()[idx_candidates]
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    y_target = 0
    solver_params = {
        'eta': 0.3,
        'max_iter': 100,
        'eps': 1e-4}

    # pgd_attack = CAttackEvasionPGDExp(
    #     classifier=clf_lin,
    #     double_init_ds=tr_set,
    #     distance=noise_type,
    #     dmax=dmax,
    #     lb=lb, ub=None,
    #     solver_params=solver_params,
    #     y_target=y_target
    # )
    pgd_attack = CFoolboxFGM(
        classifier=clf_lin,
        # double_init_ds=tr1,
        # double_init=False,
        # distance=noise_type,
        # dmax=dmax,
        # lb=lb, ub=ub,
        # solver_params=solver_params,
        # y_target=y_target
    )
    x0 = CArray(ori_examples2_x)
    y0 = CArray(ori_examples2_y)

    start = time.time()
    y_pred_pgd, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)
    print("PGD time: ", time.time() - start)
    cnt = y_pred_pgd.size - y_pred_pgd.nnz
    ad_examples_x = adv_ds_pgd.X.tondarray()
    ad_examples_y = y_pred_pgd.tondarray()
    print("PGD success rate:", cnt/len(ori_examples2_x))
    ori_examples2_x = np.array(ori_examples2_x)
    ori_examples2_y = np.array(ori_examples2_y)
    ad_examples_x = np.array(ad_examples_x)
    ad_examples_y = np.array(ad_examples_y)
    ori_dataframe = pd.DataFrame(ori_examples2_x, columns=feature_names)
    ad_dataframe = pd.DataFrame(ad_examples_x, columns=feature_names)

    # extract the success and fail examples
    ad_dataframe['ad_label'] = ad_examples_y
    ad_success = ad_dataframe.loc[ad_dataframe.ad_label == 0]
    ori_success = ori_dataframe.loc[ad_dataframe.ad_label == 0]
    ad_fail = ad_dataframe.loc[ad_dataframe.ad_label == 1]
    ori_fail = ori_dataframe.loc[ad_dataframe.ad_label == 1]

    ad_success_x = ad_success.drop(columns=['ad_label'])
    ad_fail_x = ad_fail.drop(columns=['ad_label'])
    distance = []
    if just_pgd:
        ad_success_x = ad_success_x.to_numpy()
        ori_success = ori_success.to_numpy()
        for i in range(len(ad_success_x)):
            distance.append(cityblock(ori_success[i], ad_success_x[i]))
    result = (ad_success_x - ori_success)
    return result, cnt, ad_success_x, ori_dataframe, ori_examples2_y, distance


def magical_word(x_train, x_test, y_train, y_test, result, cnt):
    x2result1 = result
    x2result1 = np.array(x2result1)
    x2result = result
    x2result = x2result.multiply(x2result1)

    sum_number = x2result.sum() / cnt
    sum_number = pd.DataFrame(sum_number, columns=['sum_number'])
    sum_number = sum_number.sort_values(
        by='sum_number', ascending=False, inplace=False)
    sum_number_pd = pd.DataFrame(sum_number.index[:100])
    # sum_number_pd.to_csv("x2result.csv")
    print(sum_number_pd)
    d = {'message': x_train, 'label': y_train}
    df = pd.DataFrame(data=d)
    d1 = {'message': x_test, 'label': y_test}
    df1 = pd.DataFrame(data=d1)
    frames = [df, df1]
    messages = pd.concat(frames)
    spam = messages[messages.label == 1]
    ham = messages[messages.label == 0]

    # # Tf-idf for spam datasets
    # vect_spam = TfidfVectorizer()
    # vect_spam.fit_transform(spam['message'])
    # header_spam = vect_spam.get_feature_names_out()
    #
    # # Tf-idf for ham datasets
    # vect_ham = TfidfVectorizer()
    # vect_ham.fit_transform(ham['message'])
    # header_ham = vect_ham.get_feature_names_out()
    #
    # # find unique ham words
    # ham_unique = list(set(header_ham).difference(set(header_spam)))
    # header_ham1 = pd.DataFrame(columns=ham_unique)
    # header_ham1.to_csv("ham_unique.csv")

    # ham_unique = list(pd.read_csv("ham_unique.csv").columns)

    # with open("x2result.csv", "r") as csvfile:
    #     reader = csv.reader(csvfile)
    #     top100_features = []
    #     for row in reader:
    #         top100_features.append(row[1])
    # top100_features = top100_features[1:]
    # print(top100_features)
    df.to_numpy().transpose()[0].astype(str)
    top100_features = sum_number_pd.to_numpy().transpose()[0].astype(str)
    print("changed top100_features", top100_features)
    # in ham & top100
    ham_unique_in_top = list(
        set(ham_unique).intersection(set(top100_features)))
    words14str = ""
    for item in ham_unique_in_top:
        words14str = words14str + " " + item
    # print("magical word: ", words14str)
    return words14str, spam, ham


def svm_attack_wothreading(clf_lin, spam_message, words14str, feature_names, vectorizer):
    m2_empty_1 = []
    spam_list = spam_message.message.tolist()
    message_original = pd.DataFrame(spam_list, columns=["message"])
    spam_list = list(map(lambda orig_string: orig_string + words14str, spam_list))
    message_14_email = pd.DataFrame(spam_list, columns=["message"])
    message_14_tf_idf = single_transform(message_14_email["message"], feature_names, vectorizer)
    message_14_y = pd.Series([1]*len(message_14_tf_idf))
    message_CData = CDataset(message_14_tf_idf, message_14_y)
    message_14_pred = clf_lin.predict(message_CData.X)
    message_14_tf_idf["pred_label"] = message_14_pred.tolist()
    attack_success = message_14_tf_idf[message_14_tf_idf.pred_label == 0]
    attack_success.reset_index(inplace=True, drop=True)
    spam_cnt_1 = len(attack_success)
    if spam_cnt_1 != 0:
        success_message_original = message_original[message_14_tf_idf.pred_label == 0]
        success_original_tfidf = single_transform(success_message_original["message"], feature_names, vectorizer)
        start_points = success_original_tfidf[feature_names].values
        end_points = attack_success[feature_names].values
        for j in range(len(success_original_tfidf)):
            m2_empty_1 = m2_empty_1 + [cityblock(start_points[j], end_points[j])]
    # for j in spam_message.message:
    #     choose_email = [j + words14str]
    #     message_14_email = pd.DataFrame(choose_email, columns=["message"])
    #     message_14_tf_idf = single_transform(message_14_email["message"], feature_names, vectorizer)
    #     # message_14_tf_idf = pd.DataFrame(message_14_tf_idf.toarray(), columns=feature_names)
    #     message_14_y = [1]
    #     message_14_y = pd.Series(message_14_y)
    #     message_CData = CDataset(message_14_tf_idf, message_14_y)
    #     message_14_pred = clf_lin.predict(message_CData.X)
    #
    #     if message_14_pred == 0:
    #         spam_cnt_1 = spam_cnt_1 + 1
    #         choose_email_original = [j]
    #         message_14_email_original = pd.DataFrame(choose_email_original, columns=["message"])
    #         j_tf_idf = single_transform(message_14_email_original["message"], feature_names, vectorizer)
    #         ma_distance = cityblock(j_tf_idf, message_14_tf_idf.to_numpy())
    #         m2_empty_1 = m2_empty_1 + [ma_distance]

    # print('White box attack with length on SVM:')
    # print('Number of samples provided:', len(spam_message))
    # print('Number of crafted sample that got misclassified:', spam_cnt_1)
    print('Magic Word Attack Successful rate:', spam_cnt_1, "/", len(spam_message), " = ", spam_cnt_1/len(spam_message))
    # print("MA distance:", m2_empty_1)
    return m2_empty_1


def whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test,
             feature_names, vectorizer, nb_attack=100, dmax=0.6, PGDonly=False):

    tr_set, ts_set, clf_lin = train_test_SVM(x_train_features, x_test_features, y_train, y_test)
    lb = np.ndarray.min(x_train_features.to_numpy())
    ub = np.ndarray.max(x_train_features.to_numpy())
    result, cnt, ad_success_x, ori_dataframe, ori_examples2_y, distance = pdg_attack(clf_lin, tr_set, ts_set, y_test,
                                                                           feature_names, nb_attack, dmax, lb, ub,
                                                                           PGDonly)
    if not PGDonly:
        words14str, spam, ham = magical_word(x_train, x_test, y_train, y_test, result, cnt)
        m2_empty = svm_attack_wothreading(clf_lin, spam, words14str, feature_names, vectorizer)
    else:
        m2_empty = distance
    return m2_empty