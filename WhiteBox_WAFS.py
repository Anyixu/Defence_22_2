from secml.data import CDataset
from secml.data.splitter import CDataSplitterKFold
from secml.ml.classifiers import CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.peval.metrics import CMetricConfusionMatrix
from secml.adv.attacks.evasion import CAttackEvasionPGD
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import threading
import time
from scipy.spatial.distance import cityblock
from sklearn import metrics


def single_transform(message, feature_names, vectorizer):
    d = vectorizer.transform(message).toarray()
    result = pd.DataFrame(data=d, columns=vectorizer.get_feature_names_out())
    result = result[feature_names]
    # result = pd.DataFrame(vectorizer.transform(message), columns=vectorizer.get_feature_names_out())[feature_names]
    return result


def train_test_SVM(x_train_features, x_test_features, y_train, y_test):
    tr_set = CDataset(x_train_features, y_train)
    # Train the SVM
    xval_splitter = CDataSplitterKFold()
    clf_lin = CClassifierSVM()
    xval_lin_params = {'C': [1]}
    best_lin_params = clf_lin.estimate_parameters(
        dataset=tr_set,
        parameters=xval_lin_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )
    clf_lin.fit(tr_set.X, tr_set.Y)

    # Test the Classifier
    ts_set = CDataset(x_test_features, y_test)
    y_pred = clf_lin.predict(ts_set.X)
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=ts_set.Y, y_pred=y_pred)
    return tr_set, ts_set, clf_lin


def pdg_attack(clf_lin, tr_set, ts_set, y_test, feature_names, nb_attack, dmax, lb, ub, just_pgd):
    class_to_attack = 1
    cnt = 0  # the number of success adversaril examples
    ori_examples2_x = []
    ori_examples2_y = []
    idx_candidates = np.where(y_test == class_to_attack)
    # print('total example: ', len(idx_candidates[0]))
    for i in idx_candidates[0]:
        # take a point at random being the starting point of the attack
        # # select nb_init_pts points randomly in candidates and make them move
        # rn = np.random.choice(idx_candidates[0].size, 1)
        x0, y0 = ts_set[i, :].X, ts_set[i, :].Y
        x0 = x0.astype(float)
        y0 = y0.astype(int)
        x2 = x0.tondarray()[0]
        y2 = y0.tondarray()[0]

        ori_examples2_x.append(x2)
        ori_examples2_y.append(y2)
    # Perform adversarial attacks
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    y_target = 0
    # dmax = 0.09  # Maximum perturbation

    # Bounds of the attack space. Can be set to `None` for unbounded

    solver_params = {
        'eta': 0.3,
        'max_iter': 100,
        'eps': 1e-4}

    # set lower bound and upper bound respectively to 0 and 1 since all features are Boolean
    pgd_attack = CAttackEvasionPGD(
        classifier=clf_lin,
        double_init_ds=tr_set,
        distance=noise_type,
        dmax=dmax,
        lb=lb, ub=None,
        solver_params=solver_params,
        y_target=y_target
    )

    ad_examples_x = []
    ad_examples_y = []
    ad_index = []
    cnt = 0
    for i in range(len(ori_examples2_x)):
        # print("Current Number:", i)
        x0 = ori_examples2_x[i]
        y0 = ori_examples2_y[i]

        y_pred_pgd, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)
        # print("Original x0 label: ", y0.item())
        # print("Adversarial example label (PGD): ", y_pred_pgd.item())
        #
        # print("Number of classifier gradient evaluations: {:}"
        #       "".format(pgd_attack.grad_eval))

        if y_pred_pgd.item() == 0:
            cnt = cnt + 1
            ad_index.append(i)

        ad_examples_x.append(adv_ds_pgd.X.tondarray()[0])
        ad_examples_y.append(y_pred_pgd.item())
        # print("original feature:", ori_examples2_x[i])
        # print("attack feature:", adv_ds_pgd.X.tondarray()[0])
        attack_pt = adv_ds_pgd.X.tondarray()[0]
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
    print("result", result)
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
    sum_number_pd.to_csv("x2result.csv")
    d = {'message': x_train, 'label': y_train}
    df = pd.DataFrame(data=d)
    d1 = {'message': x_test, 'label': y_test}
    df1 = pd.DataFrame(data=d1)
    frames = [df, df1]
    messages = pd.concat(frames)
    spam = messages[messages.label == 1]
    ham = messages[messages.label == 0]

    # Tf-idf for spam datasets
    vect_spam = TfidfVectorizer()
    print('xxxxxxxxxxxxxxxxxxxxx', spam['message'])
    vect_spam.fit_transform(spam['message'])
    header_spam = vect_spam.get_feature_names_out()

    # Tf-idf for ham datasets
    vect_ham = TfidfVectorizer()
    vect_ham.fit_transform(ham['message'])
    header_ham = vect_ham.get_feature_names_out()

    # find unique ham words
    ham_unique = list(set(header_ham).difference(set(header_spam)))
    header_ham1 = pd.DataFrame(ham_unique)
    header_ham1.to_csv("ham_unique.csv")

    with open("x2result.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        top100_features = []
        for row in reader:
            top100_features.append(row[1])
    top100_features = top100_features[1:]
    # in ham & top100

    ham_unique_in_top = list(
        set(ham_unique).intersection(set(top100_features)))
    words14str = ""
    for item in ham_unique_in_top:
        words14str = words14str + " " + item
    print("magical word: ", words14str)
    return words14str, spam, ham


def svm_attack_wothreading(clf_lin, spam_message, words14str, feature_names, vectorizer):
    m2_empty_1 = []
    spam_cnt_1 = 0
    for j in spam_message.message:
        choose_email = [j + words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        message_14_tf_idf = single_transform(message_14_email["message"], feature_names, vectorizer)
        # message_14_tf_idf = pd.DataFrame(message_14_tf_idf.toarray(), columns=feature_names)
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData = CDataset(message_14_tf_idf, message_14_y)
        message_14_pred = clf_lin.predict(message_CData.X)

        if message_14_pred == 0:
            spam_cnt_1 = spam_cnt_1 + 1
            choose_email_original = [j]
            message_14_email_original = pd.DataFrame(choose_email_original, columns=["message"])
            j_tf_idf = single_transform(message_14_email_original["message"], feature_names, vectorizer)
            ma_distance = cityblock(j_tf_idf, message_14_tf_idf.to_numpy())
            m2_empty_1 = m2_empty_1 + [ma_distance]

    print('White box attack with length on SVM:')
    print('Number of samples provided:', len(spam_message))
    print('Number of crafted sample that got misclassified:', spam_cnt_1)
    print('Successful rate:', spam_cnt_1 / len(spam_message))
    print("MA distance:", m2_empty_1)
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