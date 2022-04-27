# Black-box Attack
from collections import namedtuple
from secml.ml.classifiers import CClassifierKNN, CClassifierDecisionTree, CClassifierSGD, CClassifierSVM, CClassifierSkLearn
from secml.data import CDataset
from secml.data.splitter import CDataSplitterKFold
from secml.ml.peval.metrics import CMetricAccuracy
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from Data_extraction import data_extraction
from Preprocess import preprocess
from Feature_extraction import feature_extraction
from Feature_extraction import single_transform
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from secml.array import CArray
from sklearn.neural_network import MLPClassifier
from Feature_extraction import input_split
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB


def original_blackbox(ori_dataframe, ori_examples2_y, ad_success_x, tr_set, ts_set, m2_empty):
    metric = CMetricAccuracy()
    CLF = namedtuple('CLF', 'clf_name clf xval_parameters')

    random_state = 999

    target_clf_list = [
        CLF(clf_name='kNN',
            clf=CClassifierKNN(),
            xval_parameters={'n_neighbors': [160]}),
        CLF(clf_name='Decision Tree',
            clf=CClassifierDecisionTree(random_state=random_state),
            xval_parameters={'max_depth': [55]}),
        CLF(clf_name='Logistic(SGD)',
            clf=CClassifierSGD(random_state=random_state,
                               regularizer='l2', loss='log'),
            xval_parameters={'alpha': [1e-6, 1e-5, 1e-4, 1e-4]}),
    ]

    for i, test_case in enumerate(target_clf_list):

        clf = test_case.clf
        xval_params = test_case.xval_parameters

        print("\nEstimating the best training parameters of {:} ..."
              "".format(test_case.clf_name))
        xval_splitter = CDataSplitterKFold()
        best_params2 = clf.estimate_parameters(
            dataset=tr_set, parameters=xval_params, splitter=xval_splitter,
            metric='accuracy', perf_evaluator='xval')

        print("The best parameters for '{:}' are: ".format(test_case.clf_name),
              [(k, best_params2[k]) for k in sorted(best_params2)])

        print("Training of {:} ...".format(test_case.clf_name))
        clf.fit(tr_set.X, tr_set.Y)

        # Predictions on test set and performance evaluation
        y_pred1 = clf.predict(ts_set.X)
        acc1 = metric.performance_score(y_true=ts_set.Y, y_pred=y_pred1)

        print("Classifier: {:}\tAccuracy: {:.2%}".format(
            test_case.clf_name, acc1))

    # Add MLP classifier into original blackbox
    clf_mlp = MLPClassifier()
    print("\nEstimating the best training parameters of MLP ...")
    print("Training of MLP ...")
    clf_mlp.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    y_pred = clf_mlp.predict(ts_set.X.tondarray())
    acc_clf_mlp = metric.performance_score(
        y_true=ts_set.Y, y_pred=CArray(y_pred))
    print("Classifier: MLP \tAccuracy: {:.2%}".format(acc_clf_mlp))

    # Add GaussianNB into original blackbox
    clf_gnb = GaussianNB()
    print("\nEstimating the best training parameters of GNB ...")
    print("Training of GNB ...")
    clf_gnb.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    y_pred = clf_gnb.predict(ts_set.X.tondarray())
    acc_clf_gnb = metric.performance_score(
        y_true=ts_set.Y, y_pred=CArray(y_pred))
    print("Classifier: GNB \tAccuracy: {:.2%}".format(acc_clf_gnb))

    # Add Voting classifier into original blackbox
    clf1 = KNeighborsClassifier(n_neighbors=160)
    clf2 = tree.DecisionTreeClassifier(max_depth=55, random_state=999)
    clf3 = SGDClassifier(loss="log", penalty="l2", random_state=999)
    clf1.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    clf2.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    clf3.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    clf_eclf = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), (
        'lr', clf3), ('mlp', clf_mlp), ('gnb', clf_gnb)], voting='hard')
    print("\nEstimating the best training parameters of eclf ...")
    print("Training of eclf ...")
    clf_eclf = clf_eclf.fit(tr_set.X.tondarray(), tr_set.Y.tondarray())
    y_pred = clf_eclf.predict(ts_set.X.tondarray())
    acc_clf_eclf = metric.performance_score(
        y_true=ts_set.Y, y_pred=CArray(y_pred)
    )
    print("Classifier: eclf \tAccuracy: {:.2%} \n".format(acc_clf_eclf))

    # Dataset
    ori_100 = CDataset(ori_dataframe, ori_examples2_y)

    PGD_y = [0] * len(ad_success_x)
    PGD_y = pd.Series(PGD_y)
    PGD_100 = CDataset(ad_success_x, PGD_y)

    m2_y = [0] * len(m2_empty)
    m2_y = pd.Series(m2_y)
    m2_100 = CDataset(m2_empty, m2_y)

    for target_clf in target_clf_list:
        # original emails
        y100 = target_clf.clf.predict(ori_100.X)
        acc100 = metric.performance_score(y_true=ori_100.Y, y_pred=y100)
        print("Classifier: {:}\tAccuracy of 100: {:.2%}".format(
            target_clf.clf_name, acc100))
        # PGD
        yPGD = target_clf.clf.predict(PGD_100.X)
        accPGD = metric.performance_score(y_true=PGD_100.Y, y_pred=yPGD)
        print("Classifier: {:}\tAccuracy of PGD: {:.2%}".format(
            target_clf.clf_name, accPGD))
        # Method 2
        y_m2 = target_clf.clf.predict(m2_100.X)
        acc_m2 = metric.performance_score(y_true=m2_100.Y, y_pred=y_m2)
        print("Classifier: {:}\tAccuracy of Method 2: {:.2%}".format(
            target_clf.clf_name, acc_m2))

    # Test MLP on three datasets
    y100 = clf_mlp.predict(ori_100.X.tondarray())
    acc100 = metric.performance_score(y_true=ori_100.Y, y_pred=CArray(y100))
    print("Classifier: MLP \tAccuracy of 100: {:.2%}".format(acc100))
    # PGD
    yPGD = clf_mlp.predict(PGD_100.X.tondarray())
    accPGD = metric.performance_score(y_true=PGD_100.Y, y_pred=CArray(yPGD))
    print("Classifier: MLP \tAccuracy of PGD: {:.2%}".format(accPGD))
    # Method 2
    y_m2 = clf_mlp.predict(m2_100.X.tondarray())
    acc_m2 = metric.performance_score(y_true=m2_100.Y, y_pred=CArray(y_m2))
    print("Classifier: MLP \tAccuracy of Method 2: {:.2%}".format(acc_m2))

    # Test GNB on three datasets
    y100 = clf_gnb.predict(ori_100.X.tondarray())
    acc100 = metric.performance_score(y_true=ori_100.Y, y_pred=CArray(y100))
    print("Classifier: GNB \tAccuracy of 100: {:.2%}".format(acc100))
    # PGD
    yPGD = clf_gnb.predict(PGD_100.X.tondarray())
    accPGD = metric.performance_score(y_true=PGD_100.Y, y_pred=CArray(yPGD))
    print("Classifier: GNB \tAccuracy of PGD: {:.2%}".format(accPGD))
    # Method 2
    y_m2 = clf_gnb.predict(m2_100.X.tondarray())
    acc_m2 = metric.performance_score(y_true=m2_100.Y, y_pred=CArray(y_m2))
    print("Classifier: GNB \tAccuracy of Method 2: {:.2%}".format(acc_m2))

    # Test eclf on three datasets
    y100 = clf_eclf.predict(ori_100.X.tondarray())
    acc100 = metric.performance_score(y_true=ori_100.Y, y_pred=CArray(y100))
    print("Classifier: eclf \tAccuracy of 100: {:.2%}".format(acc100))
    # PGD
    yPGD = clf_eclf.predict(PGD_100.X.tondarray())
    accPGD = metric.performance_score(y_true=PGD_100.Y, y_pred=CArray(yPGD))
    print("Classifier: eclf \tAccuracy of PGD: {:.2%}".format(accPGD))
    # Method 2
    y_m2 = clf_eclf.predict(m2_100.X.tondarray())
    acc_m2 = metric.performance_score(y_true=m2_100.Y, y_pred=CArray(y_m2))
    print("Classifier: eclf \tAccuracy of Method 2: {:.2%}".format(acc_m2))
# Using only spam in testing set


def get_spam_ham(x_train, y_train, x_test, y_test):
    # d = {'message': x_train, 'label': y_train}
    # df = pd.DataFrame(data=d)
    d1 = {'message': x_test, 'label': y_test}
    df1 = pd.DataFrame(data=d1)
    frames = [df1]
    messages = pd.concat(frames)
    spam = messages[messages.label == 1]
    ham = messages[messages.label == 0]
    return spam, ham


def clf_train(x_train_features, x_test_features, y_train, y_test):
    clf_list = []
    acc_list = []
    cm_list = []
    defense_clf_list = []
    defense_acc_list = []

    tr = CDataset(x_train_features, y_train)
    ts = CDataset(x_test_features, y_test)
    metric = CMetricAccuracy()
    random_state = 999
    xval_splitter = CDataSplitterKFold()

    parameters = {'n_neighbors': [100, 130, 160, 190, 210, 240],
                   'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, parameters, refit=True, verbose=1)
    grid_knn.fit(tr.X.tondarray(), tr.Y.tondarray())
    print(grid_knn.best_params_)
    y_pred = grid_knn.predict(ts.X.tondarray())
    print(classification_report(ts.Y.tondarray(), y_pred))
    acc_knn = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(grid_knn)
    defense_acc_list.append(acc_knn)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    parameters = {'max_depth': [20, 55, 110, 165, 220, 275, None],
                  'criterion': ['gini', 'entropy'],
                  'class_weight': ['balanced', None]}
    dt = DecisionTreeClassifier()
    grid_dt = GridSearchCV(dt, parameters, refit=True, verbose=1)
    grid_dt.fit(tr.X.tondarray(), tr.Y.tondarray())
    print(grid_dt.best_params_)
    y_pred = grid_dt.predict(ts.X.tondarray())
    print(classification_report(ts.Y.tondarray(), y_pred))
    acc_dt = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(grid_dt)
    defense_acc_list.append(acc_dt)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    parameters = {'alpha': [1e-6, 1e-5, 1e-4],
                  'max_iter': [10000]}
    lr = SGDClassifier(random_state=999)
    grid_lr = GridSearchCV(lr, parameters, refit=True, verbose=1)
    grid_lr.fit(tr.X.tondarray(), tr.Y.tondarray())
    print(grid_lr.best_params_)
    y_pred = grid_lr.predict(ts.X.tondarray())
    print(classification_report(ts.Y.tondarray(), y_pred))
    acc_lr = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(grid_lr)
    defense_acc_list.append(acc_lr)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    parameters = {'hidden_layer_sizes': [(1000, 500, 100), (800, 400, 200), (400, 200, 100), (200, 150, 50)],
                  'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-4, 1e-3]}
    mpl = MLPClassifier(random_state=999)
    grid_mpl = GridSearchCV(mpl, parameters, refit=True, verbose=1)
    grid_mpl.fit(tr.X.tondarray(), tr.Y.tondarray())
    print(grid_mpl.best_params_)
    y_pred = grid_mpl.predict(ts.X.tondarray())
    print(classification_report(ts.Y.tondarray(), y_pred))
    acc_mlp = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(grid_mpl)
    defense_acc_list.append(acc_mlp)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    parameters = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-7, 1e-6, 1e-5]}
    gnb = GaussianNB()
    grid_gnb = GridSearchCV(gnb, parameters, refit=True, verbose=1)
    grid_gnb.fit(tr.X.tondarray(), tr.Y.tondarray())
    print(grid_gnb.best_params_)
    y_pred = grid_gnb.predict(ts.X.tondarray())
    print(classification_report(ts.Y.tondarray(), y_pred))
    acc_gnb = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(grid_gnb)
    defense_acc_list.append(acc_gnb)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    eclf = VotingClassifier(estimators=[('knn', grid_knn), ('dt', grid_dt), ('lr', grid_lr),
                                        ('mpl', grid_mpl), ('gnb', grid_gnb)],
                            voting='hard')
    eclf = eclf.fit(tr.X.tondarray(), tr.Y.tondarray())
    y_pred = eclf.predict(ts.X.tondarray())
    acc_eclf = metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred))
    defense_clf_list.append(eclf)
    defense_acc_list.append(acc_eclf)
    cm_list.append(confusion_matrix(ts.Y.tondarray(), y_pred))

    return defense_clf_list, defense_acc_list, cm_list


def add_magical(method, sample, words14str, feature_model, feature_names, scaler):
    message_list = []
    for j in sample.message:
        choose_email = [j + words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        message_14_tf_idf = single_transform(
            message_14_email["message"], method, feature_model, feature_names, scaler)
        message_14_tf_idf = pd.DataFrame(
            message_14_tf_idf.toarray(), columns=feature_names)
        message_list.append(message_14_tf_idf)
    return message_list


def clf_attack(method, clf_lin, sample, words14str, feature_model, feature_names, scaler, defense, message_list):
    spam_cnt = 0
    m2_empty = pd.DataFrame()

    for message_14_tf_idf in message_list:
        # choose_email = [j + words14str]
        # message_14_email = pd.DataFrame(choose_email, columns=["message"])
        # message_14_tf_idf = single_transform(
        #     message_14_email["message"], method, feature_model, feature_names, scaler)
        # message_14_tf_idf = pd.DataFrame(
        #     message_14_tf_idf.toarray(), columns=feature_names)
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData = CDataset(message_14_tf_idf, message_14_y)
        if defense == 'true':
            message_14_pred = clf_lin.predict(message_CData.X.tondarray())
        else:
            message_14_pred = clf_lin.predict(message_CData.X)

        if message_14_pred == 0:
            spam_cnt = spam_cnt + 1
            m2_empty = m2_empty.append(message_14_tf_idf, ignore_index=True)

    print('Number of samples provided:', len(sample))
    print('Number of crafted sample that got misclassified:', spam_cnt)
    print('Successful rate:', spam_cnt/len(sample))
    return m2_empty


def svm_200_word(x_train, x_test, y_train, y_test, words14str):
    metric = CMetricAccuracy()
    cm_list_200 = []
    temp_x_train = input_split(x_train)
    temp_x_test = input_split(x_test)
    model_train_word = Word2Vec(temp_x_train, vector_size=200)
    x_train_features_word200 = []
    x_test_features_word200 = []

    for email in temp_x_train:
        email_vec_word = []
        for token in email:
            token_vec_word = [0] * 200
            if model_train_word.wv.has_index_for(token):
                token_vec_word = model_train_word.wv.get_vector(token)
                token_vec_word = token_vec_word.tolist()
            email_vec_word.append(token_vec_word)
        temp_email_vec_word = np.mean(email_vec_word, axis=0).tolist()
        x_train_features_word200.append(temp_email_vec_word)

    for email in temp_x_test:
        email_vec_word = []
        for token in email:
            if model_train_word.wv.has_index_for(token):
                token_vec_word = model_train_word.wv.get_vector(token)
                email_vec_word.append(token_vec_word)
        email_vec_word = np.mean(email_vec_word, axis=0).tolist()
        x_test_features_word200.append(email_vec_word)
    for i in range(len(x_test_features_word200) - 1):
        if type(x_test_features_word200[i]) == float:
            x_test_features_word200 = x_test_features_word200[:i] + x_test_features_word200[(i+1):]
            y_test = np.delete(y_test, i)
            x_test = np.delete(x_test, i)
    defense_clf_list, defense_acc_list, cm_list = clf_train(x_train_features_word200,
                                                            x_test_features_word200,
                                                            y_train, y_test)
    tr_word = CDataset(x_train_features_word200, y_train)
    ts_word = CDataset(x_test_features_word200, y_test)

    svm_word = CClassifierSVM()
    svm_word.fit(tr_word.X, tr_word.Y)
    y_pred_word = svm_word.predict(ts_word.X)
    acc_word = metric.performance_score(y_true=ts_word.Y, y_pred=y_pred_word)
    cm_list_200.append(confusion_matrix(
        ts_word.Y.tondarray(), y_pred_word.tondarray()))

    spam, ham = get_spam_ham(x_train, y_train, x_test, y_test)
    spam_cnt_word = 0
    spam_cnt_other = [0] * 6
    for j in spam.message:
        choose_email = [j + words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        temp_x = message_14_email["message"].values
        temp_x = temp_x[0].split(' ')
        email_vec_word = []
        for token in temp_x:
            if model_train_word.wv.has_index_for(token):
                token_vec_word = model_train_word.wv.get_vector(token)
                email_vec_word.append(token_vec_word)
        email_vec_word = np.mean(email_vec_word, axis=0).tolist()
        email_vec_word = [email_vec_word]
        message_14_word = sparse.csr_matrix(email_vec_word)
        message_14_word = pd.DataFrame(message_14_word.toarray())
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData_word = CDataset(message_14_word, message_14_y)
        message_14_pred_word = svm_word.predict(message_CData_word.X)
        message_14_pred_other = []
        for clf in defense_clf_list:
            pred_list = clf.predict(message_CData_word.X.tondarray())
            message_14_pred_other.append(pred_list[0])

        if message_14_pred_word == 0:
            spam_cnt_word = spam_cnt_word + 1
        for index in range(0, len(message_14_pred_other)):
            if message_14_pred_other[index] == 0:
                spam_cnt_other[index] = spam_cnt_other[index] + 1

    print('SVM word2vec 200:')
    print("Accuracy on test set: {:.2%}".format(acc_word))
    print("CM:")
    print(cm_list_200[0])
    print('Number of samples provided:', len(spam))
    print('Number of crafted sample that got misclassified:', spam_cnt_word)
    print('Successful rate:', spam_cnt_word / len(spam))

    names_defense = ['KNN:', 'DT:', 'SGD:', 'MPL:', 'GNB:', 'ECLF:']
    count = 0
    for i in range(0, len(names_defense)):
        print(names_defense[i])
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[i]))
        print("CM:")
        print(cm_list[count])
        print('Number of samples provided:', len(spam))
        print('Number of crafted sample that got misclassified:',
              spam_cnt_other[count])
        print('Successful rate:', spam_cnt_other[count] / len(spam))
        count += 1


def svm_200_doc(x_train, x_test, y_train, y_test, words14str):
    metric = CMetricAccuracy()
    cm_list_200 = []
    temp_x_train = input_split(x_train)
    temp_x_test = input_split(x_test)
    documents = [TaggedDocument(doc, [i])
                 for i, doc in enumerate(temp_x_train)]
    model_train_doc = Doc2Vec(documents, vector_size=200)
    x_train_features_doc200 = []
    x_test_features_doc200 = []
    for email in temp_x_train:
        email_vec_doc = model_train_doc.infer_vector(email).tolist()
        x_train_features_doc200.append(email_vec_doc)

    for email in temp_x_test:
        email_vec_doc = model_train_doc.infer_vector(email)
        x_test_features_doc200.append(email_vec_doc)

    for i in range(len(x_test_features_doc200) - 1):
        if type(x_test_features_doc200[i]) == float:
            x_test_features_doc200 = x_test_features_doc200[:i] + x_test_features_doc200[(i+1):]
            y_test = np.delete(y_test, i)
            x_test = np.delete(x_test, i)
    defense_clf_list, defense_acc_list, cm_list = clf_train(x_train_features_doc200,
                                                            x_test_features_doc200,
                                                            y_train, y_test)

    tr_doc = CDataset(x_train_features_doc200, y_train)
    ts_doc = CDataset(x_test_features_doc200, y_test)

    svm_doc = CClassifierSVM()
    svm_doc.fit(tr_doc.X, tr_doc.Y)
    y_pred_doc = svm_doc.predict(ts_doc.X)
    acc_doc = metric.performance_score(y_true=ts_doc.Y, y_pred=y_pred_doc)
    cm_list_200.append(confusion_matrix(
        ts_doc.Y.tondarray(), y_pred_doc.tondarray()))

    spam, ham = get_spam_ham(x_train, y_train, x_test, y_test)
    spam_cnt_doc = 0
    spam_cnt_other = [0] * 6
    for j in spam.message:
        choose_email = [j + words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        temp_x = message_14_email["message"].values
        temp_x = temp_x[0].split(' ')
        email_vec_doc = model_train_doc.infer_vector(temp_x)
        email_vec_doc = [email_vec_doc]
        message_14_doc = sparse.csr_matrix(email_vec_doc)
        message_14_doc = pd.DataFrame(message_14_doc.toarray())
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData_doc = CDataset(message_14_doc, message_14_y)
        message_14_pred_doc = svm_doc.predict(message_CData_doc.X)
        message_14_pred_other = []
        for clf in defense_clf_list:
            pred_list = clf.predict(message_CData_doc.X.tondarray())
            message_14_pred_other.append(pred_list[0])

        if message_14_pred_doc == 0:
            spam_cnt_doc = spam_cnt_doc + 1
        for index in range(0, len(message_14_pred_other)):
            if message_14_pred_other[index] == 0:
                spam_cnt_other[index] = spam_cnt_other[index] + 1

    print('SVM doc2vec 200:')
    print("Accuracy on test set: {:.2%}".format(acc_doc))
    print("CM:")
    print(cm_list_200[0])
    print('Number of samples provided:', len(spam))
    print('Number of crafted sample that got misclassified:', spam_cnt_doc)
    print('Successful rate:', spam_cnt_doc / len(spam))

    names_defense = ['KNN:', 'DT:', 'SGD:', 'MPL:', 'GNB:', 'ECLF:']
    count = 0

    for i in range(0, len(names_defense)):
        print(names_defense[i])
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[i]))
        print("CM:")
        print(cm_list[count])
        print('Number of samples provided:', len(spam))
        print('Number of crafted sample that got misclassified:',
              spam_cnt_other[count])
        print('Successful rate:', spam_cnt_other[count] / len(spam))
        count += 1


def new_blackbox(dataset, words14str, blackbox_method):
    x_train, x_test, y_train, y_test = data_extraction(dataset)
    x_train, x_test = preprocess(x_train, x_test)
    if blackbox_method == 'word2vec_200':
        svm_200_word(x_train, x_test, y_train, y_test, words14str)
    if blackbox_method == 'doc2vec_200':
        svm_200_doc(x_train, x_test, y_train, y_test, words14str)

    if blackbox_method == 'TFIDF' or \
            blackbox_method == 'word2vec' or\
            blackbox_method == 'doc2vec':

        x_train_features, x_test_features, feature_names, feature_model, scaler = feature_extraction(x_train, x_test,
                                                                                                     blackbox_method)

        defense_clf_list, defense_acc_list, cm_list = clf_train(x_train_features, x_test_features, y_train, y_test)
        spam, ham = get_spam_ham(x_train, y_train, x_test, y_test)
        message_list = add_magical(blackbox_method, spam, words14str, feature_model,
                                   feature_names, scaler)
        print('KNN:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[0]))
        print("CM:")
        print(cm_list[0])
        clf_attack(blackbox_method, defense_clf_list[0], spam, words14str,
                   feature_model, feature_names, scaler, 'true', message_list)

        print('DT:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[1]))
        print("CM:")
        print(cm_list[1])
        clf_attack(blackbox_method, defense_clf_list[1], spam, words14str,
                   feature_model, feature_names, scaler, 'true', message_list)

        print('SGD:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[2]))
        print("CM:")
        print(cm_list[2])
        clf_attack(blackbox_method, defense_clf_list[2], spam,
                   words14str, feature_model, feature_names, scaler, 'true', message_list)

        print('MPL:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[3]))
        print("CM:")
        print(cm_list[3])
        clf_attack(blackbox_method, defense_clf_list[3], spam,
                   words14str, feature_model, feature_names, scaler, 'true', message_list)

        print('GNB:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[4]))
        print("CM:")
        print(cm_list[4])
        clf_attack(blackbox_method, defense_clf_list[4], spam,
                   words14str, feature_model, feature_names, scaler, 'true', message_list)

        print('ECLF:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[5]))
        print("CM:")
        print(cm_list[5])
        clf_attack(blackbox_method, defense_clf_list[5], spam,
                   words14str, feature_model, feature_names, scaler, 'true', message_list)


def new_blackbox_all(dataset, words14str_TFIDF, words14str_w2v, words14str_d2v, blackbox_method):
    x_train, x_test, y_train, y_test = data_extraction(dataset)
    x_train, x_test = preprocess(x_train, x_test)
    if blackbox_method == 'word2vec_200':
        svm_200_word(x_train, x_test, y_train, y_test, words14str_TFIDF)
        svm_200_word(x_train, x_test, y_train, y_test, words14str_w2v)
        svm_200_word(x_train, x_test, y_train, y_test, words14str_d2v)
    if blackbox_method == 'doc2vec_200':
        svm_200_doc(x_train, x_test, y_train, y_test, words14str_TFIDF)
        svm_200_doc(x_train, x_test, y_train, y_test, words14str_w2v)
        svm_200_doc(x_train, x_test, y_train, y_test, words14str_d2v)

    if blackbox_method == 'TFIDF' or \
            blackbox_method == 'word2vec' or\
            blackbox_method == 'doc2vec':

        x_train_features, x_test_features, feature_names, feature_model, scaler = feature_extraction(x_train, x_test,
                                                                                                     blackbox_method)

        print(x_train_features.get_shape())
        defense_clf_list, defense_acc_list, cm_list = clf_train(x_train_features, x_test_features,
                                                                y_train, y_test)
        spam, ham = get_spam_ham(x_train, y_train, x_test, y_test)
        message_list_words14str_TFIDF = add_magical(blackbox_method, spam, words14str_TFIDF, feature_model,
                                                    feature_names, scaler)
        message_list_words14str_w2v = add_magical(blackbox_method, spam, words14str_w2v, feature_model,
                                                    feature_names, scaler)
        message_list_words14str_d2v = add_magical(blackbox_method, spam, words14str_d2v, feature_model,
                                                    feature_names, scaler)
        print('KNN:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[0]))
        print("CM:")
        print(cm_list[0])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[0], spam, words14str_TFIDF,
                   feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[0], spam, words14str_w2v,
                   feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[0], spam, words14str_d2v,
                   feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)

        print('DT:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[1]))
        print("CM:")
        print(cm_list[1])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[1], spam,
                   words14str_TFIDF,feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[1], spam,
                   words14str_w2v, feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[1], spam,
                   words14str_d2v, feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)

        print('SGD:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[2]))
        print("CM:")
        print(cm_list[2])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[2], spam,
                   words14str_TFIDF, feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[2], spam,
                   words14str_w2v, feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[2], spam,
                   words14str_d2v, feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)

        print('MPL:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[3]))
        print("CM:")
        print(cm_list[3])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[3], spam,
                   words14str_TFIDF, feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[3], spam,
                   words14str_w2v, feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[3], spam,
                   words14str_d2v, feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)

        print('GNB:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[4]))
        print("CM:")
        print(cm_list[4])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[4], spam,
                   words14str_TFIDF, feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[4], spam,
                   words14str_w2v, feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[4], spam,
                   words14str_d2v, feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)

        print('ECLF:')
        print("Accuracy on test set: {:.2%}".format(defense_acc_list[5]))
        print("CM:")
        print(cm_list[5])
        print('With magical words from TFIDF:')
        clf_attack(blackbox_method, defense_clf_list[5], spam,
                   words14str_TFIDF, feature_model, feature_names, scaler, 'true', message_list_words14str_TFIDF)
        print('With magical words from w2v:')
        clf_attack(blackbox_method, defense_clf_list[5], spam,
                   words14str_w2v, feature_model, feature_names, scaler, 'true', message_list_words14str_w2v)
        print('With magical words from d2v:')
        clf_attack(blackbox_method, defense_clf_list[5], spam,
                   words14str_d2v, feature_model, feature_names, scaler, 'true', message_list_words14str_d2v)
