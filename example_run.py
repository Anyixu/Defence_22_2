from Data_extraction import data_extraction
from Preprocess import preprocess
from Feature_extraction import feature_extraction
from WhiteBox import whitebox
from Feature_selection import feature_selection
from BlackBox import get_spam_ham
from BlackBox import new_blackbox
from BlackBox import original_blackbox
from BlackBox import new_blackbox_all
import numpy as np
import pandas as pd


use_feature_selection = False
# this value can be 'lingspam', 'tutorial', 'enron', 'pu'
whitebox_dataset = 'lingspam'
# this value can be 'TFIDF', 'word2vec', 'doc2vec'
whitebox_method = 'TFIDF'
# this value any integer less than the size of the dataset
attack_amount = 100
try_dmax = 0.6
# this value can be 'lingspam', 'tutorial', 'enron', 'pu'
blackbox_dataset = 'enron'
# this value can be 'TFIDF', 'word2vec', 'doc2vec', 'word2vec_200', 'doc2vec_200'
blackbox_method = 'doc2vec'
print(whitebox_dataset, '-',whitebox_method, '-', attack_amount, '-', try_dmax, '-', blackbox_method)
x_train, x_test, y_train, y_test = data_extraction(whitebox_dataset)

if whitebox_dataset != 'pu':
    x_train, x_test = preprocess(x_train, x_test)

x_train_features, x_test_features, feature_names, feature_model, scalar = feature_extraction(x_train, x_test,
                                                                                             whitebox_method)

if use_feature_selection:
    print("Before feature selection, x_train_features with shape:", x_train_features.shape)
    print("Before feature selection, x_test_features with shape:", x_test_features.shape)
    x_train_features, x_test_features, selection_model = feature_selection(x_train_features, x_test_features,
                                                                           y_train, y_test)
    print("After feature selection, x_train_features with shape:", x_train_features.shape)
    print("After feature selection, x_test_features with shape:", x_test_features.shape)
    features_selected = selection_model.get_support(indices=True)
    temp_name = []
    for i in features_selected:
        temp_name.append(feature_names[i])
    feature_names = temp_name
else:
    selection_model = 'NaN'

words14str, spam, ad_success_x, m2_empty, tr_set, ts_set, ori_dataframe, ori_examples2_y = whitebox(scalar,
                                                                                                    feature_model,
                                                                                                    x_train, x_test,
                                                                                                    x_train_features,
                                                                                                    x_test_features,
                                                                                                    y_train, y_test,
                                                                                                    feature_names,
                                                                                                    attack_amount,
                                                                                                    try_dmax,
                                                                                                    whitebox_method,
                                                                                                    selection_model)
print('The magical word set is:', words14str)
# wordslist = words14str.split(" ")
# word_length = len(wordslist)-1
# print(word_length)
# # original_blackbox(ori_dataframe, ori_examples2_y,
# #                  ad_success_x, tr_set, ts_set, m2_empty)
# print('run blackbox with:', blackbox_dataset, '-', blackbox_method)
# new_blackbox_all(blackbox_dataset,
#                  'ferc counterparti enrononlin calger topica pjm sitara kaminski cera listbot ena clickathom',
#                  'ferc frevert lavo nahou bibi wassup geir kal chilkina cnt lokay noram counterparti eyeforenergi entex highstar sitara',
#                  'ferc eyeforenergi frevert calger listbot tufco chilkina nahou cnt entex counterparti geir waha lavo bibi hplr lokay sitara highstar',
#                  blackbox_method)

