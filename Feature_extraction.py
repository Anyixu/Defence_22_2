from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import TaggedDocument


vectorizer = TfidfVectorizer()


def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    print(raw_sentences[0])
    print(raw_sentences[1])
    return vectorizer.transform(raw_sentences)


def TfidfConvert(x_train, x_test):
    x_train = [o.split(" ") for o in x_train]
    x_test = [o.split(" ") for o in x_test]
    raw_sentences = [' '.join(o) for o in x_train]
    vectorizer.fit(raw_sentences)
    x_train_features = convert_to_feature(x_train)
    x_test_features = convert_to_feature(x_test)
    return x_train_features, x_test_features


def getUniqueWords(allWords):
    uniqueWords = []
    for i in allWords:
        if i not in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords


def input_split(x):
    new_x = []
    for line in x:
        newline = line.split(' ')
        new_x.append(newline)
    return new_x


def getUniqueWords(allWords):
    uniqueWords = []
    for i in allWords:
        if i not in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords


def x2vec(input_x, feature_names, model):
    x_features = []
    for index in input_x:
        model_vector = [0] * len(feature_names)

        for token in index:
            if token in feature_names:
                feature_index = feature_names.index(token)

                if model.wv.has_index_for(token):
                    token_vecs = model.wv.get_vector(token)
                    model_vector[feature_index] = token_vecs[0]
        x_features.append(model_vector)
    return x_features


def single_transform(x, method, feature_model, feature_names, scaler, selection_model):
    if method == 'TFIDF':

        result = feature_model.transform(x)
        if selection_model != 'NaN':
            result = selection_model.transform(result)
        return result
    else:
        temp_x = x.values
        temp_x = temp_x[0].split(' ')
        model_vector = [0] * len(feature_names)
        for token in temp_x:
            if token in feature_names:
                feature_index = feature_names.index(token)
                if feature_model.wv.has_index_for(token):
                    token_vecs = feature_model.wv.get_vector(token)
                    model_vector[feature_index] = token_vecs[0]
        x_features = [model_vector]
        # x_features = np.array(x_features)
        x_features = scaler.transform(x_features)
        x_train_features = sparse.csr_matrix(x_features)
        if selection_model != 'NaN':
            x_train_features = selection_model.transform(x_train_features)
        return x_train_features


# def feature_extraction_200dimension(x_train, x_test, method):
    # a


def feature_extraction(x_train, x_test, method):

    if method == 'TFIDF':
        x_train_features, x_test_features = TfidfConvert(x_train, x_test)
        feature_names = vectorizer.get_feature_names()

        return x_train_features, x_test_features, feature_names, vectorizer, 'NaN'

    if method == 'word2vec':
        temp_x_train = input_split(x_train)
        temp_x_test = input_split(x_test)
        model_train = Word2Vec(temp_x_train, vector_size=1)
        feature_space = []
        for index in temp_x_train:
            feature_space = feature_space + getUniqueWords(index)
        feature_names = getUniqueWords(feature_space)
        x_train_features = x2vec(temp_x_train, feature_names, model_train)
        x_test_features = x2vec(temp_x_test, feature_names, model_train)
        x_train_features = np.array(x_train_features)
        x_test_features = np.array(x_test_features)
        pd.DataFrame(x_train_features).to_csv("x_train_features.csv", header=None, index=False)
        pd.DataFrame(x_test_features).to_csv("x_test_features.csv", header=None, index=False)
        scaler = MinMaxScaler()
        scaler.fit(x_train_features)
        x_train_features = scaler.transform(x_train_features)
        x_test_features = scaler.transform(x_test_features)
        x_train_features = sparse.csr_matrix(x_train_features)
        x_test_features = sparse.csr_matrix(x_test_features)
        return x_train_features, x_test_features, feature_names, model_train, scaler

    if method == 'doc2vec':
        temp_x_train = input_split(x_train)
        temp_x_test = input_split(x_test)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(temp_x_test)]
        model_train = Doc2Vec(documents, vector_size=1)
        feature_space = []
        for index in temp_x_train:
            feature_space = feature_space + getUniqueWords(index)
        feature_names = getUniqueWords(feature_space)
        x_train_features = x2vec(temp_x_train, feature_names, model_train)
        x_test_features = x2vec(temp_x_test, feature_names, model_train)
        scaler = MinMaxScaler()
        scaler.fit(x_train_features)
        x_train_features_scaled = scaler.transform(x_train_features)
        x_test_features_scaled = scaler.transform(x_test_features)
        x_train_features = sparse.csr_matrix(x_train_features_scaled)
        x_test_features = sparse.csr_matrix(x_test_features_scaled)
        return x_train_features, x_test_features, feature_names, model_train, scaler


