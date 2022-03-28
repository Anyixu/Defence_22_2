from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pandas as pd
from Preprocessing import preprocess
df = pd.read_csv('messages.csv')
y = df.label
x = df.message
x, xt = preprocess(x, x)
tvec1 = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
clf = svm.SVC(kernel='linear')
sfs = SequentialFeatureSelector(clf, n_features_to_select=10, scoring="accuracy", cv=5, n_jobs=-1)

sfs.fit(x, y)
print(sfs.get_feature_names_out())