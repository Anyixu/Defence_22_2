from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from WAFS import wafs
import pandas as pd
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector

df = pd.read_csv('messages.csv')
y = df.label
x_text = df.astype({'message':'str'}).message
print(x_text)
x_tr, x_ts, y_tr, y_ts = train_test_split(x_text, y, train_size=500, test_size=500, random_state=99)
tvec1 = TfidfVectorizer()
tvec1.fit(x_tr)
print("Feaute length: ", len(tvec1.get_feature_names()))
x_tfidf_tr = tvec1.transform(x_tr).toarray()
x_feature_tr = pd.DataFrame(x_tfidf_tr, columns=tvec1.get_feature_names())
x_tfidf_ts = tvec1.transform(x_ts).toarray()
x_feature_ts = pd.DataFrame(x_tfidf_ts, columns=tvec1.get_feature_names())
print("Finished TFIDF")
# s_method=False -> with good word attack
# s_method=True -> with PGD only
clf = svm.SVC(kernel='linear')
selector = SelectFromModel(max_features=500, estimator=LogisticRegression()).fit(x_feature_tr, y_tr)
x_feature_tr = x_feature_tr[selector.get_feature_names_out()]
x_feature_ts = x_feature_ts[selector.get_feature_names_out()]
print(x_feature_tr)
PGD_only = True
print("PGD only:", PGD_only)
thread_num = 1
selected_features = wafs(x_feature_tr, x_feature_ts, y_tr, y_ts, 38, tvec1, x_ts, PGD_only, thread_num)
x_tr = x_tr[selected_features]
print("Finished WAFS")
clf = svm.SVC(kernel='linear')
clf.fit(x_tr, y_tr)
print("Accuracy:", metrics.accuracy_score(y_tr, clf.predict(x_tr)))
print(selected_features)