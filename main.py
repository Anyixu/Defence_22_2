from sklearn.feature_extraction.text import TfidfVectorizer
from WAFS import wafs
import pandas as pd
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector

df = pd.read_csv('processed_data.csv')
y = df.label
x_text = df.astype({'message':'str'}).message
x_tr, x_ts, y_tr, y_ts = train_test_split(x_text, y, train_size=2500, test_size=2500, random_state=99)
tvec1 = TfidfVectorizer(max_features=500)
tvec1.fit(x_tr)
print("Feaute length: ", len(tvec1.get_feature_names_out()))
x_tfidf_tr = tvec1.transform(x_tr).toarray()
x_feature_tr = pd.DataFrame(x_tfidf_tr, columns=tvec1.get_feature_names_out())
x_tfidf_ts = tvec1.transform(x_ts).toarray()
x_feature_ts = pd.DataFrame(x_tfidf_ts, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
# s_method=False -> with good word attack
# s_method=True -> with PGD only
selector = SelectFromModel(max_features=500, estimator=svm.SVC(kernel='linear')).fit(x_feature_tr, y_tr)
# clf = svm.SVC(kernel='linear')
# selector = SequentialFeatureSelector(clf, n_features_to_select=500, scoring="accuracy", cv=5, n_jobs=-1)
# selector.fit(x_feature_tr, y_tr)
# print("selected 500 features:", selector.get_feature_names_out())
x_feature_tr = x_feature_tr[selector.get_feature_names_out()]
x_feature_ts = x_feature_ts[selector.get_feature_names_out()]
clf = svm.SVC(kernel='linear')
clf.fit(x_feature_tr, y_tr)
print("Accuracy:", metrics.accuracy_score(y_ts, clf.predict(x_feature_ts)))
# print(selector.get_feature_names_out())
# print(x_feature_tr)
# print(x_feature_ts)
PGD_only = False
print("PGD only:", PGD_only)
thread_num = 1
selected_features = wafs(x_feature_tr, x_feature_ts, y_tr, y_ts, 500, tvec1, x_ts, PGD_only, thread_num)
x_tr = x_tr[selected_features]
print("Finished WAFS")
clf = svm.SVC(kernel='linear')
clf.fit(x_tr, y_tr)
print("Accuracy:", metrics.accuracy_score(y_tr, clf.predict(x_tr)))
print(selected_features)