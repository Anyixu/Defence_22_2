from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
from statistics import mean
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess
from WAFS import estimate_s, wafs

# Before WAFS
df = pd.read_csv('messages.csv')
y = df.label
x_text = df.message
x, xt = preprocess(x_text, x_text)
print("Finished preprocessing")
tvec1 = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
model = svm.SVC(kernel='linear')
sfs = SequentialFeatureSelector(model, n_features_to_select=39)
sfs.fit(x, y)
x = sfs.transform(x)
x = pd.DataFrame(data=x, columns=sfs.get_feature_names_out())
print("Without WAFS accuracy: ", mean(cross_val_score(model, x, y, cv=5)))
print("Without WAFS security term: ", estimate_s(x, y, tvec1, x_text))

# with WAFS selected features
df = pd.read_csv('messages.csv')
y = df.label
x_text = df.message
x, xt = preprocess(x_text, x_text)
print("Finished preprocessing")
tvec1 = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
model = svm.SVC(kernel='linear')
x = x[['acl', 'du', 'und', 'paris', 'symposium', 'benjamin', 'teacher', 'money', 'logic', 'vowel', 'et', 'au', 'mit',
      'free', 'chapter', 'click', 'movement', 'remove', 'credit', 'business', 'syntax', 'case', 'offer', 'linguistic',
      'easy', 'search', 'future', 'linguist', 'word', 'reply', 'http', 'net', 'want', 'site', 'today', 'check', 'home',
      'good', 'sent']]
print("With WAFS accuracy: ", mean(cross_val_score(model, x, y, cv=5)))
print("With WAFS security term: ", estimate_s(x, y, tvec1, x_text))

# with WAFS_PGD_only selected features
df = pd.read_csv('messages.csv')
y = df.label
x_text = df.message
x, xt = preprocess(x_text, x_text)
print("Finished preprocessing")
tvec1 = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
model = svm.SVC(kernel='linear')
PGD_only = True
selected_features = wafs(x, y, 39, tvec1, x_text, PGD_only)
x = x[selected_features]
print("With WAFS_PGD_only Only accuracy: ", mean(cross_val_score(model, x, y, cv=5)))
print("With WAFS_PGD_only security term: ", estimate_s(x, y, tvec1, x_text))

# with WAFS_PGD_only selected features
df = pd.read_csv('messages.csv')
y = df.label
x_text = df.message
x, xt = preprocess(x_text, x_text)
print("Finished preprocessing")
tvec1 = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
model = svm.SVC(kernel='linear')
PGD_only = True
selected_features = wafs(x, y, 60, tvec1, x_text, PGD_only)
x = x[selected_features]
print("With WAFS_PGD_only 60 Only accuracy: ", mean(cross_val_score(model, x, y, cv=5)))
print("With WAFS_PGD_only 60 security term: ", estimate_s(x, y, tvec1, x_text))
