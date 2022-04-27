from sklearn.feature_extraction.text import TfidfVectorizer
from WAFS import estimate_s
import pandas as pd
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import statistics
from numba import jit
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from statistics import mean
from WhiteBox_WAFS import whitebox
import threading
from threading import Thread
import time


def wafs_final_test(x_feature_tr, x_feature_ts, y_tr, y_ts, vectorizer, text, s_method, lamda=0.5):
    initial_features = x_feature_tr.columns.tolist()
    best_features = []
    g = []
    s = []
    gs = []
    lamdas = []
    srates = []
    for i in range(len(initial_features)):
        new_column = initial_features[i]
        model = svm.SVC(kernel='linear')
        new_g = mean(cross_val_score(model, x_feature_tr[best_features + [new_column]], y_tr, cv=5))
        new_s, srate = estimate_s(x_feature_ts[best_features + [new_column]], y_ts, vectorizer, text,
                                           s_method=s_method)
        new_gs = new_g + lamda * new_s
        g.append(new_g)
        s.append(new_s)
        gs.append(new_gs)
        lamdas.append(lamda)
        srates.append(srate)
        lamda = lamda * (new_s ** -1)
        best_features.append(new_column)
        print("current features: ", len(best_features))
    df = pd.DataFrame(data={'g':g, 's':s, 'gs':gs, 'lamda':lamdas, 'success_rate':srates})
    return df


df = pd.read_csv('processed_data.csv')
y = df.label
x_text = df.astype({'message': 'str'}).message
x_tr, x_ts, y_tr, y_ts = train_test_split(x_text, y, train_size=2500, test_size=2500, random_state=99)
tvec1 = TfidfVectorizer(max_features=200000)
tvec1.fit(x_text)
x_tfidf_tr = tvec1.transform(x_tr).toarray()
x_feature_tr = pd.DataFrame(x_tfidf_tr, columns=tvec1.get_feature_names_out())
x_tfidf_ts = tvec1.transform(x_ts).toarray()
x_feature_ts = pd.DataFrame(x_tfidf_ts, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
PGD_only = False

PGD_features = ['org', '2007', 'stat', 'wrote', 'list', 'samba', 'perl', 'mailing', 'code', 'branches', 'pgp', 'lists', 'cnn', 'ch', 'uwo', 'file', 'posting', 'test', 'alerts', 'satcon', 'package', 'modified', 'pm', 'article', 'unsubscribe', 'source', 'signature', 'ctdb', 'clouds', 'sunny', 'string', 'braille', 'guide', 'listinfo', 'cbs', 'struct', 'cc', 'example', 'lib', 'data', 'subscribed', 'pl', '16', 'jerry', 'debian', '3d', 'network', 'netflix', 'stories', 'forecast', 'story', 'pod', 'tdb', 'utc', '23', '12', 'mail', 'error', 'function', 'ethz', 'forwarded', '13', 'howstuffworks', 'help', 'beginners', 'open', 'parrot', 'cbsnews', 'rev', 'tridge', 'hash', 'bbc', 'june', 'rights', 'think', 'library', '_______________________________________________', 'sugar', 'run', 'oneok', 'pmc', 'use', 'news', 'thailand', 'your', 'transfer', 'bush', 'topcoder', 'patent', 'vtable', 'false', 'docs', 'week', 'thanks', 'config', 'license', '0200', 'museum', 'essential', 'return', 'configure', 'compiled', 'dept', 'install', 'self', 'get_string', 'wind', 'module', 'http', 'fog', 'language', 'destdir', 'interp', 'matrix', 'script', 'txt', 'setup', 'done', 'utilities', 'output', 'whether', 'david', 'smb', 'station', 'ktwarwic', 'theft', 'andy', 'null', 'int', 'broker', 'wednesday', 'cell', 'newsbrief', 'rolex', 'sendlen', 'options', 'adj', 'offset', 'pcwindows', 'increase', 'girl', 'her', 'following', 'question', 'lowest', 'shareholder', 'jeremy', 'talloc', 'netlogon', 'base64', 'safe', 'ipr', 'create', 'disposition', 'd0', 'inputs', 'csmo', 'producttestpanel', 'penis', 'degree', 'summit', 'pio', 'dhamma', 'aids', 'company', 'thursday', 'latest', 'ap', 'up', 'if', 'inc', 'second', 'someone', 'tidy', 'multipart', 'include', 'case', 'already', 'hard', 'line', 'alt', 'stay', '0d', 'ggplot2', 'math', 'libreplace', 'show', 'inline', 'elite', 'image', 'tonight', 'petdance', 'get_lanman2_dir_entry', 'statements', 'titannews', 'viewing', '15', 'default', 'ip', 'enum', 'digital', 'us', 'were', 'imaging', 'swisscham', 'ign', 'great', 'in', 'carbon', 'pmcs', 'non', 'gtk', 'content', 'two', '11', 'encoding', '56', 'offer', '54', 'product', 'work', 'added', 'ad0be', 'utf', 'userinput', 'preferences', 'manager', 'subplot', 'mudletta', 'dosage', '57', 'sheldon', 'td', 'rodale', 'lodrick', 'piano', 'smp', 'radiation', 'intel', 'legend', 'hi', 'using', 'acls', 'version', 'which', 'he', 'minimal', 'relations', 've', 'linux', 'columns', 'she', 'see', 'nbsp', 'goldfish', 'cid', '18098', 'watches', 'documentation', 'plot', 'learn', 'cabot', 'dataframes', 'glib', 'quintum', 'fecund', 'info', 'have', 'price', '04', 'samba_3_0_26', 'patrol', 'winding', 'kate', 'qualitystocks', 'rate', 'sysvol', 'partition', 'smb_maxcnt', 'dq0uyvjaq8htmikjz3yh', 'log', 'trunk', 'nan', 'release', 'his', 'mac', 'boot', 'src', 'anyone', 'cbt', 'lmer', 'aumix', 'continuations', 'padding', 'avcooper', 'revision', 'cm', 'gov', 'arm', 'be', 'too', 'update', 'laptop', 'experts', 'reply', 'link', 'tf_mnss_message_category', 'irs', 'powermarketers', 'qmail', 'commented', 'ctx_mem', 'compare', 'srea', 'only', 'heimdal', 'we', 'changed', 'web', 'products', 'had', 'lm', 'req', 'online', 'change', 'million', 'should', 'samba4', 'profile', 'vector', 'llc', 'smiles', 'sqlplus', 'ekiga', 'donny', 'hk', 'pugs', 'an', 'office', 'problem', 'type', 'length', '0000', 'lottery', 'beverage', 'sizeof', 'boundary', '14', 'table', 'abbott', 'their', 'also', 'minor', 'most', 'him', 'suretymailings', 'borse', 'argument', 'testdb', 'delivery', 'for', 'accuweather', 'trying', 'davison', 'dataset', 'kasger', 'twinkle', 'dc', 'suggestions', 'card', 'meds', 'allison', 'mlcr0s0ft', 'mcse', 'maximum', 'investorplace', 'dollars', 'results2', 'www', 'sequestration', 'gdata', 'minutes', 'casino', 'com', 'bulgaria', 'rlapack', 'page', 'speech', 'homes', 'îò', 'home', 'you', 'no', 'mean', 'long', 'georgia', 'countries', 'about', 'become', 'ipaths', 'recipients', 'mhln', 'project', 'sorry', 'bicycling', 'printable', 'read', 'tuesday', 'tcp', 'both', 'over', 'patch', 'printf', 'investment', '06', 'service', 'capacity', 'dell', 'yourself', '99', 'mr', 'received', 'sub', '40speedy', 'imaging3', 'replace', 'givenchy', 'corporation', 'drugs', 'travlocks', 'ermx', 'aspx', 'weeks', 'libdir', 'jeqwkqomi78l5dcsqgylvyxkmjkqdiu', 'power', 'covers', 'expression', 'bit', 'r2html', 'quoted', 'import', 'pills', 'salon', 'correct', 'uranium', 'mnss_facility_ind_id', 'larry', 'body', 'men', 'follow', 'viagra', 'pdf', 'britney', 'speed', '07752694', 'ufsj', 'reform', 'srcdir', 'rustypromdress', 'experience', 'uncensored', 'do', 'eppg', 'windows', 'h323', 'mnss_message_t', 'karen', 'bwj', 'money', 'computer', 'many', 'buy', 'state', 'net', 'mailman', 'amazing', 'the', 'gif', 'de', '22', 'to', 'me', 'but', 'alert', '07', 'tid', 'tidy_args', 'napster', 'affordable', '52', 'pygame', 'quality', 'our', 'out', 'hand', 'would', 'speakup', 'on', 'that']
x_feature_tr_t = x_feature_tr[PGD_features]
x_feature_ts_t = x_feature_ts[PGD_features]
sfs_df = wafs_final_test(x_feature_tr_t, x_feature_ts_t, y_tr, y_ts, tvec1, x_ts, PGD_only)
sfs_df.to_csv("PGDT_result.csv")

MagicWord_features = ['destdir', 'gtk', 'mnss_message_t', 'mnss_facility_ind_id', 'posting', 'rev', 'org', 'speakup', '2007', 'wrote', 'cnn', 'code', 'data', 'source', 'sunny', 'perl', 'forecast', 'oneok', 'netflix', 'file', 'test', 'lists', 'thailand', 'linux', 'struct', 'lib', 'pmc', '0200', 'think', 'example', 'mailing', 'open', 'mean', 'goldfish', '3d', 'matrix', 'abbott', 'windows', 'pm', 'fog', 'topcoder', 'table', 'module', 'tf_mnss_message_category', 'smb', 'cbsnews', 'delivery', 'tcp', 'museum', 'cbs', 'bush', 'stories', 'news', 'partition', 'get_lanman2_dir_entry', 'aumix', 'subplot', 'winding', 'gdata', 'r2html', 'dataframes', 'ign', 'piano', 'smb_maxcnt', 'libreplace', 'sub', 'vtable', 'parrot', 'affordable', 'already', 'default', '23', 'error', 'replace', '12', 'napster', 'suggestions', 'include', 'patent', 'countries', 'script', 'docs', 'increase', 'aids', 'ggplot2', 'summit', 'covers', 'amazing', 'sheldon', 'ctx_mem', 'salon', 'src', 'sizeof', 'article', 'carbon', 'for', 'tonight', 'kasger', 'continuations', '18098', 'satcon', 'dollars', 'alerts', 'buy', 'price', 'most', 'relations', 'www', 'correct', 'log', 'create', 'talloc', 'intel', 'minutes', 'sqlplus', '07752694', 'utf', 'arm', 'suretymailings', 'adj', 'britney', 'sysvol', '40speedy', 'become', 'srcdir', 'llc', 'null', 'men', 'wind', 'gif', 'jeremy', 'givenchy', 'elite', 'thursday', 'smiles', 'plot', 'speed', 'bulgaria', 'userinput', 'enum', 'inputs', 'interp', 'cabot', 'statements', 'experts', 'titannews', 'theft', 'newsbrief', 'cell', 'investorplace', 'config', 'page', 'srea', 'card', 'both', 'vector', 'speech', 'wednesday', 'boot', 'legend', 'îò', 'dell', 'recipients', 'casino', 'trying', 'import', 'lowest', 'netlogon', 'padding', 'mcse', 'million', 'base64', 'setup', 'pcwindows', 'glib', 'computer', 'license', 'but', 'his', 'bbc', 'preferences', 'argument', 'andy', 'petdance', 'latest', 'ip', 'week', 'de', 'too', 'mail', 'forwarded', 'content', 'line', '15', 'inline', 'sugar', 'meds', 'following', 'avcooper', 'lmer', 'ad0be', 'rolex', 'only', 'boundary', 'watches', 'june', '0d', 'options', 'see', 'release', 'should', 'jerry', 'tridge', 'product', 'hash', 'function', 'version', '13', 'length', 'change', 'kate', 'transfer', 'library', 'td', 'results2', 'non', 'dc', 'radiation', 'manager', 'sorry', '54', 'rlapack', 'quintum', 'story', 'show', 'documentation', 'powermarketers', 'type', 'columns', 'eppg', 'imaging', 'donny', 'http', 'debian', 'install', 'changed', 'do', 'your', 'samba', 'update', 'cc', 'received', '56', 'digital', 'mhln', 'follow', 'clouds', 'company', 'patch', 'output', 'using', 'pmcs', 'pod', 'minimal', 'dept', 'david', 'configure', 'problem', 'products', '11', 'acls', 'corporation', 'over', 'minor', 'body', 'uwo', 'encoding', 'qualitystocks', 'added', 'int', 'long', 'pio', 'twinkle', 'hk', 'money', 'utc', 'lodrick', 'safe', 'weeks', 'drugs', 'sequestration', 'aspx', 'maximum', 'uranium', 'sendlen', 'pygame', 'ipr', 'beverage', 'rodale', 'tidy', 'reform', 'uncensored', 'two', 'larry', 'be', '14', 'office', 'return', 'you', 'our', 'case', 'list', 'rustypromdress', 'alert', 'samba_3_0_26', 'accuweather', 'laptop', 'investment', 've', 'printable', 'language', 'heimdal', 'producttestpanel', '06', 'ekiga', 'pdf', 'tid', 'ufsj', 'station', 'dq0uyvjaq8htmikjz3yh', 'alt', 'stay', 'profile', 'she', 'tdb', 'allison', 'bit', 'also', 'many', 'pills', 'utilities', 'cm', 'h323', 'homes', 'compiled', 'second', 'ap', 'rights', 'qmail', 'inc', 'us', 'ermx', 'girl', '52', 'dosage', 'tidy_args', 'multipart', 'reply', 'ktwarwic', 'fecund', 'jeqwkqomi78l5dcsqgylvyxkmjkqdiu', 'dhamma', 'imaging3', 'bwj', 'lottery', 'online', 'experience', 'tuesday', 'mlcr0s0ft', 'howstuffworks', 'irs', 'borse', 'lm', 'disposition', 'that', 'nan', 'to', 'capacity', 'run', 'dataset', 'viewing', 'project', 'braille', 'work', 'trunk', 'nbsp', 'samba4', 'someone', 'package', 'shareholder', 'rate', 'service', 'txt', 'printf', '16', 'essential', 'mudletta', 'quality', 'patrol', 'smp', '99', 'branches', 'pl', 'image', 'hi', 'swisscham', 'georgia', 'had', 'listinfo', 'their', 'question', 'beginners', 'get_string', 'gov', 'pugs', 'link', '57', 'we', 'offer', 'expression', 'her', 'use', 'penis', 'com', 'hard', 'great', 'were', 'net', 'home', '_______________________________________________', 'false', 'web', 'testdb', 'done', 'have', 'up', 'csmo', 'ipaths', 'cid', 'karen', 'travlocks', 'd0', 'modified', 'thanks', 'anyone', 'state', 'mac', 'quoted','viagra', 'revision', 'offset', 'bicycling', 'mr', 'libdir', 'compare', 'cbt', 'power', 'whether', '07', 'learn', 'req', 'davison', 'yourself', '0000', 'he', 'unsubscribe', 'about', 'math', 'hand', 'network', 'which', 'out', 'read', 'in', 'no', 'broker', 'me', 'degree', 'him', 'mailman', 'info', 'subscribed', '22', 'if', 'the', 'an', 'commented', 'ethz', 'string', 'self', 'guide', 'ch', 'would', 'stat', '04', 'help', 'ctdb', 'on', 'pgp', 'signature']

x_feature_tr_f = x_feature_tr[MagicWord_features]
x_feature_ts_f = x_feature_ts[MagicWord_features]
sfs_df = wafs_final_test(x_feature_tr_f, x_feature_ts_f, y_tr, y_ts, tvec1, x_ts, PGD_only)
sfs_df.to_csv("PGDF_result.csv")

# SFS
clf = svm.SVC(kernel='linear')
sfs = SequentialFeatureSelector(clf, n_features_to_select=500, scoring='accuracy')
sfs.fit(x_feature_tr, y_tr)
x_feature_tr = x_feature_tr[sfs.get_feature_names_out()]
x_feature_ts = x_feature_ts[sfs.get_feature_names_out()]
sfs_df = wafs_final_test(x_feature_tr, x_feature_ts, y_tr, y_ts, tvec1, x_ts, PGD_only)
print("SFS selected features: ", sfs.get_feature_names_out())
sfs_df.to_csv("sfs_result.csv")