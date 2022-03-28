import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def remove_hyperlink(word):
    return re.sub(r"http\S+", " ", word)


def to_lower(word):
    result = word.lower()
    return result


def remove_number(word):
    result = re.sub(r'\d+', ' ', word)
    return result


def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(word):
    result = word.strip()
    return result


def replace_newline(word):
    return word.replace('\n', ' ')


def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation, remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


def remove_stop_words(words):
    result = [i for i in words if i not in ENGLISH_STOP_WORDS]
    return result


def word_stemmer(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(o) for o in words]


def word_lemmatizer(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(o) for o in words]


def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words, word_lemmatizer]
    for o in cleaning_utils:
        words = o(words)
    return words


def preprocess(x_train, x_test):
    x_train = [clean_up_pipeline(o) for o in x_train]
    x_test = [clean_up_pipeline(o) for o in x_test]
    x_train = [word_tokenize(o) for o in x_train]
    x_test = [word_tokenize(o) for o in x_test]
    x_train = [clean_token_pipeline(o) for o in x_train]
    x_test = [clean_token_pipeline(o) for o in x_test]
    x_train = [" ".join(o) for o in x_train]
    x_test = [" ".join(o) for o in x_test]
    return x_train, x_test