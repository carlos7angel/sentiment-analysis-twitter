import pandas as pd
from nltk import TweetTokenizer
from nltk.util import ngrams
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessor import Preprocessor


class LexiconExtractor(BaseEstimator, TransformerMixin):

    NGRAM_LENGTH = 3
    REVERSE_WORDS = ['no', 'ni', 'tampoc', 'ningun']

    _tokenizer = TweetTokenizer()
    _preprocessor = Preprocessor(twitter_features='remove', stemming=True)

    def __init__(self):
        data_positive = pd.read_csv('negative_words.txt', header=None, encoding="utf-8")
        data_negative = pd.read_csv('positive_words.txt', header=None, encoding="utf-8")
        self._neg_words = data_negative[0].tolist()
        self._pos_words = data_positive[0].tolist()

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = self._preprocessor.preprocess(tweet)
            result.append(self.count_polarity_words(tweet))

        return preprocessing.normalize(result)

    def count_polarity_words(self, text):
        num_pos_words = 0
        num_neg_words = 0

        list_ngrams = list(ngrams(self._tokenizer.tokenize(text), self.NGRAM_LENGTH, pad_left=True))

        for ngram in list_ngrams:
            pre_words = ngram[:self.NGRAM_LENGTH-1]
            word = ngram[self.NGRAM_LENGTH-1]

            if word in self._pos_words:
                if any(w in pre_words for w in self.REVERSE_WORDS):
                    num_neg_words += 1
                else:
                    num_pos_words += 1

            elif word in self._neg_words:
                if any(w in pre_words for w in self.REVERSE_WORDS):
                    num_pos_words += 1
                else:
                    num_neg_words += 1

        return [num_pos_words, num_neg_words]

    def fit(self, df, y=None):
        return self

