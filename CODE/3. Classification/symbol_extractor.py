import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessor import Preprocessor


class SentimentSymbolExtractor(BaseEstimator, TransformerMixin):

    _preprocessor = Preprocessor(twitter_features='remove')

    def __init__(self):
        data_pos_symbols = pd.read_csv('positive_symbols.txt', header=None, encoding="utf-8")
        data_neg_symbols = pd.read_csv('negative_symbols.txt', header=None, encoding="utf-8")
        data_neu_symbols = pd.read_csv('neutral_symbols.txt', header=None, encoding="utf-8")
        self._pos_symbols = data_pos_symbols[0].tolist()
        self._neg_symbols = data_neg_symbols[0].tolist()
        self._neu_symbols = data_neu_symbols[0].tolist()

    def transform(self, data, y=None):
        result = []

        for tweet in data:
            tweet = self._preprocessor.preprocess(tweet)
            result.append([sum(tweet.count(symbol) for symbol in self._pos_symbols),
                           sum(tweet.count(symbol) for symbol in self._neg_symbols),
                           sum(tweet.count(symbol) for symbol in self._neu_symbols)])

        return preprocessing.normalize(result)

    def fit(self, df, y=None):
        return self

