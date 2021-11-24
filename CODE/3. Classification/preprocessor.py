import re
from nltk import TweetTokenizer
from nltk.stem import SnowballStemmer


class Preprocessor:

    MENTION = 'twmention'
    HASHTAG = 'twhashtag'
    URL = 'twurl'
    LAUGH = 'twlaugh'

    DIACRITICAL_VOWELS = [('á','a'), ('é','e'), ('í','i'), ('ó','o'), ('ú','u'), ('ü','u')]
    SLANG = [('d','de'), ('[qk]','que'), ('xo','pero'), ('xa', 'para'), ('[xp]q','porque'),('es[qk]', 'es que'),
              ('fvr','favor'),('(xfa|xf|pf|plis|pls|porfa)', 'por favor'), ('dnd','donde'), ('tb', 'también'),
              ('(tq|tk)', 'te quiero'), ('(tqm|tkm)', 'te quiero mucho'), ('x','por'), ('\+','mas')]

    _stemmer = SnowballStemmer('spanish')
    _tokenizer = TweetTokenizer().tokenize

    def __init__(self, twitter_features=None, stemming=False):
        self._twitter_features = twitter_features
        self._stemming = stemming

    def preprocess(self, message):
        # convertir a minúscula
        message = message.lower()
        # remover némeros, retornos de carro, saltos de línea y 'RT'
        message = re.sub(r'(\d+|\n|\brt\b)', '', message)
        # remover tíldes
        for s,t in self.DIACRITICAL_VOWELS:
            message = re.sub(r'{0}'.format(s), t, message)
        # remover caracteres repetidos
        message = re.sub(r'(.)\1{2,}', r'\1\1', message)
        # normalizar risas o carcajadas
        message = self.normalize_laughs(message)
        # traducir jerga
        for s,t in self.SLANG:
            message = re.sub(r'\b{0}\b'.format(s), t, message)

        # aplicacamos la limpieza o normalizacion de hashtags, menciones o enlaces web
        message = self.process_twitter_features(message, self._twitter_features)

        # aplicamos el stemming de ser necesario al mensaje tokenizado
        if self._stemming:
            message = ' '.join(self._stemmer.stem(w) for w in self._tokenizer(message))

        return message

    @staticmethod
    def process_twitter_features(message, twitter_features):
        message = re.sub(r'[\.\,]http','. http', message, flags=re.IGNORECASE)
        message = re.sub(r'[\.\,]#', '. #', message)
        message = re.sub(r'[\.\,]@', '. @', message)

        if twitter_features == 'remove':
            # remueve menciones, hashtags and urls
            message = re.sub(r'((?<=\s)|(?<=\A))(@|#)\S+', '', message)
            message = re.sub(r'\b(https?:\S+)\b', '', message, flags=re.IGNORECASE)
        elif twitter_features == 'normalize':
            # normaliza menciones, hashtags and urls
            message = re.sub(r'((?<=\s)|(?<=\A))@\S+', Preprocessor.MENTION, message)
            message = re.sub(r'((?<=\s)|(?<=\A))#\S+', Preprocessor.HASHTAG, message)
            message = re.sub(r'\b(https?:\S+)\b', Preprocessor.URL, message, flags=re.IGNORECASE)

        return message

    @staticmethod
    def normalize_laughs(message):
        message = re.sub(r'\b(?=\w*[j])[aeiouj]{4,}\b', Preprocessor.LAUGH, message, flags=re.IGNORECASE)
        message = re.sub(r'\b(juas+|lol)\b', Preprocessor.LAUGH, message, flags=re.IGNORECASE)
        return message

    def __str__(self):
        return "Preprocessor([twitter_features={0}, stemming={1}])".format(self._twitter_features, self._stemming)

    def __repr__(self):
        return "Preprocessor([twitter_features={0}, stemming={1}])".format(self._twitter_features, self._stemming)