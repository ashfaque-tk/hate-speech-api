

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re, string, spacy
from nltk.corpus import stopwords

class PreprocessText(BaseEstimator, TransformerMixin):

    def __init__(self, remove_stop=True, lemma=True, stem=False):
        self.remove_stop = remove_stop
        self.lemma = lemma
        self.stem = stem

        self.stopwords = set(stopwords.words('english')) - {"not", "no", "never", "nor"}
        self.expand_negations = {
            "doesn't": "does not", "can't": "cannot", "don't": "do not",
            "haven't": "have not", "shouldn't": "should not",
            "wouldn't": "would not", "didn't": "did not", "needn't": "need not"
        }

        self.noise = re.compile(r'https\S+|www\S+|@\w+|<.*?>')
        self.neg_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self.expand_negations.keys()) + r')\b'
        )
        self.punct_table = str.maketrans("", "", string.punctuation)

        self.nlp = None  # lazy loaded

    def fit(self, X, y=None):
        # Load spaCy only if needed
        if self.lemma and self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return self

    def transform(self, X):

        # Input adapter
        if isinstance(X, str):
            X = pd.Series([X])
        elif isinstance(X, list):
            X = pd.Series(X)
        elif not isinstance(X, pd.Series):
            print(type(X))
            raise TypeError("Input must be str, list[str], or pd.Series")

        X = X.astype(str).str.lower().str.replace("’", "'")
        X = X.str.replace(self.noise, "", regex=True)

        def replace_match(m):
            return self.expand_negations[m.group(0)]

        X = X.str.replace(self.neg_pattern, replace_match, regex=True)
        X = X.str.translate(self.punct_table)

        texts = X.tolist()

        if not self.lemma:
            tokens = [t.split() for t in texts]
        else:
            tokens = [
                [tok.lemma_.lower() for tok in doc if tok.is_alpha]
                for doc in self.nlp.pipe(texts, batch_size=500)
            ]

        if self.remove_stop:
            tokens = [
                [t for t in tok_list if t not in self.stopwords]
                for tok_list in tokens
            ]

        return [" ".join(tok_list) for tok_list in tokens]
