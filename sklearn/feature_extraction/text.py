from __future__ import annotations
import re
import math

class _Matrix:
    def __init__(self, data):
        self._data = data
    def toarray(self):
        return [list(row) for row in self._data]

class TfidfVectorizer:
    def __init__(self, ngram_range=(1,1), stop_words=None):
        self.vocab=[]
    def _tokenize(self, text):
        return re.findall(r"[a-zA-Z0-9]+", (text or '').lower())
    def fit(self, documents):
        vocab=set()
        for doc in documents:
            vocab.update(self._tokenize(doc))
        self.vocab=sorted(vocab)
        return self
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
    def transform(self, documents):
        rows=[]
        for doc in documents:
            toks=self._tokenize(doc)
            counts={t:toks.count(t) for t in set(toks)}
            row=[]
            for term in self.vocab:
                tf=counts.get(term,0)
                idf=1.0
                row.append(tf*idf)
            norm=math.sqrt(sum(v*v for v in row))
            rows.append([v/norm if norm else 0.0 for v in row])
        return _Matrix(rows)
