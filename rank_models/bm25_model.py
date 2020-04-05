from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
from rank_bm25 import BM25L
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def word_token(tokens, lemma=False):
    tokens = str(tokens)
    tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
    tokens = re.sub(r"\s+", " ", tokens)
    if lemma:
        return " ".join([lemmatizer.lemmatize(token, 'v') for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])
    else:
        return " ".join([token for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])


def get_similarity(query, documents):
    docs = query + documents
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    # print(tokenized_corpus)
    # print(tokenized_corpus)
    bm25 = BM25Okapi(tokenized_corpus[1:])
    bm25plus = BM25Plus(tokenized_corpus[1:])
    bm25L = BM25L(tokenized_corpus[1:])

    query = tokenized_corpus[0]
    # print(query)
    bm25_scores = bm25.get_scores(query)
    bm25plus_scores = bm25plus.get_scores(query)
    bm25L_scores = bm25L.get_scores(query)
    
    bm25_scores = [(i, v) for i, v in enumerate(bm25_scores)]
    bm25plus_scores = [(i, v) for i, v in enumerate(bm25plus_scores)]
    bm25L_scores = [(i, v) for i, v in enumerate(bm25L_scores)]
    
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    bm25plus_scores.sort(key=lambda x: x[1], reverse=True)
    bm25L_scores.sort(key=lambda x: x[1], reverse=True)
    
    # print(bm25_scores)
    # print(bm25plus_scores)
    # print(bm25L_scores)
    # print(bm25_scores)
    # print(bm25plus_scores)
    # print(bm25L_scores)
    
    return bm25_scores, bm25plus_scores, bm25L_scores
