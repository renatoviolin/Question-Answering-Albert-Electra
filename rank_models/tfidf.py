import json
import os
import traceback
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def take_verb(text):
    return " ".join([tpl[0] for tpl in pos_tag(text.split()) if tpl[1] in ['NN', 'VB']])

def word_token(tokens, lemma=False):
    tokens = str(tokens)
    tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
    tokens = re.sub(r"\s+", " ", tokens)
    if lemma:
        return " ".join([lemmatizer.lemmatize(token, 'v') for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])
    else:
        return " ".join([token for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])

def get_similarity(query, documents):
    vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=True, lowercase=True)
    docs = query + documents
    # print(docs)
    # docs = [take_verb(d) for d in docs]
    docs = [word_token(d, lemma=True) for d in docs]
    # print(docs[0])

    # query = word_token(query, True)
    # docs = word_token(docs, True)
    # print(docs)
    doc_vector = vectorizer.fit_transform(docs)
    cosine_similarities = cosine_similarity(doc_vector, doc_vector)[0]
    # print(cosine_similarities[1:])
    r = []
    for i, v in enumerate(cosine_similarities[1:]):
        r.append(tuple([i, v]))
    r.sort(key=lambda x: x[1], reverse=True)
    return r
