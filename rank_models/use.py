# # %%
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# USE_PATH = '/Users/renato/Documents/deep_learning/TensorFlow/USE/5'

# model = hub.load(USE_PATH)
# def encode(input):
#     return model(input)

# def cosine(A, B):
#     return np.dot(A, B.T) / (np.sqrt(np.sum(A * A)) * np.sqrt(np.sum(B * B)))

# def get_similarity(query, documents):
#     q = encode(query)
#     doc = encode(documents)
#     res = []
#     for i, v in enumerate(doc):
#         res.append(tuple([i, cosine(q, v.numpy().reshape(1, -1))[0][0]]))
#     res.sort(key=lambda x: x[1], reverse=True)

#     return res

# %%
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
USE_PATH = '/Users/renato/Documents/deep_learning/TensorFlow/USE/5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# def encode(input):
#     return model(input)

# encode(a)


# with tf.Graph().as_default():
#     model = hub.load(USE_PATH)
#     sentences = tf.placeholder(tf.string)
#     embeddings_use = model(sentences)
#     session = tf.train.MonitoredTrainingSession()

#     r = session.run(embeddings_use, {sentences: a})

# #%%
with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    model = hub.load(USE_PATH)
    use = model(sentences)
    session = tf.train.MonitoredTrainingSession()
    # r = session.run(use, {sentences: a})

def encode(x):
    return session.run(use, {sentences: x})

# predict(a)

# def embed_use(module_use):
#     with tf.Graph().as_default():
#         sentences = tf.placeholder(tf.string)
#         use = hub.load(module_use)
#         embeddings_use = use(sentences)
#         session = tf.train.MonitoredTrainingSession(config=config)
#     return lambda x: session.run(embeddings_use, {sentences: x})

# embed_fn = embed_use(USE_PATH)


# def encode(sentences):
#     return embed_fn(sentences)

# r = encode(a)
def cosine(A, B):
    return np.dot(A, B.T) / (np.sqrt(np.sum(A * A)) * np.sqrt(np.sum(B * B)))

def get_similarity(query, documents):
    q = encode(query)
    doc = encode(documents)
    res = []
    for i, v in enumerate(doc):
        res.append(tuple([i, cosine(q, v.reshape(1, -1))[0][0]]))
    res.sort(key=lambda x: x[1], reverse=True)

    return res
