import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import KeyedVectors

sentences = []
for line in np.load('final_all_sentences.npy'):
    sentences.append(line)
print(len(sentences))


# model_1 = Word2Vec.load('200d_40minwords_10context')
model_1 = KeyedVectors.load_word2vec_format("200d_40minwords_10context", binary=False)
print(len(model_1.wv.vocab))

model_2 = KeyedVectors.load_word2vec_format("model_2", binary=False)
print(len(model_2.wv.vocab))


# # fit a 2d PCA model to the vectors
# X = model_2[model_1.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[10:, 0], result[10:, 1])
# words = list(model_1.wv.vocab)
# words = words[:10]
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()