import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import KeyedVectors

#load all sentences
sentences = []
for line in np.load('final_all_sentences.npy'):
    sentences.append(line)
# print(len(sentences))


# load keyed vector of model_1
model_1 = KeyedVectors.load_word2vec_format("200d_40minwords_10context", binary=False)
# print(len(model_1.wv.vocab))


# retrain model_1 on pre-trained model glove.6B.200d.w2vformat.txt 
model_2 = Word2Vec(size=200, min_count=40)
model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model = KeyedVectors.load_word2vec_format("glove.6B.200d.w2vformat.txt", binary=False)
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("glove.6B.200d.w2vformat.txt", binary=False, lockf=1.0)
model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)

# save trained model on pre-trained model as model_2
model_2.wv.save_word2vec_format('model_2', binary=False)

# # fit a 2d PCA model to the vectors
# X = model_2[model_1.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model_1.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()