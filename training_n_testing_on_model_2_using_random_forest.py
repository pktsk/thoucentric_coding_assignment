import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import KeyedVectors
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import re
from bs4 import BeautifulSoup
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # Remove HTML
        review_text = BeautifulSoup(review).get_text()
        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        # Convert words to lower case and split them
        words = review_text.lower().split()
        # Optionally rove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # Return a list of words
        return (words)
        

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences each sentence is a list of words,
        # so this returns a list of lists
        return sentences

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # Loop through the reviews
    for review in reviews:
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


if __name__ == '__main__':
    # load saved model 2
    model = KeyedVectors.load_word2vec_format("model_2", binary=False)
    print("model loaded")

    # Read data from files
    train = pd.read_csv('training_data.tsv', header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv('testing_data.tsv', header=0, delimiter="\t", quoting=3 )
    print("train and test data loaded")

    clean_reviews_for_train_data = []
    for line in np.load('clean_reviews_for_train_data.npy'):
        clean_reviews_for_train_data.append(line)

    print("clean reviews train len: ", len(clean_reviews_for_train_data))



    clean_reviews_for_test_data = []
    for line in np.load('clean_reviews_for_test_data.npy'):
        clean_reviews_for_test_data.append(line)
    print("clean reviews test Len:", len(clean_reviews_for_test_data))

    print ("Creating average feature vecs for training reviews")
    trainDataVecs = getAvgFeatureVecs(clean_reviews_for_train_data, model, 200)
    print("trained data vec: len:  ", len(trainDataVecs))
    print("trained data vec: len: each  ", len(trainDataVecs[0]))

    print ("Creating average feature vecs for test reviews")
    testDataVecs = getAvgFeatureVecs( clean_reviews_for_test_data, model, 200 )

    # print("test_data len ", len(testDataVecs))
    # print("test_data len each : ", len(testDataVecs[0]))
 
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier( n_estimators = 100 )

    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, train["sentiment"] )

    # Test & extract results
    result = forest.predict(testDataVecs)
    # check accuracy
    print(accuracy_score(test["sentiment"], result))

    # Write the test results
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "sentiment_on_model_2.csv", index=False, quoting=3 )


