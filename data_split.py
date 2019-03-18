import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('labeledTrainData.tsv', delimiter='\t', encoding='utf-8')
# first 20000 data points for training and last 5000 data points for testing
df3 = df.head(20000)
df4 = df.tail(5000)

df3.to_csv('training_data.tsv', sep='\t', index = False)
df4.to_csv('testing_data.tsv', sep='\t', index = False)