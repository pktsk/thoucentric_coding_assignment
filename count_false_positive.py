import pandas as pd
import numpy as np

pos_sentiment_dict = dict()
neg_sentiment_dict = dict()

df_test_data = pd.read_csv('testing_data.tsv', sep='\t')

df_test_data_model_1 = pd.read_csv('sentiment_on_model_1.csv')

df_test_data_model_2 = pd.read_csv('sentiment_on_model_2.csv')

false_positive_model_1 = 0
false_positive_model_2 = 0
for i in range(len(df_test_data)):
    if df_test_data.sentiment.iloc[i] == 0 and df_test_data_model_1.sentiment.iloc[i] == 1:
        false_positive_model_1 = false_positive_model_1 + 1
    
    if df_test_data.sentiment.iloc[i] == 0 and df_test_data_model_2.sentiment.iloc[i] == 1:
        false_positive_model_2 = false_positive_model_2 + 1
    

print("model 1", false_positive_model_1)
print("model 2", false_positive_model_2)

