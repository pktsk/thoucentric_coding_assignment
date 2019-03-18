All training, testing and saved models are contained in all_files folder ( link: https://drive.google.com/drive/folders/1OAozmc9KMUN5lJ0HqMVH5EY3KLI86pfJ?usp=sharing )


1. Run ` python3 data_split.py ` to split the labeledTrainData.tsv (link: https://drive.google.com/file/d/1npz-CMweQAY2awt7jsAg55XvZldFSOZ1/view?usp=sharing) dataset to into train ( 20000 data points ) and test (5000 data points) dataset .
   - Generated dataset: training_data.tsv (link: https://drive.google.com/file/d/1Yqplh3IUAI9XeYHyUqFxDyeqeSobbcB1/view?usp=sharing )
   - Generated dataset: testing_data.tsv  (link: https://drive.google.com/file/d/1LqIsdldaRG0MNU_Mvs_Hf8cSpwacEDKH/view?usp=sharing )
  
 2. Run ` python3 save_all_sentences.py ` to parse all sentence from training_data.tsv (link: https://drive.google.com/file/d/1Yqplh3IUAI9XeYHyUqFxDyeqeSobbcB1/view?usp=sharing ) and unlabeledTrainData.tsv (link: https://drive.google.com/file/d/1-rn5zlugZLIyjPbUDazl2FzucOu_4XJi/view?usp=sharing)
  - Above command generates an array of array of words and saves it to a file named final_all_sentences.npy (link: https://drive.google.com/file/d/1zNZnYaHzX2uNOuktL3ajrITECycJV7ce/view?usp=sharing )
  
 3. Run ` python3 train_on_all_sentences_old_word2Vec.py ` to train Traditional or old word2Vec model on all sentences parsed by above command. 
  - Above command saves the trained model ( model_1 ) as 200d_40minwords_10context (link: https://drive.google.com/file/d/1lhtZSUYBHKMHcyV3fvVnvKzCfVCIrw_P/view?usp=sharing) which word embeddings. 
  
 4. Run ` python3 train_on_pre_trained_model.py ` to train word2Vec on pre-trained model glove.6B.200d.w2vformat.txt (link: https://drive.google.com/file/d/14C0O-5Fadysnnf7BLsTkvePLLCvn7Bdg/view?usp=sharing)
  - Above command saves the newly trained model as model_2 (link: https://drive.google.com/file/d/1NZxbSUz-ZdwcFBt72PEmxGUmshpY9nR1/view?usp=sharing)
  
 5. Run ` python3 create_clean_reviews.py ` to create a clean reviews to training and testing purpose. 
   - Also save those files as clean_reviews_for_train_data (link: https://drive.google.com/file/d/1GrmO5smEUtjm6ns6CdVd_6BDrALjYPcN/view?usp=sharing) and clean_reviews_for_test_data (link: https://drive.google.com/file/d/1GrmO5smEUtjm6ns6CdVd_6BDrALjYPcN/view?usp=sharing)
   
 6. Run ` python3 training_n_testing_on_model_1_using_random_forest.py ` to train random forest model using clean_reviews_for_train_data dataset and model_1
   - Accuracy achieved: 81.4 %
 7. Run ` python3 training_n_testing_on_model_2_using_random_forest.py ` to train random forest model using clean_reviews_for_train_data dataset and model_2
   - Accuracy achieved: 81.8 %

 
