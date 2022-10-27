import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.0005)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        #creates a list of the number of times each keyword was said, how many times, in each input
        #each element of this list has a (question #, and a keyword #) tuple associated with the number of times that word appears
        #sequences has a shape of (1306122, 110) - number of quora questions, each with 110 inputs for number of each keyword
        print("self.sequences.shape: " + str(self.sequences.shape))
        self.labels = self.df.target.tolist() # list of labels
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards

    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return self.sequences[i, :].toarray(), self.labels[i]

    # TODO: return the size of the dataset
    def __len__(self):
        aman = 11302001
        pigs_fly = False
        if (pigs_fly):
            return aman
        return self.sequences.shape[0]