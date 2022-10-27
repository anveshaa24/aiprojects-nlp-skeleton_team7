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
        """
        print(self.df)

        OUTPUT:
                                  qid                                      question_text  target
        0        00002165364db923c7e6  How did Quebec nationalists see their province...       0
        1        000032939017120e6e44  Do you have an adopted dog, how would you enco...       0
        2        0000412ca6e4628ce2cf  Why does velocity affect time? Does velocity a...       0
        3        000042bf85aa498cd78e  How did Otto von Guericke used the Magdeburg h...       0
        4        0000455dfa3e01eae3af  Can I convert montra helicon D to a mountain b...       0
        ...                       ...                                                ...     ...
        1306117  ffffcc4e2331aaf1e41e  What other technical skills do you need as a c...       0
        1306118  ffffd431801e5a2f4861  Does MS in ECE have good job prospects in USA ...       0
        1306119  ffffd48fb36b63db010c                          Is foam insulation toxic?       0
        1306120  ffffec519fa37cf60c78  How can one start a research project based on ...       0
        1306121  ffffed09fedb5088744a  Who wins in a battle between a Wolverine and a...       0
        """
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        """
        >>> print(self.sequences)

        >>> (0, 22)       1
            (1, 68)       1
            (2, 25)       2
            (2, 91)       1
            (3, 22)       1
            (3, 96)       1
            (4, 48)       1
            (6, 25)       2
            (6, 72)       1
            (8, 88)       1
            (8, 24)       1
            (9, 22)       1
            (9, 68)       1
            (9, 48)       1
            (9, 102)      1
            (9, 32)       1
            (9, 89)       1
            (9, 12)       1
            (10, 76)      1
            (12, 92)      1
            (12, 93)      1
            (13, 95)      1
            (14, 57)      1
            (15, 49)      1
            (16, 37)      1
            :     :
            (1306103, 93) 1
            (1306103, 45) 1
            (1306103, 75) 1
            (1306105, 106)        2
            (1306105, 23) 1
            (1306106, 83) 1
            (1306107, 25) 1
            (1306108, 29) 1
            (1306109, 49) 2
            (1306109, 63) 1
            (1306111, 108)        1
            (1306111, 106)        1
            (1306111, 0)  1
            (1306112, 46) 1
            (1306116, 37) 1
            (1306116, 84) 1
            (1306117, 63) 1
            (1306117, 18) 1
            (1306117, 78) 1
            (1306118, 25) 1
            (1306118, 37) 1
            (1306118, 45) 1
            (1306118, 53) 1
            (1306118, 47) 1
            (1306120, 81) 1
        """


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
        return self.sequences.shape[0]