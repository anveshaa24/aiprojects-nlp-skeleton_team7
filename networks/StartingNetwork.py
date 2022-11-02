import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, dataset_size):
        """

        Arguments:
            dataset_size: The size of the input vector that will be fed to the NN.
        """

        super().__init__()
        self.fc1 = nn.Linear(dataset_size, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x.squeeze(1).float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


