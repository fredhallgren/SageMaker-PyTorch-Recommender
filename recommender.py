
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler


class Recommender(nn.Module):

    """
    Standard recommender model through matrix factorization

    :param n_items: Number of items

    :param n_factors: Number of hidden factors to explain the user preferences

    """
    def __init__(self, n_items: int, n_factors: int = 3):
        
        super().__init__()
        
        # The hidden factors in terms of the items
        self.factors = nn.Embedding(n_items, n_factors) # eigenvectors


    def init_params(self):
        """Initialize weights to small random numbers
        """
        nn.init.normal_(self.factors.weight, std=0.1)

    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for one user

        :param x: Data for one user, 2d row tensor 

        :return: Model output

        """
        # Calculate the weightings of a user on the different factors
        user_weightings: Tensor = x @ self.factors.weight.clone()

        # Normalize as though a user supplied all the ratings
        user_weightings *= len(x) / (x != 0).sum().item()

        # Reconstruct user data for all items
        predictions: Tensor = user_weightings @ self.factors.weight.clone().T

        return predictions


    def get_dim(self) -> int:
        """Getter for the hidden dimension (number of factors)
        """
        return self.factors.weight.shape[1]


class RatingsData(Dataset):

    """
    Wrapper for the ratings data matrix

    The full matrix is not stored, only the supplied ratings, and
    rows of the matrix are created on-the-fly as needed

    :param data_directory: Directory where the data is stored

    """
    def __init__(self, data_directory: str):
        
        data = pd.read_csv(Path(data_directory).joinpath('train_ratings'))

        users  = pd.Series(data['user'].unique())
        items  = pd.Series(data['item'].unique())
        groups = data.groupby('user').groups

        self.data    = data
        self.users   = users
        self.items   = items
        self.groups  = groups
   

    def __len__(self) -> int:
        """Return number of users
        """
        return len(self.users)


    def get_dim(self) -> int:
        """Getter for the matrix dimension (the number of items)
        """
        return len(self.items)


    def __getitem__(self, idx: Union[int, Tensor]):
        """Creates rows of the data matrix on the fly
        """
        if torch.is_tensor(idx):
            idx = idx.numpy()

        user_idx = self.users[idx]

        mat = torch.zeros(len(idx), len(self.items))
        for i, u in enumerate(user_idx):
            user_data = self.data.loc[self.groups[u]]
            ratings_idx = self.items.isin(user_data['item'])
            mat[i,ratings_idx] = Tensor(user_data['rating'].values)

        return mat
        

def train() -> int:

    data_directory = os.environ['SM_CHANNEL_TRAINING']

    ratings_data = RatingsData(data_directory)

    sampler = BatchSampler(RandomSampler(ratings_data),
                           batch_size=100, drop_last=False)

    train_loader = DataLoader(ratings_data, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Recommender(ratings_data.get_dim()).to(device)

    model.train()

    model.init_params()

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss_fcn = nn.MSELoss()

    for batch in train_loader:
        # 'batch' is a tensor with shape (batch_size, 1, n_items)
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        idx = batch != 0
        loss = loss_fcn(output[idx], batch[idx])

        # Add penalty to enforce orthonormal factors
        params = next(model.parameters())
        Id = torch.eye(model.get_dim())
        penalty = 0.1 * ((params.T @ params - Id)**2).sum() / model.get_dim()**2
        print(f"Penalty: {penalty}")
        
        loss_with_penalty = loss + penalty
        loss_with_penalty.backward()

        print(f"Loss: {loss.item()}")

        optimizer.step()

        print(next(model.parameters()))

    return 0 


if __name__ == '__main__':

    train()

