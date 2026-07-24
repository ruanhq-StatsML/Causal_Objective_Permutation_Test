#Simulation for the concept drift as well as the covariate shift
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

'''
Incorporate a couple of types of the distribution shift:
Ranging from the conditional permutation in the response
to the prediction on the new batch of dataset
'''

class DomainRegressor(nn.Module):
    def __init__(self, input_dim, n_layers=2, dropout=0.15):
        super().__init__()
        dims = np.round(
            np.exp(np.linspace(np.log(max(input_dim, 2)), 0, n_layers))
        ).astype(int)
        dims = np.maximum(dims, 1)
        layers_dict = []
        for i in range(n_layers - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers_dict.append(nn.Linear(in_dim, out_dim))
            layers_dict.append(nn.BatchNorm1d(out_dim))
            layers_dict.append(nn.ReLU())
            if dropout > 0:
                layers_dict.append(nn.Dropout(dropout))
        layers_dict.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers_dict)
    def forward(self, X):
        return self.net(X).squeeze(-1)


class TabDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.Y = torch.tensor(Y, dtype = torch.float32)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


#Train the regression model to predict the response here.
def train_domain_regressor(X, Y, n_epochs = 5, batch_size = 16, device = 'cpu'):
    p = X.shape[1]
    model = DomainRegressor(input_dim = p, n_layers = 3)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    X_t = torch.tensor(X, dtype = torch.float32, device = device).detach().requires_grad_(True)
    Y_t = torch.tensor(Y, dtype = torch.float32, device = device).detach().requires_grad_(True)
    #DataLoader:
    train_ds = TabDataset(X_t, Y_t)
    dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    for epoch in range(n_epochs):
        model.train()
        for feature, response in dataloader:
            optimizer.zero_grad()
            loss = criterion(model.forward(feature), response)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model.forward(feature), response)
    print(f" epoch {epoch + 1}/{n_epochs}, loss = {train_loss:.4f}")
    return model



#Also conduct the shifts on Y here:

from adversarial_conceptdrift import AdvConceptDrift

def impose_adv_shift(df1_X, df1_Y, df2_X, df2_Y, method, task = 'reg',
  steps = 5, eps = 0.1, 
  n_layers = 2, n_shift_prop = 0.3,
  n_epochs = 5):
    n1 = df1_X.shape[0]
    n2 = df2_X.shape[0]
    p = df1_X.shape[1]
    df_X = np.vstack([df1_X, df2_X])
    df_Y = np.hstack([df1_Y, df2_Y])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_X)
    n_shift = max(1, int(np.round(p * n_shift_prop)))
    attack_features = sorted(np.random.choice(np.arange(p), n_shift, replace = False))
    feature_ind = attack_features.copy()
    model = train_domain_regressor(X_scaled, df_Y, n_layers = n_layers, n_epochs = n_epochs)
    model.eval()
    attacker = AdvConceptDrift(
        model = model,
        attack_feature_list = feature_ind,
        steps = steps, eps = eps
    )
    METHOD_ADV = [
    "miFGSM", 'sini_FGSM',
    'vmi_FGSM', 'rFGSM', 'jitter'
    ]
    METHOD_MAP = {
      'miFGSM': attacker.forward_miFGSM,
      'sini_FGSM': attacker.sini_FGSM,
      'vmi_FGSM': attacker.vmi_FGSM,
      'rFGSM': attacker.forward_rFGSM,
      'jitter': attacker.forward_jitter
    }
    X_adv_t, Y_t = METHOD_MAP[method](X_scaled, df_Y, n1, task = 'reg')
    X_adv_t = X_adv_t.cpu().numpy()
    Y_t = Y_t.cpu().numpy().ravel()
    return X_adv_t, Y_t, feature_ind, n1
