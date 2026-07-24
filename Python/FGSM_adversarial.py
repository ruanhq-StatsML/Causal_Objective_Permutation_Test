import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

def adv_pert_reg(df1_X, df1_Y, df2_X, df2_Y, n_shift_prop=0.3, batch_size=16,
                 n_epoch=5, n_hidden_layers=2, dropout_rate=0.15,
                 epsilon=0.15, random_state=28, device='cpu'):
    X = np.vstack([df1_X, df2_X])
    Y = np.hstack([df1_Y, df2_Y])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n1 = df1_X.shape[0]
    p = df1_X.shape[1]
    n_shift = max(1, int(np.round(p * n_shift_prop)))
    X_df1_scaled = X_scaled[:n1, :]
    X_df2_scaled = X_scaled[n1:, :]
    class DomainRegressor(nn.Module):
        def __init__(self, input_dim, n_layers, dropout):
            super().__init__()
            dims = np.round(np.exp(np.linspace(np.log(input_dim), 0, n_layers))).astype(int)
            dims = np.maximum(dims, 1)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)
    model = DomainRegressor(input_dim=p, n_layers=n_hidden_layers,
                            dropout=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    for epoch in range(n_epoch):
        model.train()
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i: i + batch_size]
            batch_Y = Y_tensor[i: i + batch_size]
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_tensor), Y_tensor).item()
        print(f"Epoch {epoch+1}/{n_epoch}, 全量训练 Loss: {train_loss:.6f}")
    model.eval()
    samples = torch.tensor(X_df1_scaled, dtype=torch.float32).to(device)
    samples.requires_grad = True
    y_true = torch.tensor(df1_Y, dtype=torch.float32).to(device)
    preds = model(samples)
    loss = criterion(preds, y_true)
    model.zero_grad()
    loss.backward()
    grad_abs = samples.grad.abs()
    mean_grad_abs = grad_abs.mean(dim=0)
    topk_vals, topk_idx = torch.topk(mean_grad_abs, n_shift)
    mask = torch.zeros_like(samples.grad)
    mask[:, topk_idx] = 1.0
    perturb = epsilon * mask * torch.sign(samples.grad)
    adv_samples = samples + perturb
    adv_samples_np = adv_samples.detach().cpu().numpy()
    X_adv = np.vstack([adv_samples_np, X_df2_scaled])   
    Y_adv = np.concatenate([df1_Y, df2_Y])              
    return X_adv, Y_adv, topk_idx.cpu(), n1
