#TAR-NET:

import torch
import numpy as np

def pdist2sq(x, y):
    # Pairwise squared euclidean distance in PyTorch
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    dist = x2 + y2.t() - 2.0 * torch.matmul(x, y.t())
    return dist


import torch
import torch.nn as nn
import torch.nn.functional as F

class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super(DragonNet, self).__init__()
        
        # Representation (phi)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        # Hypothesis heads (y0 and y1)
        self.y0_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        self.y1_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        # Propensity head
        self.t_head = nn.Linear(200, 1)

    def forward(self, x):
        phi_out = self.phi(x)
        y0_pred = self.y0_head(phi_out)
        y1_pred = self.y1_head(phi_out)
        t_pred = self.t_head(phi_out)
        
        # Concatenating to match your Keras output format: [y0, y1, t, phi]
        return torch.cat([y0_pred, y1_pred, t_pred, phi_out], dim=1)

class BaseLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(BaseLoss, self).__init__()
        self.alpha = alpha
    def split_pred(self, concat_pred):
        preds = {
            'y0_pred': concat_pred[:, 0],
            'y1_pred': concat_pred[:, 1],
            't_pred':  concat_pred[:, 2],
            'phi':     concat_pred[:, 3:]
        }
        return preds
    def treatment_acc(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        # Apply sigmoid to t_pred since it outputs raw logits
        t_prob = torch.sigmoid(p['t_pred'])
        preds = (t_prob > 0.5).float()
        return (preds == t_true).float().mean()
    def treatment_bce(self, t_true, t_pred):
        # BCEWithLogitsLoss expects (input, target)
        return F.binary_cross_entropy_with_logits(t_pred, t_true, reduction='sum')
    def regression_loss(self, y_true, t_true, y0_pred, y1_pred):
        loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_pred))
        loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
        return loss0 + loss1
    def forward(self, concat_pred, concat_true):
    	y_true = concat_pred[:, 0]
    	t_true = concat_pred[:, 1]
        p = self.split_pred(concat_pred)
        lossR = self.regression_loss(y_true, t_true, p['y0_pred'], p['y1_pred'])
        lossP = self.treatment_bce(t_true, p['t_pred'])
        return lossR + self.alpha * lossP

class AIPW_Metrics:
    def __init__(self, data, model, verbose=0):
        self.data = data
        self.model = model
        self.verbose = verbose
        # Prepare indices for PEHEnn
        t_flat = self.data['t'].squeeze()
        self.c_idx = (t_flat == 0).nonzero(as_tuple=True)
        self.t_idx = (t_flat == 1).nonzero(as_tuple=True)

    def split_pred(self, concat_pred):
        # Note: self.data['y_scaler'] is assumed to be a sklearn-style scaler
        y0_raw = concat_pred[:, 0:1].detach().cpu().numpy()
        y1_raw = concat_pred[:, 1:2].detach().cpu().numpy()
        
        preds = {
            'y0_pred': torch.tensor(self.data['y_scaler'].inverse_transform(y0_raw)).to(concat_pred.device),
            'y1_pred': torch.tensor(self.data['y_scaler'].inverse_transform(y1_raw)).to(concat_pred.device),
            't_pred': concat_pred[:, 2],
            'phi': concat_pred[:, 3:]
        }
        return preds

    def find_ynn(self, phi):
        # PyTorch equivalent of dynamic_partition/dynamic_stitch
        phi_c = phi[self.c_idx]
        phi_t = phi[self.t_idx]
        
        dists = torch.sqrt(torch.clamp(pdist2sq(phi_c, phi_t), min=0.0))
        
        # Nearest neighbor indices
        yT_nn_idx = self.c_idx[torch.argmin(dists, dim=0)]
        yC_nn_idx = self.t_idx[torch.argmin(dists, dim=1)]
        
        # Retrieve y values
        yT_nn = self.data['y'][yT_nn_idx]
        yC_nn = self.data['y'][yC_nn_idx]
        
        # Reconstruct full y_nn vector
        y_nn = torch.zeros_like(self.data['y'])
        y_nn[self.t_idx] = yT_nn
        y_nn[self.c_idx] = yC_nn
        return y_nn

    def PEHEnn(self, p):
        y_nn = self.find_ynn(p['phi'])
        t = self.data['t']
        y = self.data['y']
        
        cate_nn_err = torch.mean(((1 - 2*t) * (y_nn - y) - (p['y1_pred'] - p['y0_pred']))**2)
        return cate_nn_err

    def AIPW(self, p):
        t_pred = torch.sigmoid(p['t_pred']).unsqueeze(1)
        t_pred = (t_pred + 0.001) / 1.002 # Stability trick
        
        y_pred = p['y0_pred'] * (1 - self.data['t']) + p['y1_pred'] * self.data['t']
        
        # Clever covariate calculation
        cc = self.data['t'] * (1.0 / t_pred) - (1.0 - self.data['t']) / (1.0 - t_pred)
        cate = cc * (self.data['y'] - y_pred) + p['y1_pred'] - p['y0_pred']
        return cate

    def run_metrics(self, epoch, writer=None):
        self.model.eval()
        with torch.no_grad():
            concat_pred = self.model(self.data['x'])
            p = self.split_pred(concat_pred)
            
            # Calculations
            ate_pred = torch.mean(p['y1_pred'] - p['y0_pred'])
            pehe_nn = self.PEHEnn(p)
            aipw_vec = self.AIPW(p)
            aipw_val = torch.mean(aipw_vec)
            
            # Simulation-based ground truth comparison
            ate_true = torch.mean(self.data['mu_1'] - self.data['mu_0'])
            ate_err = torch.abs(ate_true - ate_pred)
            aipw_err = torch.abs(ate_true - aipw_val)
            
            # Compute raw PEHE
            cate_true = self.data['mu_1'] - self.data['mu_0']
            pehe = torch.mean((cate_true - (p['y1_pred'] - p['y0_pred']))**2)

            if self.verbose > 0:
                print(f"Epoch {epoch} — ate_err: {ate_err:.4f} — aipw_err: {aipw_err:.4f} "
                      f"— cate_err: {torch.sqrt(pehe):.4f} — cate_nn_err: {torch.sqrt(pehe_nn):.4f}")
            
            # If using TensorBoard (SummaryWriter)
            if writer:
                writer.add_scalar('ate_err', ate_err, epoch)
                writer.add_scalar('aipw_err', aipw_err, epoch)
                writer.add_scalar('cate_err', torch.sqrt(pehe), epoch)

def pdist2(X, Y):
    x2 = torch.sum(X ** 2, dim = -1, keepdim = True)
    y2 = torch.sum(Y ** 2, dim = -1, keepdim = True)
    dist = x2 + y2.t() - 2.0 * torch.matmul(X, Y.t())
    return dist

class AIPWModel(nn.Module):
    def __init__(self, input_dim):
        super(AIPWModel, self).__init__()
        self.phi_net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.y0_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        self.y1_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        #propensity score head:
        self.t_head = nn.Linear(200, 1)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.0, std = 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
    	phi = self.phi_net(x)
    	y0_preds = self.y0_head(phi)
    	y1_preds = self.y1_head(phi)
    	t_preds = self.t_head(phi)
    	#concatenate together:
    	return torch.cat([y0_preds, y1_preds, t_preds, phi], dim = 1)
    	#(batch_size, 1)

model = AIPWModel(input_dim = 10)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = reg_l2)

class AIPW_Loss(Loss):
    def __init__(self, data, model, verbose = 0):
        self.data = data
        self.model = model
        self.verbose = verbose
        #PEHE(L2 loss for the treatment effect)
        t_flat = self.data['t'].squeeze()
        self.c_idx = (t_flat == 0).nonzero(as_tuple = True)
        self.t_idx = (t_flat == 1).nonzero(as_tuple = True)
    def split_pred(self, concat_pred):
        y0_raw = concat_pred[:, 0:1].detach().cpu().numpy()
        y1_raw = concat_pred[:, 1:2].detach().cpu().numpy()
        preds = {
          'y0_pred': torch.tensor(self.data['y_scaler'].inverse_transform(y0_raw)).to(concat_pred.device)
          'y1_pred': torch.tensor(self.data['y_scaler'].inverse_transform(y1_raw)).to(concat_pred.device)
          't_pred': concat_pred[:, 2],
          'phi': concat_pred[:, 3]
        }
        return preds
    def PEHEnn(self, p):
        y_nn = self.find_ynn(p['phi'])
        t = self.data['t']
        y = self.data['y']
        cate_nn_err = torch.mean(((1 - 2 * t) * (y_nn - y) -
        	(p['y1_pred'] - p['y0_pred'])) ** 2)
        return cate_nn_err
    def AIPW(self, p):
        t_pred = torch.sigmoid(p['t_pred']).unsqueeze(1)
        t_pred = (t_pred + 0.001)/0.002
        y_pred = p['y0']