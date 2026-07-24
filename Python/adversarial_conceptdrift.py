"""Adversarial concept-drift attacks used by impose_adv_shift."""
import numpy as np
import torch
import torch.nn as nn


class AdvConceptDrift:
    def __init__(self, model, attack_feature_list, steps=5, eps=0.1):
        self.model = model
        self.attack_feature_list = [int(i) for i in attack_feature_list]
        self.steps = int(steps)
        self.eps = float(eps)

    def _feature_mask(self, p, device):
        mask = torch.zeros(p, dtype=torch.float32, device=device)
        for idx in self.attack_feature_list:
            mask[idx] = 1.0
        return mask

    def _run_attack(self, X, Y, n1, jitter=False, random_sign=False):
        device = next(self.model.parameters()).device
        X_t = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32, device=device)
        Y_t = torch.tensor(np.asarray(Y, dtype=float).ravel(), dtype=torch.float32, device=device)
        feat_mask = self._feature_mask(X_t.shape[1], device)
        X_adv = X_t.clone().detach()

        for _ in range(self.steps):
            X_work = X_adv.clone().detach()
            X_work[:n1].requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            preds = self.model(X_work[:n1])
            loss = nn.MSELoss()(preds, Y_t[:n1])
            loss.backward()
            grad = X_work.grad[:n1].detach()
            if jitter:
                grad = grad + 0.01 * torch.randn_like(grad)
            if random_sign:
                grad = grad * torch.sign(torch.randn_like(grad))
            step = self.eps * feat_mask * torch.sign(grad)
            X_adv = X_adv.clone()
            X_adv[:n1] = X_adv[:n1] + step
        return X_adv, Y_t

    def forward_miFGSM(self, X, Y, n1, task="reg"):
        return self._run_attack(X, Y, n1)

    def sini_FGSM(self, X, Y, n1, task="reg"):
        return self._run_attack(X, Y, n1, random_sign=True)

    def vmi_FGSM(self, X, Y, n1, task="reg"):
        return self._run_attack(X, Y, n1, jitter=True)

    def forward_rFGSM(self, X, Y, n1, task="reg"):
        device = next(self.model.parameters()).device
        X_t = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32, device=device)
        Y_t = torch.tensor(np.asarray(Y, dtype=float).ravel(), dtype=torch.float32, device=device)
        feat_mask = self._feature_mask(X_t.shape[1], device)
        delta = self.eps * feat_mask * torch.randn_like(X_t[:n1])
        X_pert = X_t.clone()
        X_pert[:n1] = X_pert[:n1] + delta
        attacker = AdvConceptDrift(
            model=self.model,
            attack_feature_list=self.attack_feature_list,
            steps=self.steps,
            eps=self.eps,
        )
        return attacker.forward_miFGSM(
            X_pert.detach().cpu().numpy(), Y_t.detach().cpu().numpy(), n1, task=task
        )

    def forward_jitter(self, X, Y, n1, task="reg"):
        return self._run_attack(X, Y, n1, jitter=True)
