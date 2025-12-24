import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from utils import expected_calibration_error
import torch.nn.functional as F

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for x, y in train_loader:  # Iterate in batches over the training dataset.
        x = x.to(device)
        y = y.to(device)
        mu, sigma = model(x)  # Perform a single forward pass.
        prob_total = torch.zeros((50, y.size(0), 2)).to(device)
        for t in range(50):
            epsilon = torch.randn(sigma.size()).to(device)
            logit = mu + torch.mul(sigma, epsilon)
            prob_total[t] = F.softmax(logit, dim=1)
        prob_ave = torch.mean(prob_total, dim=0)
        loss = criterion(torch.log(prob_ave), y) # Compute the loss.

        optimizer.zero_grad()
        loss.backward()  # Derive gradients.
        optimizer.step()

def test(model, loader, criterion, device):
    model.eval()
    model.apply(apply_dropout)
    with torch.no_grad():
        pred_list, prob_list = [], []
        gt_list, unc_list = [], []
        loss_ = 0
        total_cnt = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prob_total = torch.zeros((50, y.size(0), 2)).to(device)
            sigma_total = torch.zeros((50, y.size(0), 2)).to(device)
            for t in range(50):
                mu, sigma = model(x)
                prob_total[t] = F.softmax(mu, dim=1)
                sigma_total[t] = sigma
            prob_ave = torch.mean(prob_total, dim=0)
            loss = criterion(torch.log(prob_ave), y)
            loss_ += loss.item()
            total_cnt += 1
            pred = prob_ave.argmax(dim=1)
            prob_list.extend(prob_ave.detach().cpu().tolist())
            pred_list.extend(pred.detach().cpu().tolist())
            gt_list.extend(y.view(-1).detach().cpu().tolist())
            unc = (torch.var(prob_total, dim=-1).mean(dim=0) + torch.pow(sigma_total, 2).mean(dim=-1).mean(dim=0)).detach().cpu().tolist()
            unc_list.extend(unc)
        acc = accuracy_score(gt_list, pred_list)
        f1 = f1_score(gt_list, pred_list)
        ece = expected_calibration_error(prob_list, gt_list)
    return loss_ / total_cnt, acc, f1, ece