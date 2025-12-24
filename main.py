from utils import *
from sklearn.model_selection import StratifiedKFold
from model.EEGNet import EEGNet_BNN
from trainer.trainer import train, test
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='End-to-End Training for Deep Learning Models')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training (default: 8)')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout (default: 0.25)')
params = parser.parse_args()
# optimized parameters: learning_rate, weight_decay, dropout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed_everything(42)

subjs = ['02', '03', '04', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23']
X_list, y_list = data_load(subjs)
cv_acc, cv_f1, cv_ece = [], [], []
for i, s in enumerate(subjs):
    X, y = X_list[i], y_list[i]
    acc_list, f1_list, ece_list = [], [], []
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        X_train, y_train, X_val, y_val = train_valid_split(X_train, y_train, test_size=0.25)

        train_tensor = TensorDataset(make_Tensor(X_train), torch.Tensor(y_train).long())
        valid_tensor = TensorDataset(make_Tensor(X_val), torch.Tensor(y_val).long())
        test_tensor = TensorDataset(make_Tensor(X_test), torch.Tensor(y_test).long())
        train_loader = DataLoader(dataset=train_tensor, batch_size=params.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_tensor, batch_size=params.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_tensor, batch_size=params.batch_size, shuffle=False)

        model = EEGNet_BNN(dropout=params.dropout)
        model.to(device)
        criterion = torch.nn.NLLLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        best_f1 = 0
        best_epoch = 0
        with tqdm(range(params.epochs), unit='batch') as tepoch:
            for e in tepoch:
                train(model, train_loader, criterion, optimizer, device)
                val_loss, _, val_f1, _ = test(model, valid_loader, criterion, device)
                tepoch.set_postfix(loss=val_loss, f1=val_f1, best_f1=best_f1, best_epoch=best_epoch)
                if best_f1 < val_f1:
                    best_f1 = val_f1
                    best_epoch = e
                    save_path = './bnn_ckpt/subj={}_fold={}.pth'.format(s, fold)
                    torch.save({'epoch': e + 1, 'model_state_dict': model.state_dict()}, save_path)
        checkpoint = torch.load('./bnn_ckpt/subj={}_fold={}.pth'.format(s, fold))
        model.load_state_dict(checkpoint['model_state_dict'])
        _, test_acc, test_f1, test_ece = test(model, test_loader, criterion, device)
        acc_list.append(test_acc)
        f1_list.append(test_f1)
        ece_list.append(test_ece)
        print('{}-fold CV test ACC: {:.4f}, F1: {:.4f}, ECE: {:.4f}'.format(fold, test_acc, test_f1, test_ece))

    print('Subj={}, LOOCV test ACC: {:.4f}, F1: {:.4f}, ECE: {:.4f}'.format(s, np.mean(acc_list), np.mean(f1_list), np.mean(ece_list)))
    cv_acc.append(np.mean(acc_list))
    cv_f1.append(np.mean(f1_list))
    cv_ece.append(np.mean(ece_list))


print('LOOCV test ACC: {:.4f}, F1: {:.4f}, ECE: {:.4f}'.format(np.mean(cv_acc), np.mean(cv_f1)), np.mean(cv_ece))
