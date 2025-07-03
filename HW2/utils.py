import torch
import numpy as np
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, num_classes=3, source='random'):
    if source == 'random':
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, n_features=4, n_classes=num_classes, n_informative=3,n_redundant=0) #Генерируем синтетические данные для многоклассовой классификации
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    elif source == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    y_pred_class = y_pred.argmax(dim=1) #Получаем индекс класса с наибольшей вероятностью
    correct = (y_pred_class == y_true).float() #Сравниваем предсказанный и истинный класс
    return correct.mean().item()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)

def metrics(y_pred_logits, y_true):
    """
    Вычисляем precision, recall, F1 и ROC-AUC для многоклассовой задачи.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import torch.nn.functional as F

    y_pred_probability = F.softmax(y_pred_logits, dim=1).detach().cpu().numpy() #переводим логиты в вероятности принадлежности к классам
    y_pred = y_pred_probability.argmax(axis=1) #берем макс
    y_true = y_true.detach().cpu().numpy() #истинные значения

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    y_true_onehot = F.one_hot(torch.tensor(y_true), num_classes=y_pred_probability.shape[1]).numpy()
    roc_auc = roc_auc_score(y_true_onehot, y_pred_probability, average='macro', multi_class='ovr')

    return precision, recall, f1, roc_auc
