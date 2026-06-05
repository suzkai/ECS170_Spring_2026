#############################################################
# BASELINE: 2-layer GCN, hidden_dim=64, dropout=0.5,
#           Adam optimizer, weight_decay=5e-4, 200 epochs
#############################################################
from local_code.base_class.method import method

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import os


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # H = A_hat * X * W  (A_hat is the pre-normalized adjacency from the dataset loader)
        return torch.spmm(adj, x @ self.weight)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)


class Method_GCN(method):
    hidden_dim = 64
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    num_epochs = 200
    dataset_name = 'cora'
    save_curve_path = 'result/stage_5_result/'

    def __init__(self):
        super().__init__('GCN', 'Graph Convolutional Network for Node Classification')
        self.data = None

    def _train_step(self, model, optimizer, features, adj, labels, idx_train):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = output[idx_train].argmax(dim=1).cpu().numpy()
        acc = accuracy_score(labels[idx_train].cpu().numpy(), preds)
        return loss.item(), acc

    def _eval(self, model, features, adj, labels, idx):
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
        loss = F.nll_loss(output[idx], labels[idx]).item()
        preds = output[idx].argmax(dim=1).cpu().numpy()
        acc = accuracy_score(labels[idx].cpu().numpy(), preds)
        return loss, acc, preds

    def _plot_curves(self, history):
        os.makedirs(self.save_curve_path, exist_ok=True)
        epochs = range(1, len(history['train_loss']) + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history['train_loss'], label='Train Loss')
        ax.plot(epochs, history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'GCN Loss - {self.dataset_name}')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.save_curve_path + f'{self.dataset_name}_GCN_curve_loss.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history['train_acc'], label='Train Accuracy')
        ax.plot(epochs, history['val_acc'], label='Val Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'GCN Accuracy - {self.dataset_name}')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.save_curve_path + f'{self.dataset_name}_GCN_curve_acc.png')
        plt.close(fig)

    def run(self):
        graph = self.data['graph']
        train_test_val = self.data['train_test_val']

        features = graph['X']
        labels = graph['y']
        adj = graph['utility']['A']
        idx_train = train_test_val['idx_train']
        idx_val = train_test_val['idx_val']
        idx_test = train_test_val['idx_test']

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print('device:', device)

        features = features.to(device)
        labels = labels.to(device)
        adj = adj.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

        in_features = features.shape[1]
        num_classes = int(labels.max().item()) + 1

        model = GCN(in_features, self.hidden_dim, num_classes, self.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_step(model, optimizer, features, adj, labels, idx_train)
            val_loss, val_acc, _ = self._eval(model, features, adj, labels, idx_val)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if epoch % 50 == 0:
                print(f'Epoch {epoch}/{self.num_epochs} - train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        test_loss, test_acc, test_preds = self._eval(model, features, adj, labels, idx_test)
        true_labels = labels[idx_test].cpu().numpy()

        print(f'\nTest Results ({self.dataset_name}):')
        print('Accuracy:', accuracy_score(true_labels, test_preds))
        print('Precision:', precision_score(true_labels, test_preds, average='macro', zero_division=0))
        print('Recall:', recall_score(true_labels, test_preds, average='macro', zero_division=0))
        print('F1 Score:', f1_score(true_labels, test_preds, average='macro', zero_division=0))
        print('\nClassification Report:')
        print(classification_report(true_labels, test_preds, zero_division=0))

        os.makedirs(self.save_curve_path, exist_ok=True)
        result_path = self.save_curve_path + f'{self.dataset_name}_Results.txt'
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'GCN Node Classification - {self.dataset_name}\n')
            f.write(f'Accuracy: {accuracy_score(true_labels, test_preds)}\n')
            f.write(f'Precision: {precision_score(true_labels, test_preds, average="macro", zero_division=0)}\n')
            f.write(f'Recall: {recall_score(true_labels, test_preds, average="macro", zero_division=0)}\n')
            f.write(f'F1 Score: {f1_score(true_labels, test_preds, average="macro", zero_division=0)}\n\n')
            f.write('Classification Report:\n')
            f.write(classification_report(true_labels, test_preds, zero_division=0))

        self._plot_curves(history)

        return {
            'pred_y': test_preds.tolist(),
            'true_y': true_labels.tolist(),
            'final_acc': test_acc,
            'history': history,
        }
