#############################################################
# ADDED 3-layer GCN with residual, batch norm, early stopping, 
#       LR scheduler, hyperparameters for each dataset
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
import os


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # H = A_hat * X * W  (A_hat is the pre-normalized adjacency from the dataset loader)
        return torch.spmm(adj, x @ self.weight)


class GCN_Improved(nn.Module):
    """3-layer GCN with residual connection and batch normalization."""

    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim)
        self.layer3 = GCNLayer(hidden_dim, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        h1 = F.relu(self.bn1(self.layer1(x, adj)))
        h1 = F.dropout(h1, self.dropout, training=self.training)

        # residual keeps gradient flowing and prevents over-smoothing
        h2 = F.relu(self.bn2(self.layer2(h1, adj))) + h1
        h2 = F.dropout(h2, self.dropout, training=self.training)

        return F.log_softmax(self.layer3(h2, adj), dim=1)


class Method_GCN_Improved(method):
    hidden_dim = 128
    lr = 0.01
    num_epochs = 500   # high ceiling; early stopping cuts this short
    patience = 50
    dataset_name = 'cora'
    save_curve_path = 'result/stage_5_result/'

    # per-dataset dropout / weight_decay applied automatically in run()
    _dataset_configs = {
        'cora':     {'dropout': 0.5, 'weight_decay': 5e-4},
        'citeseer': {'dropout': 0.6, 'weight_decay': 1e-3},
        'pubmed':   {'dropout': 0.5, 'weight_decay': 1e-3},
    }

    def __init__(self):
        super().__init__('GCN_Improved', 'Improved GCN: 3-layer, residual, BN, early stopping')
        self.data = None
        self.dropout = 0.5
        self.weight_decay = 5e-4

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

    def _plot_curves(self, history, best_epoch):
        os.makedirs(self.save_curve_path, exist_ok=True)
        epochs = range(1, len(history['train_loss']) + 1)
        prefix = self.save_curve_path + f'{self.dataset_name}_GCN_improved_curve'

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history['train_loss'], label='Train Loss')
        ax.plot(epochs, history['val_loss'], label='Val Loss')
        ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'GCN Improved Loss - {self.dataset_name}')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(prefix + '_loss.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history['train_acc'], label='Train Accuracy')
        ax.plot(epochs, history['val_acc'], label='Val Accuracy')
        ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'GCN Improved Accuracy - {self.dataset_name}')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(prefix + '_acc.png')
        plt.close(fig)

    def run(self):
        # apply dataset-specific hyperparameters
        if self.dataset_name in self._dataset_configs:
            cfg = self._dataset_configs[self.dataset_name]
            self.dropout = cfg['dropout']
            self.weight_decay = cfg['weight_decay']

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

        model = GCN_Improved(in_features, self.hidden_dim, num_classes, self.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # halve lr when val loss stalls for 20 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_step(model, optimizer, features, adj, labels, idx_train)
            val_loss, val_acc, _ = self._eval(model, features, adj, labels, idx_val)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch} (best epoch {best_epoch})')
                    break

            if epoch % 50 == 0:
                print(f'Epoch {epoch} - train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        # restore weights from best val epoch before evaluating on test set
        model.load_state_dict(best_state)

        _, test_acc, test_preds = self._eval(model, features, adj, labels, idx_test)
        true_labels = labels[idx_test].cpu().numpy()

        print(f'\nTest Results ({self.dataset_name}):')
        print('Accuracy:', accuracy_score(true_labels, test_preds))
        print('Precision:', precision_score(true_labels, test_preds, average='macro', zero_division=0))
        print('Recall:', recall_score(true_labels, test_preds, average='macro', zero_division=0))
        print('F1 Score:', f1_score(true_labels, test_preds, average='macro', zero_division=0))
        print('\nClassification Report:')
        print(classification_report(true_labels, test_preds, zero_division=0))

        os.makedirs(self.save_curve_path, exist_ok=True)
        result_path = self.save_curve_path + f'{self.dataset_name}_GCN_improved_Results.txt'
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'GCN Improved Node Classification - {self.dataset_name}\n')
            f.write(f'Best epoch: {best_epoch}\n')
            f.write(f'Accuracy:  {accuracy_score(true_labels, test_preds)}\n')
            f.write(f'Precision: {precision_score(true_labels, test_preds, average="macro", zero_division=0)}\n')
            f.write(f'Recall:    {recall_score(true_labels, test_preds, average="macro", zero_division=0)}\n')
            f.write(f'F1 Score:  {f1_score(true_labels, test_preds, average="macro", zero_division=0)}\n\n')
            f.write('Classification Report:\n')
            f.write(classification_report(true_labels, test_preds, zero_division=0))

        self._plot_curves(history, best_epoch)

        return {
            'pred_y': test_preds.tolist(),
            'true_y': true_labels.tolist(),
            'final_acc': test_acc,
            'best_epoch': best_epoch,
            'history': history,
        }
