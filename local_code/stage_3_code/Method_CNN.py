from local_code.base_class.method import method

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, instances, channel='single'):
        self.data = []
        for inst in instances:
            img   = np.array(inst['image'], dtype=np.float32) / 255.0
            label = int(inst['label'])

            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim == 3 and img.shape[2] in (1, 3):
                img = img.transpose(2, 0, 1)
                if channel == 'single':
                    img = img[0:1, :, :]

            self.data.append((
                torch.tensor(img,   dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, conv_configs=None, hidden_dim=256, dropout_rate=0.5):
        super().__init__()

        if conv_configs is None:
            conv_configs = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            ]

        layers = []
        current_channels = in_channels
        for cfg in conv_configs:
            layers += [
                nn.Conv2d(current_channels, cfg['out_channels'],
                          cfg.get('kernel_size', 3), cfg.get('stride', 1), cfg.get('padding', 1)),
                nn.BatchNorm2d(cfg['out_channels']),
                nn.ReLU(inplace=True),
            ]
            if cfg.get('pool', True):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = cfg['out_channels']

        self.conv_net       = nn.Sequential(*layers)
        self.adaptive_pool  = nn.AdaptiveAvgPool2d((4, 4))
        flat_size           = current_channels * 4 * 4
        self.classifier     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)


class Method_CNN(method):
    # dataset config
    in_channels   = 1
    num_classes   = 10
    channel_mode  = 'single'
    label_offset  = 0       # set to -1 when labels are 1-indexed (e.g. ORL)

    # model hyper-params
    conv_configs  = None
    hidden_dim    = 256
    dropout_rate  = 0.5

    # training hyper-params
    batch_size    = 64
    num_epochs    = 20
    lr            = 1e-3
    loss_fn       = 'cross_entropy'   # 'cross_entropy' or 'nll'

    # output
    dataset_name          = 'Dataset'
    save_curve_path       = None   # filepath to save the learning-curve PNG
    save_convergence_path = None   # filepath to save the convergence plot PNG

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)

    # ── private helpers ───────────────────────────────────────────────────────

    def _train_one_epoch(self, model, loader, criterion, optimizer, device):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
        return total_loss / total, correct / total

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)
                all_preds .extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return total_loss / total, correct / total, all_preds, all_labels

    def _plot_curves(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'{self.dataset_name} – Learning Curves', fontsize=14)

        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['test_loss'],  label='Test Loss',  marker='s', markersize=3)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss over Epochs'); ax1.legend(); ax1.grid(True)

        ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
        ax2.plot(epochs, history['test_acc'],  label='Test Accuracy',  marker='s', markersize=3)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy over Epochs'); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        if self.save_curve_path:
            plt.savefig(self.save_curve_path, dpi=150)
            print(f'  Saved learning curve → {self.save_curve_path}')
        plt.close()

    def _plot_convergence(self, history):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f'{self.dataset_name} – Convergence', fontsize=14)

        ax.plot(history['train_acc'], history['train_loss'], marker='o', markersize=3, label='Train')
        ax.plot(history['test_acc'],  history['test_loss'],  marker='s', markersize=3, label='Test', color='orange')
        ax.set_xlabel('Accuracy'); ax.set_ylabel('Loss')
        ax.set_title('Convergence'); ax.legend(); ax.grid(True)

        plt.tight_layout()
        if self.save_convergence_path:
            plt.savefig(self.save_convergence_path, dpi=150)
            print(f'  Saved convergence plot → {self.save_convergence_path}')
        plt.close()

    # ── public entry point ────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {self.dataset_name}")
        print(f"{'='*60}")

        train_instances = self.data['train']
        test_instances  = self.data['test']

        if self.label_offset != 0:
            for inst in train_instances + test_instances:
                inst['label'] = inst['label'] + self.label_offset

        train_ds     = ImageDataset(train_instances, channel=self.channel_mode)
        test_ds      = ImageDataset(test_instances,  channel=self.channel_mode)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, num_workers=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = CNN(self.in_channels, self.num_classes, self.conv_configs,
                     self.hidden_dim, self.dropout_rate).to(device)
        print(f'  Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

        if self.loss_fn == 'nll':
            criterion = nn.NLLLoss()
            model     = nn.Sequential(model, nn.LogSoftmax(dim=1))
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(1, self.num_epochs + 1):
            tr_loss, tr_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
            te_loss, te_acc, _, _ = self._evaluate(model, test_loader, criterion, device)
            scheduler.step(te_loss)
            history['train_loss'].append(tr_loss)
            history['train_acc'] .append(tr_acc)
            history['test_loss'] .append(te_loss)
            history['test_acc']  .append(te_acc)
            print(f"Epoch {epoch:>3}/{self.num_epochs} | "
                  f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
                  f"Test  loss: {te_loss:.4f}  acc: {te_acc:.4f}")

        _, final_acc, preds, true_labels = self._evaluate(model, test_loader, criterion, device)
        print(f"\n  Final Test Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
        print("\n  Classification Report:")
        print(classification_report(true_labels, preds, zero_division=0))

        self._plot_curves(history)
        self._plot_convergence(history)

        return {'pred_y': preds, 'true_y': true_labels, 'final_acc': final_acc, 'history': history}
