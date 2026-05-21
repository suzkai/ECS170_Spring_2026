import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim

# taken from https://gist.github.com/sebleier/554280
stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
}
def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub(r'<[^>]+>', ' ', text) # remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text) # remove punctuation
    # tokenize and remove stop words
    tokens = []
    for word in text.split():
        if word not in stop_words and len(word) > 1:
            tokens.append(word)
    return tokens


def build_vocab(all_tokens, max_size):
    # count word frequencies
    freq = {}
    for tokens in all_tokens:
        for word in tokens:
            freq[word] = freq.get(word, 0) + 1

    # sort by frequency and take top words
    sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in sorted_words[:max_size - 2]:
        vocab[word] = len(vocab)
    return vocab


def encode_data(all_tokens, vocab, max_len):
    # convert tokens to integer sequences
    unk_idx = vocab['<UNK>']
    pad_idx = vocab['<PAD>']
    result = np.full((len(all_tokens), max_len), pad_idx, dtype=np.int32)
    for i, tokens in enumerate(all_tokens):
        ids = [vocab.get(w, unk_idx) for w in tokens]
        ids = ids[:max_len]
        result[i, :len(ids)] = ids
    return result

# RNN: single layer Tanh
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.Wxh = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.Whh.in_features, device=x.device)
        for t in range(x.size(1)):
            emb = self.embedding(x[:, t])
            h = torch.tanh(self.Wxh(emb) + self.Whh(h))
        return self.fc(h)

    def step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            preds = self(x).argmax(dim=1)
        return loss.item(), preds.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            output = self(x)
        return output.argmax(dim=1).cpu().numpy()

# RNN1: LSTM only
# Note* LSTM replaces relu with its own gating, so no activation
class RNN1(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out[:, -1, :])

    def step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        loss = criterion(self(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            preds = self(x).argmax(dim=1)
        return loss.item(), preds.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            return self(x).argmax(dim=1).cpu().numpy()


# RNN2: LSTM, multilayer
# Multilayer set to 2 but can be changed for extra testing
class RNN2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2): # Change Number of layers here
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out[:, -1, :])

    def step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        loss = criterion(self(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            preds = self(x).argmax(dim=1)
        return loss.item(), preds.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            return self(x).argmax(dim=1).cpu().numpy()


# RNN3: LSTM, multilayer, dropout
class RNN3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(self.dropout(self.embedding(x)))
        return self.fc(self.dropout(out[:, -1, :]))

    def step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        loss = criterion(self(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            preds = self(x).argmax(dim=1)
        return loss.item(), preds.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            return self(x).argmax(dim=1).cpu().numpy()

class Method_RNN:
    dataset_name = 'IMDB Sentiment'
    save_curve_path = 'result/stage_4_result/classification_curve.png'

    def __init__(self):
        self.data = None
        self.model_class = RNN

    def _plot_curves(self, history, model_name):
        epochs = range(1, len(history['train_loss']) + 1)
        dir_path = self.save_curve_path.replace('classification_curve.png', '')
        base = f'{dir_path}{model_name}_classification_curve'

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(epochs, history['train_loss'], label='Train Loss')
        ax1.plot(epochs, history['test_loss'], label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss - ' + self.dataset_name)
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()
        loss_path = base + '_loss.png'
        fig1.savefig(loss_path)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
        ax2.plot(epochs, history['test_acc'], label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy - ' + self.dataset_name)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        acc_path = base + '_acc.png'
        fig2.savefig(acc_path)
        plt.close(fig2)

    def run(self):
        train_pairs = self.data['train']
        test_pairs = self.data['test']

        # clean and tokenize text
        print('cleaning...')
        train_tokens = [clean_text(text) for text, _ in train_pairs]
        test_tokens = [clean_text(text) for text, _ in test_pairs]
        train_labels = np.array([label for _, label in train_pairs], dtype=np.int32)
        test_labels = np.array([label for _, label in test_pairs], dtype=np.int32)

        # build vocab from training data
        vocab = build_vocab(train_tokens, self.max_vocab_size)
        print('vocabulary size:', len(vocab))

        # encode sequences
        X_train = encode_data(train_tokens, vocab, self.max_seq_len)
        X_test = encode_data(test_tokens, vocab, self.max_seq_len)

        # set up device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print('device:', device)

        # create model, optimizer, loss
        
        model = self.model_class(len(vocab), self.embed_dim, self.hidden_dim, self.num_classes,
                                 **({} if self.model_class in (RNN, RNN1) else
                                    {'num_layers': self.num_layers} if self.model_class == RNN2 else
                                    {'num_layers': self.num_layers, 'dropout': self.dropout})).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        n = len(X_train)

        for epoch in range(1, self.num_epochs + 1):
            # shuffle training data each epoch
            idx = np.random.permutation(n)
            X_train = X_train[idx]
            train_labels = train_labels[idx]

            # training loop
            model.train()
            total_loss = 0.0
            total_correct = 0
            for start in range(0, n, self.batch_size):
                xb = torch.tensor(X_train[start:start + self.batch_size], dtype=torch.long).to(device)
                yb = torch.tensor(train_labels[start:start + self.batch_size], dtype=torch.long).to(device)
                loss, preds = model.step(xb, yb, optimizer, criterion)
                total_loss += loss * len(xb)
                total_correct += (preds == yb.cpu().numpy()).sum()

            train_loss = total_loss / n
            train_acc = total_correct / n

            # evaluate on test set
            model.eval()
            test_preds_list = []
            test_loss_list = []
            for start in range(0, len(X_test), self.batch_size):
                xb = torch.tensor(X_test[start:start + self.batch_size], dtype=torch.long).to(device)
                yb = torch.tensor(test_labels[start:start + self.batch_size], dtype=torch.long).to(device)
                with torch.no_grad():
                    logits = model(xb)
                test_loss_list.append(criterion(logits, yb).item())
                test_preds_list.append(logits.argmax(dim=1).cpu().numpy())

            test_loss = float(np.mean(test_loss_list))
            test_acc = accuracy_score(test_labels, np.concatenate(test_preds_list))

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f'Epoch {epoch}/{self.num_epochs} - train loss: {train_loss}, train acc: {train_acc} / test loss: {test_loss}, test acc: {test_acc}')

        # EVALUATION
        model.eval()
        all_preds = []
        for start in range(0, len(X_test), self.batch_size):
            xb = torch.tensor(X_test[start:start + self.batch_size], dtype=torch.long).to(device)
            all_preds.append(model.predict(xb))
        final_preds = np.concatenate(all_preds).tolist()
        true_list = test_labels.tolist()

        print('\nResults:')
        print('Accuracy:', accuracy_score(true_list, final_preds))
        print('Precision:', precision_score(true_list, final_preds, average='binary', zero_division=0))
        print('Recall:', recall_score(true_list, final_preds, average='binary', zero_division=0))
        print('F1 Score:', f1_score(true_list, final_preds, average='binary', zero_division=0))
        print('\nClassification Report:')
        print(classification_report(true_list, final_preds, target_names=['negative', 'positive'], zero_division=0))

        result_path = 'result/stage_4_result/Results.txt'
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'{self.model_class.__name__} {self.num_epochs} epochs:\n')
            f.write(f'Accuracy: {accuracy_score(true_list, final_preds)}\n')
            f.write(f'Precision: {precision_score(true_list, final_preds, average="binary", zero_division=0)}\n')
            f.write(f'Recall: {recall_score(true_list, final_preds, average="binary", zero_division=0)}\n')
            f.write(f'F1 Score: {f1_score(true_list, final_preds, average="binary", zero_division=0)}\n')
            f.write('\nClassification Report:\n')
            f.write(classification_report(true_list, final_preds, target_names=['negative', 'positive'], zero_division=0) + '\n\n')

        self._plot_curves(history, type(model).__name__)

        return {'pred_y': final_preds, 'true_y': true_list, 'final_acc': accuracy_score(true_list, final_preds), 'history': history}
