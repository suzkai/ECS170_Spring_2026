import os
import re
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def clean_text_gen(text):
    text = text.lower() # lowercase
    text = re.sub(r'[^a-z0-9\s\'\.\,\!\?]', ' ', text) # keep basic punctuation
    return text.split()

class SequenceDataset(Dataset):
    def __init__(self, token_lists, word2idx, seq_len):
        self.samples = []
        pad = word2idx['<PAD>'] # unknown token index
        unk = word2idx['<UNK>'] # end of sequence token index
        eos = word2idx['<EOS>'] # padding token index

        for tokens in token_lists:
            ids = [word2idx.get(t, unk) for t in tokens] + [eos]
            for i in range(len(ids) - 1):
                # get context window
                ctx = ids[max(0, i - seq_len + 1): i + 1]
                if len(ctx) < seq_len:
                    ctx = [pad] * (seq_len - len(ctx)) + ctx
                self.samples.append((
                    torch.tensor(ctx, dtype=torch.long),
                    torch.tensor(ids[i + 1], dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # convert word to embed
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True) # single-layer RNN
        self.fc = nn.Linear(hidden_dim, vocab_size) # projects hidden state to vocab scores

    def forward(self, x, hidden=None):
        emb = self.embedding(x) # embed tokens
        out, hidden = self.rnn(emb, hidden) # RNN layer
        logits = self.fc(out) # output
        return logits, hidden


class Method_RNN_Generation:
    dataset_name = 'Joke Generation'
    save_curve_path = 'result/stage_4_result/generation_curve.png'

    def __init__(self):
        self.word2idx = None
        self.idx2word = None

    def _build_vocab(self, token_lists):
        # count word frequencies
        freq = {}
        for tokens in token_lists:
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

        specials = ['<PAD>', '<UNK>', '<EOS>']
        word2idx = {w: i for i, w in enumerate(specials)}
        for word in sorted(freq, key=lambda w: freq[w], reverse=True)[:self.max_vocab_size - len(specials)]:
            word2idx[word] = len(word2idx)
        idx2word = {i: w for w, i in word2idx.items()}
        return word2idx, idx2word

    def _generate(self, model, seed_words, device):
        model.eval()
        unk_idx = self.word2idx['<UNK>'] # unknown token index
        eos_idx = self.word2idx['<EOS>'] # end of sequence token index
        pad_idx = self.word2idx['<PAD>'] # padding token index

        # encode seed words
        seed_ids = [self.word2idx.get(w, unk_idx) for w in seed_words]
        context = seed_ids[-self.seq_len:]
        if len(context) < self.seq_len:
            context = [pad_idx] * (self.seq_len - len(context)) + context

        generated = []
        for w in seed_words:
            known = w in self.word2idx
            generated.append(w if known else f'[{w}?]') # marks unknown words by wrapping in brackets
            
        hidden = None

        with torch.no_grad():
            for _ in range(self.generate_length):
                x = torch.tensor([context], dtype=torch.long).to(device)
                logits, hidden = model(x, hidden) # get logits for next word prediction

                # apply temperature and sample from top-k
                last_logits = logits[0, -1] / self.temperature
                top_vals, top_idx = torch.topk(last_logits, self.top_k)
                probs = torch.softmax(top_vals, dim=0).cpu().numpy()
                choice = np.random.choice(top_idx.cpu().numpy(), p=probs)

                if choice == eos_idx:
                    break

                if choice == unk_idx:
                    # if UNK sampled, look in training data
                    current_phrase = ' '.join(generated).lower()
                    continuation = None
                    for text in self.data:
                        text_lower = text.lower()
                        idx = text_lower.find(current_phrase)
                        if idx != -1: # if found, get rest of phrase
                            rest = text_lower[idx + len(current_phrase):].strip()
                            rest_words = clean_text_gen(rest)
                            # filter out UNK words not in vocab
                            special_tokens = {'<UNK>', '<PAD>', '<EOS>'}
                            filtered = []
                            for w in rest_words: # filter out words not in vocab or special tokens
                                in_vocab = w in self.word2idx
                                special = w in special_tokens
                                if in_vocab and not special:
                                    filtered.append(w)
                            if rest_words:
                                continuation = rest_words
                                break

                    if continuation:
                        generated.extend(continuation)
                        break

                    # no training match — resample excluding <UNK> and <PAD>
                    last_logits[unk_idx] = -float('inf')
                    last_logits[pad_idx] = -float('inf')
                    top_vals, top_idx = torch.topk(last_logits, self.top_k)
                    probs = torch.softmax(top_vals, dim=0).cpu().numpy()
                    choice = np.random.choice(top_idx.cpu().numpy(), p=probs)

                    if choice == eos_idx:
                        break

                generated.append(self.idx2word[choice])
                context = context[1:] + [choice]

        return ' '.join(generated)

    def _plot_curve(self, loss, seed_words):
        nsw = len(seed_words)
        dir_path = self.save_curve_path.replace('generation_curve.png', '')
        base = f'{dir_path}{nsw}_generation_curve'
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(loss) + 1), loss)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss - ' + self.dataset_name)
        ax.grid(True)
        fig.tight_layout()
        loss_path = base + '_loss.png'
        fig.savefig(loss_path)
        plt.close(fig)

    def run(self):
        print('starting training on', self.dataset_name)

        texts = self.data
        print('number of texts:', len(texts))

        # tokenize
        token_lists = [clean_text_gen(t) for t in texts]
        self.word2idx, self.idx2word = self._build_vocab(token_lists)
        print('vocabulary size:', len(self.word2idx))

        # dataset and loader
        dataset = SequenceDataset(token_lists, self.word2idx, self.seq_len)
        print('total training sequences:', len(dataset))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # device check GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print('device:', device)

        # setup
        model = RNNLanguageModel(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # training
        losses = []
        for epoch in range(1, self.num_epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            for seqs, targets in loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits, _ = model(seqs)
                loss = criterion(logits[:, -1, :], targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * seqs.size(0)
                total += seqs.size(0)

            epoch_loss = total_loss / total
            losses.append(epoch_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch {epoch}/{self.num_epochs} - loss: {epoch_loss}')

        # generate from seed words
        generated = self._generate(model, self.seed_words, device)
        print('Generated text:', generated)
        print('\nSeed words:', self.seed_words)

        # compare with training data
        print('\nTraining samples containing seed phrase:')
        seed_str = ' '.join(self.seed_words).lower()
        matches = []
        for i in texts:
            if seed_str in i.lower():
                matches.append(i)
        if matches:
            for i in matches:
                print('-', i)
        else:
            print('no matches')

        # "Perplexity, a measure of the uncertainty or unpredictability of a language model,
        # plays a crucial role in the performance and evaluation of RNNs,
        # particularly in the realm of natural language processing."
        # https://canvas4everyone.com/blogs/news/unraveling-the-enigma-exploring-perplexity-in-recurrent-neural-networks

        # lower perplexity --> model better learned training text patterns --> more correct
        correctness = np.exp(losses[-1])
        print(f'\nCorrectness: {correctness}')
        print('Final training loss:', round(losses[-1], 4))

        txt_path = getattr(self, 'jokes_file', 'result/stage_4_result/generated_jokes.txt')
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f'Generated: {generated} Seed words: {self.seed_words}\n')

        self._plot_curve(losses, self.seed_words)

        return {
            'generated_text': generated,
            'seed_words': self.seed_words,
            'train_losses': losses,
            'correctness': correctness,
        }