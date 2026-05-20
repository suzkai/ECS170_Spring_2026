import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_4_code.Method_RNN import Method_RNN

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stage_4_data', 'text_classification')


def load_split(split):
    pairs = []
    for label_str, label_int in [('pos', 1), ('neg', 0)]:
        folder = os.path.join(DATA_DIR, split, label_str)
        for fname in os.listdir(folder):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(folder, fname), 'r', encoding='utf-8', errors='replace') as f:
                pairs.append((f.read().strip(), label_int))
    return pairs


print('Loading IMDB data…')
train_data = load_split('train')
test_data  = load_split('test')
print(f'Train: {len(train_data)}, Test: {len(test_data)}')

method_obj = Method_RNN()

# HYPERPARAMETERS
method_obj.max_vocab_size = 20000
method_obj.max_seq_len = 200
method_obj.embed_dim = 128
method_obj.hidden_dim = 256
method_obj.num_layers = 2
method_obj.dropout = 0.5
method_obj.rnn_type = 'lstm'
method_obj.bidirectional = True
method_obj.num_classes = 2
method_obj.batch_size = 64
method_obj.num_epochs = 10
method_obj.lr = 1e-3

method_obj.data = {'train': train_data, 'test': test_data}
method_obj.run()
