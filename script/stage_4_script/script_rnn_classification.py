### TO RUN SCRIPT:
# python3 script/stage_4_script/script_rnn_classification.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_4_code.Method_RNN import Method_RNN, RNN, RNN1, RNN2, RNN3

DATA_DIR = 'data/stage_4_data/text_classification'
RESULT_DIR = 'result/stage_4_result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
path = 'result/stage_4_result/results'
RESULTS_FILE = path + '.txt'
i = 1
while os.path.exists(RESULTS_FILE):
    RESULTS_FILE = f'{path}{i}.txt'
    i += 1

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


print(f'\nLoading {DATA_DIR} data…')
train_data = load_split('train')
test_data  = load_split('test')
print(f'Train: {len(train_data)}, Test: {len(test_data)}')

method_obj = Method_RNN()

# MODEL: RNN, RNN1, RNN2, RNN3
method_obj.model_class = RNN1

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
method_obj.num_epochs = 20
method_obj.lr = 1e-3

method_obj.data = {'train': train_data, 'test': test_data}

def run_all():
    for model_c in [RNN, RNN1, RNN2, RNN3]:
        print(f'\n> Running {model_c.__name__}')
        method_obj.model_class = model_c
        method_obj.run()

# individual run
method_obj.run()

# run all models
# run_all()