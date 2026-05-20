import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_4_code.Method_RNN_gen import Method_RNN_Generation

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stage_4_data', 'text_generation', 'data')


def load_jokes(filepath):
    jokes = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            joke = row.get('Joke', '').strip()
            if joke:
                jokes.append(joke)
    return jokes


print('Loading jokes data…')
jokes = load_jokes(DATA_FILE)
print(f'Loaded {len(jokes)} jokes')

method_obj = Method_RNN_Generation()

# HYPERPARAMETERS
method_obj.max_vocab_size = 5000
method_obj.seq_len = 15
method_obj.embed_dim = 128
method_obj.hidden_dim = 256
method_obj.num_layers = 2
method_obj.dropout = 0.3
method_obj.batch_size = 128
method_obj.num_epochs = 80
method_obj.lr = 1e-3
method_obj.seed_words = ['what', 'did', 'the']
method_obj.generate_length = 25
method_obj.temperature = 0.8
method_obj.top_k = 10

method_obj.data = jokes
method_obj.run()
