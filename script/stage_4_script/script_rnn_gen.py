### TO RUN SCRIPT:
# python3 script/stage_4_script/script_rnn_gen.py

import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_4_code.Method_RNN_gen import Method_RNN_Generation

DATA_FILE = 'data/stage_4_data/text_generation/data'
RESULT_DIR = 'result/stage_4_result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def load_jokes(filepath):
    jokes = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            joke = row.get('Joke', '').strip()
            if joke:
                jokes.append(joke)
    return jokes


print(f'\nLoading {DATA_FILE} data…')
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
method_obj.num_epochs = 100
method_obj.lr = 1e-3
method_obj.seed_words = ['what', 'did', 'the']
# method_obj.seed_words = ['what', 'did']
# method_obj.seed_words = ['what']
# method_obj.seed_words = ['what', 'did', 'the', 'fish']
method_obj.generate_length = 25
method_obj.temperature = 0.8
method_obj.top_k = 10

method_obj.data = jokes

base = 'result/stage_4_result/generated_jokes'
JOKES_FILE = base + '.txt'
i = 1
while os.path.exists(JOKES_FILE):
    JOKES_FILE = f'{base}{i}.txt'
    i += 1
method_obj.jokes_file = JOKES_FILE
print(f'saving jokes to {JOKES_FILE}')

all_seed_words = [
    ['what'],
    ['what', 'did'],
    ['what', 'did', 'the'],
    ['what', 'did', 'the', 'chicken'],
]

def run_all():
    for seed_words in all_seed_words:
        print(f'\n> Running seed: {seed_words}')
        method_obj.seed_words = seed_words
        method_obj.run()

# individual run
# method_obj.run()

# run all seed words
run_all()