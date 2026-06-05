### TO RUN SCRIPT:
# python3 script/stage_5_script/script_gcn.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Method_GCN_improved import Method_GCN_Improved

# TOGGLE IMPROVED HERE
IMPROVED = False   # False = baseline, True = improved

DATASETS = ['cora', 'citeseer', 'pubmed']


RESULT_DIR = 'result/stage_5_result'
os.makedirs(RESULT_DIR, exist_ok=True)

all_results = {}

for dataset_name in DATASETS:
    print(f'Dataset: {dataset_name}  |  model: {"improved" if IMPROVED else "baseline"}')

    dataset_loader = Dataset_Loader(dName=dataset_name)
    dataset_loader.dataset_source_folder_path = f'data/stage_5_data/{dataset_name}'
    dataset_loader.dataset_name = dataset_name
    data = dataset_loader.load()

    if IMPROVED:
        method_obj = Method_GCN_Improved()
        method_obj.hidden_dim = 128
    else:
        method_obj = Method_GCN()
        method_obj.hidden_dim = 64
        method_obj.dropout = 0.5
        method_obj.lr = 0.01
        method_obj.weight_decay = 5e-4
        method_obj.num_epochs = 500

    method_obj.dataset_name = dataset_name
    method_obj.data = data

    results = method_obj.run()
    all_results[dataset_name] = results['final_acc']

print('\nSummary')
for name, acc in all_results.items():
    print(f'{name}  test accuracy: {acc}')