import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_code.stage_3_code.Method_CNN import Method_CNN

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stage_3_data')


def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded '{filename}': {len(data['train'])} train, {len(data['test'])} test instances")
    return data


# ── MNIST ─────────────────────────────────────────────────────────────────────
if 1:
    method_obj = Method_CNN('CNN-MNIST', '')
    method_obj.dataset_name          = 'MNIST Digits'
    method_obj.in_channels           = 1
    method_obj.num_classes           = 10
    method_obj.channel_mode          = 'single'
    method_obj.batch_size            = 128
    method_obj.num_epochs            = 20
    method_obj.lr                    = 1e-3
    method_obj.save_curve_path       = 'MNIST_curves.png'
    method_obj.save_convergence_path = 'MNIST_convergence.png'
    method_obj.data                  = load_data('MNIST')
    method_obj.run()


# ── ORL Faces ─────────────────────────────────────────────────────────────────
if 1:
    method_obj = Method_CNN('CNN-ORL', '')
    method_obj.dataset_name          = 'ORL Faces'
    method_obj.in_channels           = 1
    method_obj.num_classes           = 40
    method_obj.channel_mode          = 'single'
    method_obj.batch_size            = 8
    method_obj.num_epochs            = 40
    method_obj.lr                    = 5e-4
    method_obj.label_offset          = -1
    method_obj.save_curve_path       = 'ORL_curves.png'
    method_obj.save_convergence_path = 'ORL_convergence.png'
    method_obj.data                  = load_data('ORL')
    method_obj.run()


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
if 1:
    method_obj = Method_CNN('CNN-CIFAR', '')
    method_obj.dataset_name          = 'CIFAR-10 Objects'
    method_obj.in_channels           = 3
    method_obj.num_classes           = 10
    method_obj.channel_mode          = 'multi'
    method_obj.batch_size            = 128
    method_obj.num_epochs            = 90
    method_obj.lr                    = 1e-3
    method_obj.save_curve_path       = 'CIFAR_curves.png'
    method_obj.save_convergence_path = 'CIFAR_convergence.png'
    method_obj.data                  = load_data('CIFAR')
    method_obj.run()
