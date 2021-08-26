import warnings

import nni
import math
import torch
import argparse
import pretty_errors
from torch.functional import split
from tqdm import tqdm

import GCL.utils.simple_param as SP

from time import time_ns
from torch.optim import Adam
from GCL.eval import SVM_classification, LR_classification
from GCL.utils import seed_everything
from sklearn.exceptions import ConvergenceWarning
from torch_geometric.data import DataLoader

from utils import get_activation, load_graph_dataset, get_compositional_augmentor
from models.BGRL import BGRL, Encoder


def test(model, loader, device, seed):
    model.eval()
    x = []
    y = []
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        g1, g2, _, _, _, _ = model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)
        # z = g2

        x.append(z.detach().cpu())
        y.append(data.y.cpu())

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    res = LR_classification(x.to(device), y.to(device), None, split_mode='rand', train_ratio=0.1, test_ratio=0.8)

    return res


def main():
    default_param = {
        'seed': 39788,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'proj_dim': 256,
        'weight_decay': 1e-5,
        'activation': 'prelu',
        'base_model': 'GINConv',
        'augmentor1:scheme': 'ER+FM',
        'augmentor2:scheme': 'ER+FM',
        'num_layers': 2,
        'patience': 100,
        'num_epochs': 1000,
        'batch_size': 32,
        'dropout': 0.2
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--dataset', type=str, default='IMDB-MULTI')
    parser.add_argument('--param_path', type=str, default='params/BGRL/imdb_multi.json')
    for k, v in default_param.items():
        if type(v) is dict:
            for subk, subv in v.items():
                parser.add_argument(f'--{k}:{subk}', type=type(subv), nargs='?')
        else:
            parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    sp.update(args.param_path, preprocess_nni=False)
    overwrite_params = {k: v for k, v in args.__dict__.items() if v is not None}
    sp.load(overwrite_params)
    param = sp()

    use_nni = args.param_path == 'nni'

    seed_everything(param['seed'])
    device = torch.device(args.device if not use_nni else 'cuda')
    dataset = load_graph_dataset('datasets', args.dataset)
    input_dim = dataset.num_features if dataset.num_features > 0 else 1
    train_loader = DataLoader(dataset, batch_size=param['batch_size'])
    test_loader = DataLoader(dataset, batch_size=math.ceil(param['batch_size']))

    print(dataset)
    print(dataset[0])

    print(param)
    print(args.__dict__)

    aug1 = get_compositional_augmentor(param['augmentor1'])
    aug2 = get_compositional_augmentor(param['augmentor2'])

    model = BGRL(encoder=Encoder(input_dim, param['hidden_dim'],
                                 activation=get_activation(param['activation']),
                                 num_layers=param['num_layers'],
                                 dropout=param['dropout'],
                                 encoder_norm=param['bootstrap']['encoder_norm'],
                                 projector_norm=param['bootstrap']['projector_norm'],
                                 base_conv='GINConv'),
                 augmentor=(aug1, aug2),
                 hidden_dim=param['hidden_dim'],
                 dropout=param['dropout'],
                 predictor_norm=param['bootstrap']['predictor_norm'],
                 mode='G2G').to(device)

    print('\n=== Final ===')
    test_result = test(model, test_loader, device, param['seed'])
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'])


if __name__ == '__main__':
    main()
