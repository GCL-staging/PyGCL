import argparse
from typing import *
from tabulate import tabulate
import train_config
from train_config import *
import torch
import numpy as np

_ANALYZER_DICT = dict()


def register(func):
    _ANALYZER_DICT[func.__name__] = func
    return func


def map_dict(func, d: Dict) -> Dict:
    return {k: func(v) for k, v in d.items()}


def group_with(func, xs):
    res = dict()
    for x in xs:
        k, v = func(x)
        if k in res:
            res[k].append(v)
        else:
            res[k] = [v]
    return res


def group_by(func, xs):
    func1 = lambda x: (func(x), x)
    return group_with(func1, xs)


@register
def g2l_g2g_e2(results):
    data = group_by(lambda x: x['config'].mode, results)
    data = map_dict(lambda x: group_by(lambda y: y['config'].obj.loss, x), data)
    data = map_dict(lambda a: map_dict(lambda b: max(b, key=lambda c: c['result']['micro_f1']), a), data)

    acc_table = []
    for obj in [Objective.InfoNCE, Objective.JSD, Objective.Triplet]:
        row = [obj.value]
        for mode in [ContrastMode.G2L, ContrastMode.G2G]:
            row.append(data[mode][obj]['result']['micro_f1'])
        acc_table.append(row)

    print(tabulate(acc_table, headers=['Objective', 'G2L', 'G2G']))


@register
def l2l_e2(results):
    data = group_by(lambda x: x['config'].mode, results)
    data = map_dict(lambda x: group_by(lambda y: y['config'].obj.loss, x), data)
    data = map_dict(lambda a: map_dict(lambda b: max(b, key=lambda c: c['result']['micro_f1']), a), data)

    acc_table = []
    for obj in [Objective.InfoNCE, Objective.JSD, Objective.Triplet]:
        row = [obj.value]
        for mode in [ContrastMode.L2L]:
            row.append(data[mode][obj]['result']['micro_f1'])
        acc_table.append(row)

    print(tabulate(acc_table, headers=['Objective', 'L2L', 'G2L', 'G2G']))


@register
def general_e2(results):
    for x in results:
        if 'error' in x:
            print(f"error in trial: {x['error']}")
    results = [x for x in results if 'error' not in x]
    data = group_by(lambda x: x['config'].mode, results)
    data = map_dict(lambda x: group_by(lambda y: y['config'].obj.loss, x), data)
    data = map_dict(lambda a: map_dict(lambda b: max(b, key=lambda c: c['result']['micro_f1']['mean']), a), data)

    acc_table = []
    settings_table = []
    for obj in [Objective.InfoNCE, Objective.JSD, Objective.Triplet, Objective.BarlowTwins, Objective.VICReg]:
        row = [obj.value]
        row1 = [obj.value]
        for mode in [ContrastMode.L2L, ContrastMode.G2L, ContrastMode.G2G]:
            if mode in data and obj in data[mode]:
                acc = data[mode][obj]['result']['micro_f1']['mean']
                std = data[mode][obj]['result']['micro_f1']['std']
                row.append(f'{acc * 100:.2f} +- {std * 100:.2f}')
                cfg: ExpConfig = data[mode][obj]['config']
                s = f'lr={cfg.opt.learning_rate}'
                s += f'\nwd={cfg.opt.weight_decay}'
                s += f'\nep={cfg.opt.num_epochs}'
                row1.append(s)
            else:
                row.append('---')
                row1.append('---')
        acc_table.append(row)
        settings_table.append(row1)

    print(tabulate(acc_table, headers=['Objective', 'L2L', 'G2L', 'G2G'], tablefmt='tsv'))
    print(tabulate(settings_table, headers=['Objective', 'L2L', 'G2L', 'G2G']))


@register
def extra_e2(results):
    for x in results:
        if 'error' in x:
            print(f"error in trial: {x['error']}")
    results = [x for x in results if 'error' not in x]
    data = group_by(lambda x: x['config'].mode, results)
    data = map_dict(lambda x: group_by(lambda y: y['config'].obj.loss, x), data)
    data = map_dict(lambda a: map_dict(lambda b: max(b, key=lambda c: c['result']['micro_f1']['mean']), a), data)

    acc_table = []
    for obj in [Objective.BarlowTwins, Objective.VICReg]:
        row = [obj.value]
        for mode in [ContrastMode.L2L, ContrastMode.G2G]:
            row.append(data[mode][obj]['result']['micro_f1'])
        acc_table.append(row)

    print(tabulate(acc_table, headers=['Objective', 'L2L', 'G2G']))


@register
def prob_sensitivity(results):
    for x in results:
        if 'error' in x:
            print(f"error in trial {x['config'].augmentor1.scheme}: {x['error']}")

        print(x['config'].augmentor1.EA.prob)

    data = group_with(lambda x: (x['config'].augmentor1.EA.prob, x['result']), results)

    prob_list = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8]

    table = []
    for prob in prob_list:
        row = [prob]
        if prob in data:
            acc = data[prob][0]['micro_f1']
            txt = f'{acc["mean"]:.6f} +- {acc["std"]:.6f}'
            row.append(txt)
        else:
            row.append('---')
        table.append(row)

    print(tabulate(table, headers=['Prob', 'Acc']))


@register
def general_e1(results):
    for x in results:
        if 'error' in x:
            print(f"error in trial {x['config'].augmentor1.scheme}: {x['error']}")
    results = [x for x in results if 'error' not in x]
    data = group_with(lambda x: ((x['config'].dataset, x['config'].augmentor1.scheme), x['result']), results)

    # def calc_mean_std(xs):
    #     if len(xs) == 0:
    #         return dict()
    #     keys = xs[0].keys()
    #     res = dict()
    #
    #     for key in keys:
    #         values = [x[key] for x in xs]
    #         mean = np.mean(values)
    #         std = np.std(values)
    #         res[key] = {'mean': mean, 'std': std}
    #
    #     return res
    #
    # data = {k: calc_mean_std(v) for k, v in data.items()}

    print(data.keys())

    dataset_list = ['NCI1', 'PROTEINS', 'IMDB-MULTI', 'COLLAB']
    aug_list = [
        'ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR', 'MKD', 'RWS',
        'FM', 'FD',
        'ER+FM', 'ER+FD',
        'ND+FM', 'ND+FD',
        'EA+FM', 'EA+FD',
        'RWS+FM', 'RWS+FD',
        'PPR+ER', 'PPR+FD', 'PPR+EA', 'PPR+ND',
        'MKD+ER', 'MKD+FD', 'MKD+EA', 'MKD+ND',
    ]

    table = []
    for aug in aug_list:
        row = []
        row.append(aug)
        for dataset in dataset_list:
            symp = (dataset, aug)
            if symp in data:
                acc = data[symp][0]['micro_f1']['mean']
                std = data[symp][0]['micro_f1']['std']
                row.append(f'{acc * 100:.2f} +- {std * 100:.2f}')
            else:
                row.append('---')
        table.append(row)

    print(tabulate(table, headers=dataset_list, tablefmt='tsv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input experiement data.')
    parser.add_argument('--name', '-n', type=str, help='Name of the experiement.')
    args = parser.parse_args()

    results = torch.load(args.input)
    analyzer = _ANALYZER_DICT[args.name]

    analyzer(results)
