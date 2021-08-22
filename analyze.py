import argparse
from typing import *
from tabulate import tabulate
import train_config
from train_config import *
import torch

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
    data = group_by(lambda x: x['config'].mode, results)
    data = map_dict(lambda x: group_by(lambda y: y['config'].obj.loss, x), data)
    data = map_dict(lambda a: map_dict(lambda b: max(b, key=lambda c: c['result']['micro_f1']), a), data)

    acc_table = []
    for obj in [Objective.InfoNCE, Objective.JSD, Objective.Triplet]:
        row = [obj.value]
        for mode in [ContrastMode.L2L, ContrastMode.G2L, ContrastMode.G2G]:
            row.append(data[mode][obj]['result']['micro_f1'])
        acc_table.append(row)

    print(tabulate(acc_table, headers=['Objective', 'L2L', 'G2L', 'G2G']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input experiement data.')
    parser.add_argument('--name', '-n', type=str, help='Name of the experiement.')
    args = parser.parse_args()

    results = torch.load(args.input)
    analyzer = _ANALYZER_DICT[args.name]

    analyzer(results)
