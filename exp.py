import argparse
from typing import *
from time import time_ns, perf_counter
import os
import os.path as osp
from copy import deepcopy
from filelock import FileLock

import numpy as np
import torch
import ray

from train_config import *
from trial import GCLTrial
from HC import ConfigLoader

_EXP_DICT = dict()


@ray.remote(num_gpus=1)
def run_trial(idx: int, total: int, config: ExpConfig, num_repeats: int = 1):
    cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    print(f'>>> Trial {idx} / {total} <<<')
    print(f'cuda: {cuda_devices}')
    config.show()

    try:
        results = []
        durations = []
        for _ in range(num_repeats):
            tic = perf_counter()
            trial = GCLTrial(config, mute_pbar=True)
            result = trial.execute()
            results.append(result)
            toc = perf_counter()
            durations.append(toc - tic)

        result = dict()
        metric_keys = results[0].keys()
        for metric in metric_keys:
            values = [x[metric] for x in results]
            mean = np.mean(values)
            std = np.std(values)
            result[metric] = {'mean': mean, 'std': std}
    except Exception as e:
        print(f'>>> Trial {idx} / {total} <<<')
        print(f'!!! catched error: {e}')
        return {'config': config, 'result': None, 'error': str(e)}

    print(f'>>> Trial {idx} / {total} <<<')
    print(f'result: {result}')
    print(f'durations: {durations}')

    res = {'config': config, 'result': result, 'idx': idx}

    with FileLock(result_lock_path):
        if osp.exists(result_path):
            existing = torch.load(result_path)
        else:
            existing = []
        existing.append(res)
        torch.save(existing, result_path)

    return res


def register(func):
    _EXP_DICT[func.__name__] = func
    return func


def load_config(path: str, **kwargs) -> ExpConfig:
    loader = ConfigLoader(ExpConfig, config=path, disable_argparse=True)
    return loader(**kwargs)


@register
def nci1_g2l_g2g_e2():
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/nci1@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/nci1@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [100, 200, 500, 1000, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(3):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        for obj in objective:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def imdbm_all_modes_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [100, 200, 500, 1000, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(3):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        for obj in objective:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def imdbm_all_modes_extra_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [100, 200, 500, 1000, 1200, 2000]
    objective = [Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(3):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        for obj in objective:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2g_config)]
    return generated


@register
def nci1_g2g_e2():
    # l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@l2l.json', after_args={'device': 'cuda'})
    # g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/nci1@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-5, 1e-7]
    num_epochs = [500, 1000, 1200]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet, Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for lr in learning_rate:
            for wd in weight_decay:
                for e in num_epochs:
                    for obj in objective:
                        config = deepcopy(base_cfg)
                        config.opt.learning_rate = lr
                        config.opt.weight_decay = wd
                        config.opt.num_epochs = e
                        config.obj.loss = obj
                        res.append(config)
        return res

    generated = [*generate_config(g2g_config)]
    return generated


@register
def proteins_g2g_e2():
    # l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@l2l.json', after_args={'device': 'cuda'})
    # g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-5, 1e-7]
    num_epochs = [500, 1000, 1200]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet, Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(3):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        for obj in objective:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(g2g_config)]
    return generated


@register
def proteins_all_modes_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [100, 200, 500, 1000, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(3):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        for obj in objective:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def proteins_all_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/proteins@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [300, 500, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]
    extra_objective = objective + [Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for lr in learning_rate:
            for wd in weight_decay:
                for e in num_epochs:
                    obj_list = extra_objective if base_cfg.mode != ContrastMode.G2L else objective
                    for obj in obj_list:
                        config = deepcopy(base_cfg)
                        config.opt.learning_rate = lr
                        config.opt.weight_decay = wd
                        config.opt.num_epochs = e
                        config.obj.loss = obj
                        res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def imdbm_all_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/imdb_multi@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [300, 500, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]
    extra_objective = objective + [Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for lr in learning_rate:
            for wd in weight_decay:
                for e in num_epochs:
                    obj_list = extra_objective if base_cfg.mode != ContrastMode.G2L else objective
                    for obj in obj_list:
                        config = deepcopy(base_cfg)
                        config.opt.learning_rate = lr
                        config.opt.weight_decay = wd
                        config.opt.num_epochs = e
                        config.obj.loss = obj
                        res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def collab_all_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.01, 0.001, 0.0001]
    weight_decay = [1e-4, 1e-5, 1e-6]
    num_epochs = [300, 500, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]
    extra_objective = objective + [Objective.BarlowTwins, Objective.VICReg]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(1):
            for lr in learning_rate:
                for wd in weight_decay:
                    for e in num_epochs:
                        obj_list = extra_objective if base_cfg.mode != ContrastMode.G2L else objective
                        for obj in obj_list:
                            config = deepcopy(base_cfg)
                            config.opt.learning_rate = lr
                            config.opt.weight_decay = wd
                            config.opt.num_epochs = e
                            config.obj.loss = obj
                            res.append(config)
        return res

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
    return generated


@register
def topo_aug_interaction():
    base_config = load_config(
        '/home/xuyichen/dev/PyGCL/params/general.json',
        after_args={'device': 'cuda', 'mode': 'L2L', 'dataset': 'WikiCS', 'obj:loss': 'infonce'}
    )

    topo_aug_list = ['ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR+ND', 'MKD+ND', 'RWS']

    res = []
    for aug1_scheme in topo_aug_list:
        for aug2_scheme in topo_aug_list:
            config = deepcopy(base_config)
            config.augmentor1.scheme = aug1_scheme
            config.augmentor2.scheme = aug2_scheme
            res.append(config)

    return res


@register
def aug_ablation():
    aug_list = [
        'ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR', 'MKD', 'RWS', 'ER+FM', 'ER+FD', 'ND+FM', 'ND+FD', 'EA+FM', 'EA+FD',
        'FM', 'FD',
        'RWS+FM', 'RWS+FD', 'PPR+ER', 'PPR+FD', 'PPR+ND', 'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}@l2l.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['nci1', 'proteins', 'imdb_multi']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


@register
def imdbm_all_e1():
    aug_list = [
        'ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR', 'MKD', 'RWS', 'ER+FM', 'ER+FD', 'ND+FM', 'ND+FD', 'EA+FM', 'EA+FD',
        'FM', 'FD',
        'RWS+FM', 'RWS+FD', 'PPR+ER', 'PPR+FD', 'PPR+ND', 'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['imdb_multi']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


@register
def aug_ablation():
    aug_list = [
        'ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR', 'MKD', 'RWS', 'ER+FM', 'ER+FD', 'ND+FM', 'ND+FD', 'EA+FM', 'EA+FD',
        'FM', 'FD'
        'RWS+FM', 'RWS+FD', 'PPR+ER', 'PPR+FD', 'PPR+ND', 'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}@l2l.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['nci1', 'proteins', 'imdb_multi']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


@register
def nci1_all_e1():
    aug_list = [
        'ORI',
        'EA', 'ER', 'EA+ER', 'ND',
        'PPR',
        'MKD',
        'RWS',
        'ER+FM', 'ER+FD',
        'ND+FM', 'ND+FD',
        'EA+FM', 'EA+FD',
        'FM', 'FD',
        'RWS+FM', 'RWS+FD',
        'PPR+ER', 'PPR+FD', 'PPR+ND',
        'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['nci1']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


@register
def proteins_all_e1():
    aug_list = [
        'ORI',
        'EA', 'ER', 'EA+ER', 'ND',
        'PPR',
        'MKD',
        'RWS',
        'ER+FM', 'ER+FD',
        'ND+FM', 'ND+FD',
        'EA+FM', 'EA+FD',
        'FM', 'FD',
        'RWS+FM', 'RWS+FD',
        'PPR+ER', 'PPR+FD', 'PPR+ND',
        'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['proteins']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


@register
def cs_ER_sensitivity():
    probs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8]
    base_cfg: ExpConfig = load_config('/home/xuyichen/dev/PyGCL/params/coauthor_cs.json', after_args={'device': 'cuda'})
    base_cfg.augmentor1.scheme = 'ER'
    base_cfg.augmentor2.scheme = 'ER'

    res = []
    for prob in probs:
        cfg = deepcopy(base_cfg)
        cfg.augmentor1.ER.prob = prob
        cfg.augmentor2.ER.prob = prob
        res.append(cfg)

    return res


@register
def cs_ND_sensitivity():
    probs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8]
    base_cfg: ExpConfig = load_config('/home/xuyichen/dev/PyGCL/params/coauthor_cs.json', after_args={'device': 'cuda'})
    base_cfg.augmentor1.scheme = 'ND'
    base_cfg.augmentor2.scheme = 'ND'

    res = []
    for prob in probs:
        cfg = deepcopy(base_cfg)
        cfg.augmentor1.ND.prob = prob
        cfg.augmentor2.ND.prob = prob
        res.append(cfg)

    return res


@register
def cs_EA_sensitivity():
    probs = [0.01, 0.05, 0.1, 0.2, 0.5]
    base_cfg: ExpConfig = load_config('/home/xuyichen/dev/PyGCL/params/coauthor_cs.json', after_args={'device': 'cuda'})
    base_cfg.augmentor1.scheme = 'EA'
    base_cfg.augmentor2.scheme = 'EA'

    res = []
    for prob in probs:
        cfg = deepcopy(base_cfg)
        cfg.augmentor1.EA.prob = prob
        cfg.augmentor2.EA.prob = prob
        res.append(cfg)

    return res


@register
def collab_all_e1():
    aug_list = [
        'ORI',
        'EA', 'ER', 'EA+ER', 'ND',
        'PPR',
        # 'MKD',
        'RWS',
        'ER+FM', 'ER+FD',
        'ND+FM', 'ND+FD',
        'EA+FM', 'EA+FD',
        'FM', 'FD',
        'RWS+FM', 'RWS+FD',
        'PPR+ER', 'PPR+FD', 'PPR+ND',
        # 'MKD+ER', 'MKD+FD', 'MKD+ND'
    ]

    def gen(base_cfg: ExpConfig):
        res = []
        for scheme in aug_list:
            func = ExpConfig._augmentor1._scheme.set(scheme) @ ExpConfig._augmentor2._scheme.set(scheme)
            res.append(func(base_cfg))
        return res

    def gen_dataset(dataset: str):
        path = f'/home/xuyichen/dev/PyGCL/params/{dataset}@best.json'
        cfg = load_config(path, after_args={'device': 'cuda'})
        return gen(cfg)

    dataset_list = ['collab']
    configs = []
    for dataset in dataset_list:
        configs = configs + gen_dataset(dataset)

    return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help='Name of the experiment.')
    parser.add_argument('--port', '-p', type=int, default=10000, help='Dashboard port.')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='Number of times to repeat the experiment.')
    args = parser.parse_args()
    exp_name = args.name
    ray.init(dashboard_host='0.0.0.0', dashboard_port=args.port, num_cpus=8)

    result_dir = osp.expanduser(f'~/exp_results/{exp_name}')
    os.makedirs(result_dir, exist_ok=True)

    result_path = osp.join(result_dir, 'result.pkl')
    final_path = osp.join(result_dir, 'final.pkl')
    result_lock_path = osp.join(result_dir, 'result.pkl.lock')

    config_generator = _EXP_DICT[exp_name]
    config_list = config_generator()
    total = len(config_list)

    if osp.exists(result_path):
        print('found intermediate results')
        results = torch.load(result_path)
        finished_ids = [res['idx'] for res in results]
        print(f'finished ids: {finished_ids}')
        finished_ids = set(finished_ids)
    else:
        results = []
        finished_ids = set()

    futures = [run_trial.remote(idx, total, cfg, args.repeat) for idx, cfg in enumerate(config_list) if idx not in finished_ids]
    results = ray.get(futures) + results

    torch.save(results, final_path)
