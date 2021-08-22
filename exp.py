import argparse
from typing import *
from time import time_ns
import os
from copy import deepcopy

import torch
import ray
from tqdm import tqdm

from train_config import *
from trial import GCLTrial
from HC import ConfigLoader

_EXP_DICT = dict()


@ray.remote(num_gpus=1)
def run_trial(idx: int, total: int, config: ExpConfig):
    cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    print(f'>>> Trial {idx} / {total} <<<')
    print(f'cuda: {cuda_devices}')
    config.show()

    trial = GCLTrial(config, mute_pbar=True)
    result = trial.execute()

    print(f'>>> Trial {idx} / {total} <<<')
    print(f'result: {result}')

    return {'config': config, 'result': result}


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

    generated = [*generate_config(l2l_config), *generate_config(g2l_config), *generate_config(g2g_config)]
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
def collab_all_modes_e2():
    l2l_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@l2l.json', after_args={'device': 'cuda'})
    g2l_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@g2l.json', after_args={'device': 'cuda'})
    g2g_config = load_config('/home/xuyichen/dev/PyGCL/params/collab@g2g.json', after_args={'device': 'cuda'})

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    weight_decay = [1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs = [100, 200, 500, 1000, 1200, 2000]
    objective = [Objective.InfoNCE, Objective.JSD, Objective.Triplet]

    def generate_config(base_cfg: ExpConfig) -> List[ExpConfig]:
        res = []
        for _ in range(1):
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


if __name__ == '__main__':
    ray.init(dashboard_host='0.0.0.0', dashboard_port=10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help='Name of the experiement.')
    args = parser.parse_args()
    exp_name = args.name

    os.makedirs('./exp_results/', exist_ok=True)

    result_path = f'./exp_results/{exp_name}-{time_ns()}.pkl'

    config_generator = _EXP_DICT[exp_name]
    config_list = config_generator()
    total = len(config_list)

    futures = [run_trial.remote(idx, total, cfg) for idx, cfg in enumerate(config_list)]
    results = ray.get(futures)

    torch.save(results, result_path)
