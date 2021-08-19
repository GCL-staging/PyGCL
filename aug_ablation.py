import os
from trial import GCLTrial
from train_config import ExpConfig
from HC.config_loader import ConfigLoader

from ray import tune


def run_trial(tune_config):
    loader = ConfigLoader(model=ExpConfig, config='/home/xuyichen/dev/PyGCL/params/GRACE/general.json', disable_argparse=True)
    config: ExpConfig = loader()
    config.device = 'cuda'
    config.dataset = tune_config['dataset']
    config.augmentor1.scheme = tune_config['aug']
    config.augmentor2.scheme = tune_config['aug']
    trial = GCLTrial(config)

    # def report_loss(data):
    #     tune.report(loss=data['loss'])

    # trial.register_train_step_callback(report_loss)
    result = trial.execute()

    tune.report(loss=trial.best_loss, test_acc=result['micro_f1'])


if __name__ == '__main__':
    aug_list = [
        'ORI', 'EA', 'ER', 'EA+ER', 'ND', 'PPR', 'MKD', 'RWS', 'ER+FM', 'ER+FD', 'ND+FM', 'ND+FD', 'EA+FM', 'EA+FD',
        'RWS+FM', 'RWS+FD', 'PPR+ER', 'PPR+FD', 'PPR+EA', 'PPR+ND', 'MKD+ER', 'MKD+FD', 'MKD+EA', 'MKD+ND'
    ]
    # datasets = ['WikiCS', 'Coauthor-CS', 'Amazon-Computers', 'PTC_MR', 'PROTEINS', 'IMDB-MULTI']
    datasets = ['WikiCS']
    space = {
        'aug': tune.grid_search(aug_list),
        'dataset': tune.grid_search(datasets)
    }

    result = \
        tune.run(
            run_trial,
            name='wiki-aug',
            config=space, metric='test_acc', mode='max', verbose=2,
            resources_per_trial={'gpu': 1}, num_samples=3
        )
