import os
from trial import GCLTrial
from train_config import ExpConfig
from HC.config_loader import ConfigLoader

from ray import tune


def run_trial(ray_config):
    loader = ConfigLoader(model=ExpConfig, config='/home/xuyichen/dev/PyGCL/params/GRACE/wikics@ng.json', disable_argparse=True)
    config: ExpConfig = loader()
    config.augmentor2.drop_feat_prob = ray_config['drop_feat_prob']
    config.augmentor2.drop_edge_prob = ray_config['drop_feat_prob']
    config.device = 'cuda'
    trial = GCLTrial(config)

    def report_loss(data):
        tune.report(loss=data['loss'])

    trial.register_train_step_callback(report_loss)
    result = trial.execute()

    tune.report(loss=trial.best_loss, test_acc=result['micro_f1'])


if __name__ == '__main__':
    space = {
        'drop_feat_prob': tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        'drop_edge_prob': tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    }

    result = tune.run(run_trial, config=space, metric='loss', mode='min', verbose=2, resources_per_trial={'gpu': 1})
