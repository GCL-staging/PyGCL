from trial import GCLTrial
from train_config import *
from HC.config_loader import ConfigLoader

from ray import tune

learning_rate = [0.01, 0.001, 0.0001]
weight_decay = [0.00001]
num_epochs = [100, 200, 500]
objective = ['infonce', 'jsd', 'triplet']
mode = ['L2L', 'G2L', 'G2G']
base_config = '/home/xuyichen/dev/PyGCL/params/nci1.json'
dataset = 'NCI1'


def run_trial(tune_config):
    loader = ConfigLoader(model=ExpConfig, config=base_config, disable_argparse=True)
    config: ExpConfig = loader()
    config.device = 'cuda'
    config.dataset = dataset
    config.opt.learning_rate = tune_config['learning_rate']
    config.opt.weight_decay = tune_config['weight_decay']
    config.opt.num_epochs = tune_config['num_epochs']
    config.obj.loss = Objective(tune_config['objective'])
    config.mode = ContrastMode(tune_config['mode'])
    trial = GCLTrial(config)

    # def report_loss(data):
    #     tune.report(loss=data['loss'])

    # trial.register_train_step_callback(report_loss)
    result = trial.execute()

    tune.report(loss=trial.best_loss, test_acc=result['micro_f1'])


if __name__ == '__main__':
    space = {
        'learning_rate': tune.grid_search(learning_rate),
        'weight_decay': tune.grid_search(weight_decay),
        'num_epochs': tune.grid_search(num_epochs),
        'objective': tune.grid_search(objective),
        'mode': tune.grid_search(mode),
    }

    result = \
        tune.run(
            run_trial,
            config=space, metric='test_acc', mode='max', verbose=2,
            resources_per_trial={'gpu': 1}, num_samples=3
        )
