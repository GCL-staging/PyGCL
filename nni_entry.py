import os
import numpy as np
from dataclasses import asdict
from pprint import PrettyPrinter
import nni
from HC import ConfigLoader
from train_config import *
from trial import GCLTrial


if __name__ == '__main__':
    import pretty_errors  # noqa
    loader = ConfigLoader(model=ExpConfig, config='nni')
    config: ExpConfig = loader()
    config.device = 'cuda'
    config.obj.loss = Objective.InfoNCE
    config.encoder.conv = ConvType.GINConv
    config.opt.batch_size = 200
    config.opt.patience = 500
    config.opt.reduce_lr_patience = -1  # let's disable LR scheduling for now
    config.augmentor1.scheme = 'ND+FM'
    config.augmentor2.scheme = 'ND+FM'
    num_repeats = 5

    def calc_mean_std(xs):
        if len(xs) == 0:
            return dict()
        keys = xs[0].keys()
        res = dict()

        for key in keys:
            values = [x[key] for x in xs]
            mean = np.mean(values)
            std = np.std(values)
            res[key] = {'mean': mean, 'std': std}

        return res

    printer = PrettyPrinter(indent=2)
    printer.pprint(asdict(config))
    print('gpu:' + os.getenv('CUDA_VISIBLE_DEVICES'))

    results = []
    for i in range(num_repeats):
        trial = GCLTrial(config)
        result = trial.execute()

        print("=== Final ===")
        print(f'(T): Best epoch={trial.best_epoch}, best loss={trial.best_loss:.4f}')
        print(f'(E): {result}')

        results.append(result)
    result = calc_mean_std(results)

    nni.report_final_result(result['micro_f1']['mean'])
    print(f'final: {result}')
