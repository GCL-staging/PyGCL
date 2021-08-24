import os
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
    config.opt.patience = 200
    config.opt.reduce_lr_patience = 100
    config.augmentor1.scheme = 'ND+FM'
    config.augmentor2.scheme = 'ND+FM'

    printer = PrettyPrinter(indent=2)
    printer.pprint(asdict(config))
    print('gpu:' + os.getenv('CUDA_VISIBLE_DEVICES'))

    trial = GCLTrial(config)
    result = trial.execute()

    print("=== Final ===")
    print(f'(T): Best epoch={trial.best_epoch}, best loss={trial.best_loss:.4f}')
    print(f'(E): {result}')

    nni.report_final_result(result['micro_f1'])
