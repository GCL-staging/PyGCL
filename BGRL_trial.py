from itertools import chain
from typing import *
import os
import os.path as osp
import secrets
from time import time_ns
from tqdm import tqdm
import torch
from visualdl import LogWriter

from GCL.eval import LREvaluator, get_split, GeneralLREvaluator
import GCL.losses as L
from GCL.utils import seed_everything, batchify_dict
from GCL.models import BootstrapContrast

from HC.config_loader import ConfigLoader

from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool

from ogb.graphproppred import Evaluator

from utils import load_dataset, get_activation, get_loss, is_node_dataset, get_augmentor
from models.BGRL import EncoderModel, GConv
from train_config import *


class BGRLTrial(object):
    def __init__(self, config: ExpConfig, mute_pbar: bool = False):
        super(BGRLTrial, self).__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.writer = LogWriter(logdir=f'./log/{config.visualdl}/train')
        self.dataset = load_dataset('datasets', config.dataset, to_sparse_tensor=False)
        self.train_loader = DataLoader(self.dataset, batch_size=config.opt.batch_size)
        self.test_loader = DataLoader(self.dataset, batch_size=config.opt.batch_size, shuffle=False)
        self.mute_pbar = mute_pbar

        input_dim = 1 if self.dataset.num_features == 0 else self.dataset.num_features

        def augmentor_from_conf(conf: AugmentorConfig):
            scheme = conf.scheme.split('+')

            augs = [get_augmentor(aug_name, asdict(conf)[aug_name]) for aug_name in scheme]

            aug = augs[0]
            for a in augs[1:]:
                aug = aug >> a
            return aug

        aug1 = augmentor_from_conf(config.augmentor1)
        aug2 = augmentor_from_conf(config.augmentor2)

        is_molecole_dataset = config.dataset in {'ogbg-molhiv'}
        encoder = GConv(
            input_dim=input_dim,
            hidden_dim=config.encoder.hidden_dim,
            num_layers=config.encoder.num_layers,
            use_atom_encoder=is_molecole_dataset,
            use_bond_encoder=is_molecole_dataset,
        ).to(self.device)
        self.encoder = encoder

        encoder_model = EncoderModel(
            encoder=encoder, augmentor=(aug1, aug2), hidden_dim=config.encoder.hidden_dim
        ).to(self.device)
        self.encoder_model = encoder_model

        contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode=config.mode.value).to(self.device)

        self.contrast_model = contrast_model

        optimizer = torch.optim.Adam(
            chain(encoder_model.parameters(), contrast_model.parameters()),
            lr=config.opt.learning_rate
        )
        if config.opt.reduce_lr_patience > 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.opt.reduce_lr_patience)
        else:
            lr_scheduler = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.model_save_dir = osp.expanduser('~/gcl_trial_data/')
        self.model_save_path = osp.join(self.model_save_dir, f'{secrets.token_hex(16)}.bin')
        self.best_loss = 1e20
        self.best_epoch = -1
        self.wait_window = 0
        self.trained = False
        self.train_step_cbs = []

    def register_train_step_callback(self, cb: Callable[[dict], None]):
        self.train_step_cbs.append(cb)

    def train_step(self):
        self.encoder_model.train()
        epoch_losses = []

        for data in self.train_loader:  # noqa
            data = data.to(self.device)

            self.optimizer.zero_grad()

            if data.x is None:
                num_nodes = data.batch.size()[0]
                x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            else:
                x = data.x

            model_output = self.encoder_model(
                x, data.edge_index, data.edge_attr, batch=data.batch
            )

            loss = self.contrast_model(**model_output)

            loss.backward()
            self.optimizer.step()

            self.encoder_model.update_target_encoder(0.99)

            epoch_losses.append(loss.item())

        return sum(epoch_losses) / len(epoch_losses)

    def evaluate(self):
        self.encoder_model.eval()

        x = []
        y = []
        for data in self.test_loader:  # noqa
            data = data.to(self.config.device)

            if data.x is None:
                num_nodes = data.batch.size()[0]
                input_x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            else:
                input_x = data.x

            model_output = self.encoder_model(input_x, data.edge_index, data.edge_attr, batch=data.batch)
            h1 = model_output['h1']
            h2 = model_output['h2']
            z = torch.cat([h1, h2], dim=1)
            g1 = model_output['g1']
            g2 = model_output['g2']
            g = torch.cat([g1, g2], dim=1)

            x.append(z if is_node_dataset(self.config.dataset) else g)
            y.append(data.y)

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        if self.config.dataset.startswith('ogb'):
            split = self.dataset.get_idx_split()
        elif self.config.dataset == 'WikiCS':
            data = self.dataset[0]
            train_mask = data['train_mask']
            val_mask = data['val_mask']
            test_mask = data['test_mask']
            num_folds = train_mask.size()[1]

            split = [
                {
                    'train': train_mask[:, i],
                    'valid': val_mask[:, i],
                    'test': test_mask
                }
                for i in range(num_folds)
            ]
        else:
            split = get_split(num_samples=x.size()[0])

        if isinstance(split, list):
            results = []
            for sp in split:
                evaluator = LREvaluator(mute_pbar=self.mute_pbar)
                result = evaluator.evaluate(x, y, sp)
                results.append(result)
            result = batchify_dict(results, aggr_func=lambda xs: sum(xs) / len(xs))
        elif self.config.dataset == 'ogbg-molhiv':
            def metric(y_true, y_pred):
                evaluator = Evaluator(name='ogbg-molhiv')
                return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            evaluator = GeneralLREvaluator(metric, metric_name='AUC', mute_pbar=self.mute_pbar)
            result = evaluator.evaluate(x, y, split)
        else:
            evaluator = LREvaluator(mute_pbar=self.mute_pbar)
            result = evaluator.evaluate(x, y, split)

        return result

    def run_train_loop(self):
        if self.trained:
            return

        if not self.mute_pbar:
            pbar = tqdm(total=self.config.opt.num_epochs, desc='(T)')

        for epoch in range(1, self.config.opt.num_epochs + 1):
            loss = self.train_step()
            if self.config.obj.loss == Objective.BarlowTwins:
                self.lr_scheduler.step()  # noqa
            else:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(loss)

            if not self.mute_pbar:
                pbar.set_postfix({'loss': f'{loss:.4f}', 'wait': self.wait_window, 'lr': self.optimizer_lr})
                pbar.update()

            for cb in self.train_step_cbs:
                cb({'loss': loss})

            if self.writer is not None:
                self.writer.add_scalar('loss', step=epoch, value=loss)
                self.writer.add_scalar('lr', step=epoch, value=self.optimizer_lr)

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = epoch
                self.wait_window = 0
                self.save_checkpoint()
            else:
                self.wait_window += 1

            if self.wait_window > self.config.opt.patience:
                break

        if not self.mute_pbar:
            pbar.close()

        self.trained = True
        if self.writer is not None:
            self.writer.close()

    def save_checkpoint(self):
        torch.save(self.encoder_model.state_dict(), self.model_save_path)

    def load_checkpoint(self):
        saved_state = torch.load(self.model_save_path)
        self.encoder_model.load_state_dict(saved_state)

    @property
    def optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _prepare_env(self):
        seed_everything(self.config.seed)
        os.makedirs(self.model_save_dir, exist_ok=True)
        print(f'checkpoint file {self.model_save_path}')

    def execute(self):
        self._prepare_env()
        self.run_train_loop()
        if self.config.opt.num_epochs > 0:
            self.load_checkpoint()
        result = self.evaluate()
        return result


if __name__ == '__main__':
    import pretty_errors  # noqa
    loader = ConfigLoader(model=ExpConfig, config='params/proteins.json')
    config = loader()

    printer = PrettyPrinter(indent=2)
    printer.pprint(asdict(config))

    trial = BGRLTrial(config)
    result = trial.execute()

    print("=== Final ===")
    print(f'(T): Best epoch={trial.best_epoch}, best loss={trial.best_loss:.4f}')
    print(f'(E): {result}')
