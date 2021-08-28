import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from GCL.eval import BaseEvaluator


class MLPRegressor(nn.Module):
    def __init__(self, num_features, num_hidden, num_targets):
        super(MLPRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_targets)
        )

        self.apply(
            lambda layer:
            torch.nn.init.xavier_uniform_(layer.weight.data) if isinstance(layer, torch.nn.Linear) else None
        )

    def forward(self, x):
        z = self.mlp(x)
        return z


class MLPRegEvaluator(BaseEvaluator):
    def __init__(self,
                 metric,
                 metric_name: str,
                 num_epochs: int = 5000,
                 learning_rate: float = 0.01, weight_decay: float = 0.0,
                 test_interval: int = 20,
                 mute_pbar: bool = False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.mute_pbar = mute_pbar
        self.mute_pbar = False
        self.metric = metric
        self.metric_name = metric_name

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)

        net = MLPRegressor(input_dim, input_dim, 1).to(device)
        optimizer = Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = lambda x: x.view(-1)
        criterion = nn.L1Loss()

        best_val_metric = 1e100
        best_test_metric = 1e100
        best_epoch = 0

        if not self.mute_pbar:
            pbar = tqdm(total=self.num_epochs, desc='(LR)',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')

        for epoch in range(self.num_epochs):
            net.train()
            optimizer.zero_grad()

            output = net(x[split['train']])
            output = output_fn(output)
            loss = criterion(output, y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                net.eval()
                y_test = y[split['test']].detach().cpu()
                y_pred = net(x[split['test']]).detach().cpu()

                test_metric = self.metric(y_test, y_pred)[self.metric_name]

                y_val = y[split['valid']].detach().cpu()
                y_pred = net(x[split['valid']]).detach().cpu()

                val_metric = self.metric(y_val, y_pred)[self.metric_name]

                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                    best_epoch = epoch

                if not self.mute_pbar:
                    pbar.set_postfix({f'test': best_test_metric, 'val': best_val_metric})
                    pbar.update(self.test_interval)

        if not self.mute_pbar:
            pbar.close()

        return {
            self.metric_name: best_test_metric
        }
