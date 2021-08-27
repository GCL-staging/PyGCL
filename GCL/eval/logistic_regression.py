import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from GCL.eval import BaseEvaluator


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20, binary: bool = False, mute_pbar: bool = False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.mute_pbar = mute_pbar
        self.binary = binary

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)

        num_classes = y.max().item() + 1

        if self.binary:
            assert num_classes == 2, f'Binary classification only handle 2 classes, but found {num_classes}.'
            classifier = LogisticRegression(input_dim, 1).to(device)
        else:
            classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.binary:
            output_fn = lambda output: output.view(-1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            output_fn = nn.LogSoftmax(dim=-1)
            criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 1
        best_epoch = 0

        if not self.mute_pbar:
            pbar = tqdm(total=self.num_epochs, desc='(LR)',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')

        for epoch in range(self.num_epochs):
            classifier.train()
            optimizer.zero_grad()

            output = classifier(x[split['train']])
            output = output_fn(output)
            loss = criterion(output, y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                y_test = y[split['test']].detach().cpu().numpy()
                y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                test_micro = f1_score(y_test, y_pred, average='micro')
                test_macro = f1_score(y_test, y_pred, average='macro')

                y_val = y[split['valid']].detach().cpu().numpy()
                y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average='micro')

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_epoch = epoch

                if not self.mute_pbar:
                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        if not self.mute_pbar:
            pbar.close()

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_val_micro
        }
