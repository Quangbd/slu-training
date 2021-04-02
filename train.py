import os
import time
import torch
import argparse
from data import *
import pandas as pd
from glob import glob
import torch.nn as nn
from datetime import datetime
import torch.utils.data as data
from sklearn.metrics import f1_score
from transformers import Wav2Vec2Config
from model import SLUModel


def set_seed(seed: int = 54):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = config.output_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = os.path.join(self.base_dir, 'log.txt')
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device
        self.accumulate_steps = config.accumulate_steps
        self.model.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **dict(T_max=config.n_epochs))
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader, test_loader):
        for e in range(self.config.n_epochs):
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, '
                     f'summary_loss: {summary_loss.avg:.5f}, '
                     f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            if (self.epoch + 1) % self.config.val_freq == 0:
                t = time.time()
                summary_loss, score = self.validation(self.model, validation_loader)

                self.log(f'[RESULT]: Val. Epoch: {self.epoch}, '
                         f'summary_loss: {summary_loss.avg:.5f}, '
                         f'acc score: {score:.5f}, '
                         f'time: {(time.time() - t):.5f}')
                if summary_loss.avg < self.best_summary_loss:
                    self.best_summary_loss = summary_loss.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/best-checkpoint-{self.epoch:06d}epoch.bin')
                    for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                        os.remove(path)

                summary_loss, score = self.validation(self.model, test_loader)
                self.log(f'[RESULT]: Test. Epoch: {self.epoch}, '
                         f'summary_loss: {summary_loss.avg:.5f}, '
                         f'acc score: {score:.5f}, '
                         f'time: {(time.time() - t):.5f}')

            if self.config.validation_scheduler:
                self.scheduler.step()

            self.epoch += 1

    def validation(self, model, val_loader):
        model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        running_corrects = 0.0

        for step, (audios, labels) in enumerate(val_loader):
            print(f'Val Step {step}/{len(val_loader)}, '
                  f'summary_loss: {summary_loss.avg:.5f}, '
                  f'time: {(time.time() - t):.5f}', end='\r')
            with torch.no_grad():
                audios = audios.to(self.device).float()
                labels = labels.to(self.device).float()
                batch_size = audios.shape[0]
                output = self.model(audios)
                loss = self.loss_fn(output, labels)

                preds = torch.sigmoid(output).data > 0.5
                preds = preds.to(torch.float32)
                # print(output.shape)
                # preds = np.argmax(output.cpu().numpy(), axis=1)
                # pred_results.extend(list(preds.cpu().to(torch.float).numpy()))
                # print(preds)
                # origin_labels.extend(list(labels.cpu().numpy()))
                running_corrects += f1_score(labels.to("cpu").to(torch.int).numpy(),
                                             preds.to("cpu").to(torch.int).numpy(), average="samples") * batch_size

                summary_loss.update(loss.detach().item(), batch_size)

        # print(origin_labels)
        # print(pred_results)
        score = running_corrects / len(val_loader.dataset)

        return summary_loss, score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        self.optimizer.zero_grad()  # very important
        for step, (audios, labels) in enumerate(train_loader):
            print(f'Train Step {step}/{len(train_loader)}, '
                  f'summary_loss: {summary_loss.avg:.5f}, '
                  f'time: {(time.time() - t):.5f}', end='\r')

            audios = audios.to(self.device).float()
            labels = labels.to(self.device).float()
            batch_size = audios.size()[0]

            output = self.model(audios)
            # labels = labels.squeeze(1)
            # print(output, labels)
            loss = self.loss_fn(output, labels)
            loss.backward()
            if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                self.optimizer.step()  # Now we can do an optimizer step
                self.optimizer.zero_grad()

            if step % 50 == 0:
                print(f'{step}/{len(train_loader)},loss={summary_loss.avg}')

            summary_loss.update(loss.detach().item(), batch_size)

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def run_training(net, config, device):
    net.to(device)
    train_df = pd.read_csv(os.path.join(config.data_dir, 'data/train_data.csv'))
    val_df = pd.read_csv(os.path.join(config.data_dir, 'data/valid_data.csv'))
    test_df = pd.read_csv(os.path.join(config.data_dir, 'data/test_data.csv'))

    train_dataset = SLUDataset(config, train_df, transform=get_train_transforms())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        num_workers=config.num_workers)

    validation_dataset = SLUDataset(config, val_df, test=True)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False)

    test_dataset = SLUDataset(config, test_df, test=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False)

    fitter = Fitter(model=net, device=device, config=config)
    fitter.fit(train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000005)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--step_scheduler', type=bool, default=False)
    parser.add_argument('--validation_scheduler', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--val_freq', type=int, default=1)

    parser.add_argument('--data_dir', type=str,
                        default='data/fluent_speech_commands_dataset')
    parser.add_argument('--output_dir', type=str,
                        default='data/model/slu')
    parser.add_argument('--pretrained_dir', type=str,
                        default='data/model/wav2vec2-base-960h')
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-960h')

    args = parser.parse_args()
    set_seed(args.seed)
    _config = Wav2Vec2Config.from_pretrained(args.pretrained_model, cache_dir=args.pretrained_dir)
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _net = SLUModel(_config, args.pretrained_model, args.pretrained_dir).to(_device)

    run_training(_net, args, _device)
