import os
import argparse
from collections import defaultdict

import regex

import wandb

import torch
import torch.utils.data

import tqdm.autonotebook as tqdm

from sequence_models.s4 import S4
from sequence_models.utils import PassthroughSequential
from sequence_models.pool import registry as pool_registry
from sequence_models.residual import registry as residual_registry
from sequence_models.model import SequenceModel, SequenceDecoder, SequenceModelWrapper

from ss_datasets import SequentialCIFAR10
from ss_datasets.lra.configure import configure_lra


class HashDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def train(model, optimizer, scheduler, loss_fn, dl, device, logger=None):
    model.train()

    n_objects, total_loss, accuracy = 0, 0.0, 0
    for idx, (images, labels, *_) in enumerate((pbar := tqdm.tqdm(dl, total=len(dl), leave=False))):
        optimizer.zero_grad()

        images = images.to(device=device)
        labels = labels.to(device=device, dtype=torch.long)
        y = model(images)
        predictions = torch.argmax(y, dim=1)

        loss = loss_fn(y, labels)
        loss.backward()
        optimizer.step()
        n_objects += predictions.shape[0]
        total_loss += loss.item() * predictions.shape[0]
        accuracy += torch.sum(torch.eq(predictions, labels)).item()

        scheduler.step()
        if logger is not None:
            for p_idx, p_lr in enumerate(scheduler.get_last_lr()):
                logger.log({f'lr/pg{p_idx}': p_lr})
            logger.log({'running_loss/train': total_loss / n_objects})
            logger.log({'running_accuracy/train': accuracy / n_objects})
        pbar.set_description('Loss: {0:.3f}. Accuracy: {1:.3f}'.format(total_loss / n_objects, accuracy / n_objects))

    return total_loss / n_objects, accuracy / n_objects


def test(model, loss_fn, dl, device):
    model.eval()
    with torch.no_grad():
        n_objects, total_loss, accuracy = 0, 0.0, 0
        for images, labels, *_ in tqdm.tqdm(dl, total=len(dl), leave=False):
            images = images.to(device=device)
            labels = labels.to(device=device, dtype=torch.long)
            y = model(images)
            predictions = torch.argmax(y, dim=1)

            loss = loss_fn(y, labels)

            n_objects += predictions.shape[0]
            total_loss += loss.item() * predictions.shape[0]
            accuracy += torch.sum(torch.eq(predictions, labels)).item()

    return total_loss / n_objects, accuracy / n_objects


def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    wandb.login(key=args.wandb_key)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    if args.dataset == 'CIFAR10':
        in_features, d_output = 3, 10

        ds_test = SequentialCIFAR10(args.data_path, train=False, download=True)
        ds_train = SequentialCIFAR10(args.data_path, train=True, download=False)

        dl_test = torch.utils.data.DataLoader(
            ds_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
        dl_train = torch.utils.data.DataLoader(
            ds_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )
    elif args.dataset.startswith('pathfinder'):
        in_features, d_output = 1, 2

        dl = configure_lra(data_dir=os.path.join(args.data_path, 'lra_release', args.dataset))
        dl_train = dl.train_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        dl_test = dl.val_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)[None]
    else:
        raise ValueError(f'Unknows dataset: {args.dataset}')

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_kwargs = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'transposed': False,
        'dropout': args.dropout,
        'tie_dropout': True,
        'prenorm': False,
        'n_repeat': 1,
        'layer': {
            'class': S4,
            'd_state': args.d_state,
            'l_max': None,
            'channels': 1,
            'bidirectional': True,
            'activation': 'gelu',
            'postact': 'glu',
            'initializer': None,
            'weight_norm': False,
            'hyper_act': None,
            'dropout': 0.0, 'tie_dropout': True,
            'bottleneck': None,
            'gate': None,
            'transposed': True,

            # SSKernel arguments
            'measure': 'legs',
            'rank': 1,
            'dt_min': 0.001,
            'dt_max': 0.1,
            'deterministic': False,
            'lr': 0.001,
            'mode': 'nplr',
            'n_ssm': args.n_ssm,

            # SSKernelNPLR:
            # keops=False,
            # real_type='exp',  # ['none' | 'exp' | 'relu' | sigmoid']
            # real_tolerance=1e-3,
            # bandlimit=None,

            # SSKernelDiag:
            # disc='bilinear',
            # real_type='exp',
            # bandlimit=None,
        },
        'residual': {
            'class': residual_registry['R'],
        },
        'norm': 'layer',
        'pool': {
            'class': pool_registry['pool'],
            'expand': None,
            'stride': 1
        },
        'track_norms': True,
        'dropinp': 0.0,
    }
    encoder_kwargs = {
        'in_features': in_features,
        'out_features': model_kwargs['d_model'],
    }
    decoder_kwargs = {
        'l_output': 0,
        'd_model': model_kwargs['d_model'],
        'd_output': d_output,
    }

    model = SequenceModelWrapper(
        PassthroughSequential(torch.nn.Linear(**encoder_kwargs)),
        PassthroughSequential(SequenceDecoder(**decoder_kwargs)),
        SequenceModel(**model_kwargs),
    ).to(device)

    # Pick output directory.
    base_dir, dataset_name = './training-runs', args.dataset
    prev_run_dirs = []
    if os.path.isdir(base_dir):
        prev_run_dirs = [
            x for x in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, x))
        ]
    prev_run_ids = [regex.match(rf'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    exp_name = '{0:0>5d}-{1}-lr{2}_nws{3}_wd{4}_ep{5}_bs{6}'.format(
        cur_run_id, dataset_name, args.lr, args.num_warmup_steps, args.wd, args.epochs, args.batch_size
    )

    log_dir = os.path.join(base_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    w_logger = wandb.init(
        dir=log_dir, project='s4', name=exp_name, config={
            'args': args, 'model': model_kwargs, 'encoder': encoder_kwargs, 'decoder': decoder_kwargs
        }
    )

    params_groups_dict = defaultdict(list)
    for parameter in model.parameters():
        opt_params = (
            getattr(parameter, '_optim') if
            hasattr(parameter, '_optim') else
            {'lr': args.lr, 'weight_decay': args.wd}
        )
        params_groups_dict[HashDict(opt_params)].append(parameter)
    params_groups = []
    for key, value in params_groups_dict.items():
        params_groups.append(key)
        params_groups[-1].update({'params': value})
    lr_lambda = (
        lambda step: (step + 1) / (args.num_warmup_steps + 1) if step < args.num_warmup_steps else 1.0
    )

    print(model)
    print('Number of trainable parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    all_losses_test, all_accuracies_test = [], []
    all_losses_train, all_accuracies_train = [], []
    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        _, _ = train(model, optimizer, scheduler, loss_fn, dl_train, device=device, logger=w_logger)

        loss_train, accuracy_train = test(model, loss_fn, dl_train, device=device)
        loss_test, accuracy_test = test(model, loss_fn, dl_test, device=device)

        all_losses_test.append(loss_test)
        all_losses_train.append(loss_train)
        all_accuracies_test.append(accuracy_test)
        all_accuracies_train.append(accuracy_train)

        print('Epoch: {0:d}. Loss: {1:.3f}/{2:.3f}. Accuracy: {3:.3f}/{4:.3f}'.format(
            epoch, all_losses_train[-1], all_losses_test[-1], all_accuracies_train[-1], all_accuracies_test[-1]
        ))
        w_logger.log({
            'loss/test': all_losses_test[-1],
            'loss/train': all_losses_train[-1],
            'accuracy/test': all_accuracies_test[-1],
            'accuracy/train': all_accuracies_train[-1]
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument(
        "--data-path",
        type=str,
        default="~/datasets/",
        metavar="PATH",
        help="path to ss_datasets location (default: ~/datasets/)",
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help="GPU to use"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        help="Target dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size (default: 50)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=1000,
        help="number of steps to warmup lr (default: 1000)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        '--wandb-key',
        type=str,
        default=None,
        help='API key for wandb'
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--d_state",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--n_ssm",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()
    main(args)
