import os
import argparse

import torch
import torch.utils.data

import tqdm.autonotebook as tqdm

from model.s4 import S4
from model.s4d import S4D
from model.s4_model import S4Model
from ss_datasets import SequentialCIFAR10


def train(model, optimizer, loss_fn, dl, device):
    model.train()

    n_objects, total_loss, accuracy = 0, 0.0, 0
    for images, labels in tqdm.tqdm(dl, total=len(dl), leave=False):
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

    return total_loss / n_objects, accuracy / n_objects


def test(model, loss_fn, dl, device):
    model.eval()
    with torch.no_grad():
        n_objects, total_loss, accuracy = 0, 0.0, 0
        for images, labels in tqdm.tqdm(dl, total=len(dl), leave=False):
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

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    ds_test = SequentialCIFAR10(args.data_path, train=False, download=True)
    ds_train = SequentialCIFAR10(args.data_path, train=True, download=False)

    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # model = S4Model(
    #     d_input=3,
    #     d_output=len(ds_train.data.classes),
    #     d_model=1024,
    #     n_layers=6,
    #     dropout=0.25,
    #     prenorm=False,
    #     block_class=S4,
    #     block_kwargs={
    #         'bidirectional': True, 'postact': 'glu', 'tie_dropout': True,
    #         # 'mode': 'diag', 'measure': 'diag-lin', 'disc': 'zoh', 'real_type': 'exp',
    #         'n_ssm': 2
    #     },
    #     dropout_fn=torch.nn.Dropout1d
    # ).to(device)
    model = S4Model(
        d_input=3,
        d_output=len(ds_train.data.classes),
        d_model=128,
        n_layers=4,
        dropout=0.1,
        prenorm=False,
        block_class=S4D,
        block_kwargs={
            # 'bidirectional': True, 'postact': 'glu', 'tie_dropout': True,
            # 'mode': 'diag', 'measure': 'diag-lin', 'disc': 'zoh', 'real_type': 'exp',
            # 'n_ssm': 2
        },
        dropout_fn=torch.nn.Dropout1d
    ).to(device)

    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    all_losses_test, all_accuracies_test = [], []
    all_losses_train, all_accuracies_train = [], []
    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        _, _ = train(model, optimizer, loss_fn, dl_train, device=device)

        loss_train, accuracy_train = test(model, loss_fn, dl_train, device=device)
        loss_test, accuracy_test = test(model, loss_fn, dl_test, device=device)

        all_losses_test.append(loss_test)
        all_losses_train.append(loss_train)
        all_accuracies_test.append(accuracy_test)
        all_accuracies_train.append(accuracy_train)

        print('Epoch: {0:d}. Loss: {1:.3f}/{2:.3f}. Accuracy: {3:.3f}/{4:.3f}'.format(
            epoch, all_losses_train[-1], all_losses_test[-1], all_accuracies_train[-1], all_accuracies_test[-1]
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument(
        "--data_path",
        type=str,
        default="~/ss_datasets/",
        metavar="PATH",
        help="path to ss_datasets location (default: ~/ss_datasets/)",
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help="GPU to use"
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
        default=3,
        metavar="N",
        help="number of workers (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=101,
        metavar="N",
        help="number of epochs to train (default: 101)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )

    args = parser.parse_args()
    main(args)
