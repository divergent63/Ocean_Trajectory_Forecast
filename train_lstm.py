from pathlib import Path
import argparse
import random
import shutil
import logging

import pandas as pd
import numpy as np
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision

# import keras
# import tensorflow as tf

import matplotlib.pyplot as plt

from models import TrajectoryGenerator, RNN

from data.loader import data_loader
import utils
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="drift", type=str)
# parser.add_argument("--dataset_name", default="zara1", type=str)
parser.add_argument("--delim", default="\t")
# parser.add_argument("--delim", default=" ")

parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=6, type=int)
parser.add_argument("--pred_len", default=4, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")

# parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--batch_size", default=8, type=int)

parser.add_argument("--num_epochs", default=251, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
#
parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
#
parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
#
parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
#
#
parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
#
parser.add_argument("--best_k", default=20, type=int)           # K=20 samples
# parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--print_every", default=100, type=int)

parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
#
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best_lstm.pth.tar")


def train(model, optimizer, train_loader, test_code=False):
    """
    parse data
    """

    losses = utils.AverageMeter("Loss", ":.6f")
    loss_lst_per_batch = []
    for batch_idx, batch in enumerate(train_loader):
        # print(batch_idx)

        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        # Forward pass
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []

        model_input = obs_traj
        # model_input = obs_traj
        pred_traj_fake = model(
            model_input
        )           # TODO: batch size disappear in pred_traj_fake. SOLVED: Because out = self.fc(out[:, -1, :])
        l2_loss_rel.append(
            l2_loss(pred_traj_fake, model_input[-args.pred_len :], loss_mask, mode="raw")
        )
        # l2_loss_rel.append(
        #     l2_loss(pred_traj_fake, model_input, loss_mask, mode="raw")
        # )
        # Backward and optimize
        optimizer.zero_grad()
        l2_loss_sum = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake.shape[0]) * (end - start)
            )
            l2_loss_sum += _l2_loss_rel

        loss += l2_loss_sum
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()

        loss_lst_per_batch.append(loss.cpu().detach().numpy()[0])
        # logging.info('loss:  ', loss)
        # logging.info('batch_idx:  %s, loss per batch:  %s', str(batch_idx), str(loss.cpu().detach().numpy()[0]))
        optimizer.step()
    return model, optimizer, loss_lst_per_batch


def validate(args, model, val_loader, epoch, writer):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch
            loss_mask = loss_mask[:, args.obs_len:]

            model_input = obs_traj_rel
            pred_traj_fake_rel = model(model_input)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def main():
    global best_ade
    best_ade = 100

    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    loss_last_ep = 1000

    model = RNN(args.obs_len, 32, 1, args.pred_len).to(device)
    # model = RNN(input_size=2, hidden_size=32, num_layers=1, num_classes=args.pred_len).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_per_epoch = []
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        logging.info('\ncurrent epoch:  %s', str(epoch))
        model, optimizer, loss_lst_per_batch = train(model, optimizer, train_loader)
        # logging.info('loss per epoch:  %s\n', str(loss_lst_per_batch[-1]))
        loss_avg = np.average(loss_lst_per_batch)
        logging.info('loss per epoch:  %s\n', str(loss_avg))
        loss_per_epoch.append(loss_avg)

        if epoch > 2 and loss_avg < loss_last_ep:
            ade = validate(args, model, val_loader, epoch, writer)
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)

            # is_best = True
            print('is best:  ', is_best)

            # save_checkpoint(
            #     {
            #         "epoch": epoch + 1,
            #         "state_dict": model.state_dict(),
            #         # "best_ade": best_ade,
            #         "best_ade": loss_lst_per_batch,
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     is_best,
            #     f"./checkpoint/checkpoint_lstm_{epoch}.pth.tar",
            # )
            # is_best = False
        loss_last_ep = loss_avg
    plt.plot(loss_per_epoch)
    plt.show()


if __name__ == '__main__':
    logging.info(
        "program start"
    )
    main()
    logging.info('complete!!')

