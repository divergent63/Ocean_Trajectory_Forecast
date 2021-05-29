"""

"""
from pathlib import Path
import argparse
import random
import shutil
import logging

import os, sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision

# import keras
#
# import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

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
parser.add_argument("--dset_type", default="test", type=str)

parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=6, type=int)
parser.add_argument("--pred_len", default=4, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--num_samples", default=20, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--num_epochs", default=2, type=int)

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
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
#
parser.add_argument(
    "--resume",
    default="./checkpoint/checkpoint158.pth.tar",
    # default="./checkpoint/checkpoint_lstm_215.pth.tar",
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


def evaluate_helper(error, seq_start_end, model_output_traj, model_output_traj_best):
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        min_index = _error.min(0)[1].item()
        model_output_traj_best[:, start:end, :] = model_output_traj[min_index][
            :, start:end, :
        ]
    return model_output_traj_best


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    de = pred_traj_gt.permute(1, 0, 2) - pred_traj_fake.permute(1, 0, 2)
    return ade, fde


def get_generator(checkpoint):
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def plot_trajectory(args, loader, generator):
    ground_truth_input = []
    all_model_output_traj = []
    ground_truth_output = []
    pic_cnt = 0

    traj_arr_lst_all = []
    with torch.no_grad():
        for bat_id, batch in enumerate(loader):
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
            ade = []
            ground_truth_input.append(obs_traj)
            ground_truth_output.append(pred_traj_gt)
            model_output_traj = []
            model_output_traj_best = torch.ones_like(pred_traj_gt).cuda()

            for _ in range(args.num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj_rel, obs_traj, seq_start_end, 0, 3
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                model_output_traj.append(pred_traj_fake)
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
            model_output_traj_best = evaluate_helper(
                ade, seq_start_end, model_output_traj, model_output_traj_best
            )
            all_model_output_traj.append(model_output_traj_best)

            traj_list = []

            for idx, (start, end) in enumerate(seq_start_end):
                # plt.figure(figsize=(16,9), dpi=300)
                ground_truth_input_x_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_input_y_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                ground_truth_output_x_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_output_y_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                model_output_x_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                model_output_y_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 1].T
                )

                for i in range(ground_truth_output_x_piccoor.shape[0]):

                    traj_list.append(np.concatenate([list(ground_truth_input_x_piccoor[i, :]),
                                      list(ground_truth_output_x_piccoor[i, :]),
                                      list(model_output_x_piccoor[i, :]),
                                      list(ground_truth_input_y_piccoor[i, :]),
                                      list(ground_truth_output_y_piccoor[i, :]),
                                      list(model_output_y_piccoor[i, :])
                                      ]))

                pic_cnt += 1

            traj_arr = np.reshape(traj_list, (-1, args.pred_len*4+args.obs_len*2))
            xin_true_key_list = ['observed input x_%d'%int(i+1) for i in range(args.obs_len)]
            xout_true_key_list = ['ground truth output xt_%d'%int(i+1) for i in range(args.pred_len)]
            xout_pred_key_list = ['predicted output xp_%d'%int(i+1) for i in range(args.pred_len)]
            yin_true_key_list = ['observed input y_%d'%int(i+1) for i in range(args.obs_len)]
            yout_true_key_list = ['ground truth output yt_%d'%int(i+1) for i in range(args.pred_len)]
            yout_pred_key_list = ['predicted output yp_%d'%int(i+1) for i in range(args.pred_len)]
            key_list = np.concatenate(
                [xin_true_key_list,
                 xout_true_key_list,
                 xout_pred_key_list,
                 yin_true_key_list,
                 yout_true_key_list,
                 yout_pred_key_list]
            )

            traj_df = pd.DataFrame(traj_arr, columns=key_list)
            traj_df_csv = traj_df
            traj_df_csv.to_csv(",/stgat/traj_test_%d.csv" % bat_id)

            traj_arr_lst_all.append(traj_arr)


def visualize(args):
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)
    print("path: \n" + path)

    _, loader = data_loader(args, path)
    plot_trajectory(args, loader, generator)


if __name__ == '__main__':
    logging.info(
        "program start"
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        os.mkdir("./traj_fig_stgat")
    except FileExistsError:
        print("file/dir already exists!")

    visualize(args)
    print('complete!!')
