import argparse
import logging
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch
import random

from utils.metrics import *
from utils.util import viz_results, print_info
from utils.logger import Logger
from utils.visualization import Visualizer

from models.encoder.trajectory import *
from dataset import *

parser = argparse.ArgumentParser()

parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=10,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum of lr')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight_decay on l2 reg')
parser.add_argument('--lr_sh_rate', type=int, default=100,
                    help='number of steps to drop the lr')
parser.add_argument('--milestones', type=int, default=[50, 100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='sgcn',
                    help='personal tag for the model ')
parser.add_argument("--mode", type=str, default='intention',
                    choices=['intention', 'addressor_warm', 'addressor', 'trajectory'], help='Stage of training.')
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()

print("Training initiating....")
print(args)


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}

writer = SummaryWriter('./log' + '/' + args.tag + '/' + args.dataset)


def train(epoch, model, optimizer, checkpoint_dir, loader_train, use_wandb=False, VISUALIZE=False):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
    if use_wandb:
        logger = Logger("train", project='star', viz_backend="wandb")
    else:
        logger = logging.Logger("train")

    vis = Visualizer(mode='plot')

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr, velocity_x, velocity_y = batch

        # obs_traj observed absolute coordinate [1 N 2 obs_len]
        # pred_traj_gt ground truth absolute coordinate [1 N 2 pred_len]
        # obs_traj_rel velocity of observed trajectory [1 N 2 obs_len]
        # pred_traj_gt_rel velocity of ground-truth [1 N 2 pred_len]
        # non_linear_ped 0/1 tensor indicated whether the trajectory of pedestrians n is linear [1 N]
        # loss_mask 0/1 tensor indicated whether the trajectory point at time t is loss [1 N obs_len+pred_len]
        # V_obs input graph of observed trajectory represented by velocity  [1 obs_len N 3]
        # V_tr target graph of ground-truth represented by velocity  [1 pred_len N 2]
        #    velocity_x [1,N,12]
        # identity_spatial = torch.ones((V_obs.shape[1]+1, V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
        #                    torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
        # identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1]+1, V_obs.shape[1]+1), device='cuda') * \
        #                     torch.eye(V_obs.shape[1]+1, device='cuda')  # [N obs_len obs_len]
        # identity = [identity_spatial, identity_temporal]
        identity_obs = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                       torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
        identity_pred = torch.ones((V_tr.shape[1], V_tr.shape[2], V_tr.shape[2]), device='cuda') * \
                        torch.eye(V_tr.shape[2], device='cuda')  # [pred_len N N]
        optimizer.zero_grad()

        pred_goal, pred_traj, loss_dict = model(V_obs, identity_obs, obs_traj_rel, V_tr, identity_pred, \
                                                pred_traj_gt_rel, velocity_x, velocity_y)  # A_obs <8, #, #>

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld'] + 0.5 * loss_dict['loss_velo']
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
        writer.add_scalar('Loss/train', loss_batch / batch_count, epoch)
        if use_wandb and batch_count % args.batch_size == 0:
            print_info(epoch, model, optimizer, loss_dict, logger)
            if hasattr(logger, 'log_values'):
                logger.log_values({
                    'Loss': loss_batch / batch_count,
                })

    metrics['train_loss'].append(loss_batch / batch_count)

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val, use_wandb=False, VISUALIZE=False):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    if use_wandb:
        logger = Logger("train", project='star', viz_backend="wandb")
    else:
        logger = logging.Logger("train")

    vis = Visualizer(mode='plot')
    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr, velocity_x, velocity_y = batch

        with torch.no_grad():
            identity_obs = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
                V_obs.shape[2])
            identity_pred = torch.ones((V_tr.shape[1], V_tr.shape[2], V_tr.shape[2])) * \
                            torch.eye(V_tr.shape[2])  # [pred_len N N]

            identity_obs = identity_obs.cuda()
            identity_pred = identity_pred.cuda()
            pred_goal, pred_traj, loss_dict = model(V_obs, identity_obs, obs_traj_rel, V_tr, identity_pred,
                                                    pred_traj_gt_rel, \
                                                    velocity_x, velocity_y)  # A_obs <8, #, #>

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld'] + loss_dict['loss_velo']
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

            if VISUALIZE:
                pred_gt = pred_traj_gt.permute(0, 1, 3, 2)
                pred_gt = pred_gt.squeeze()
                pred_gt = pred_gt.data.cpu().numpy().copy()
                obs = obs_traj.permute(0, 1, 3, 2)
                obs = obs.squeeze()
                obs = obs.data.cpu().numpy().copy()
                pred_tr = []
                V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

                for k in range(20):
                    V_pred = pred_traj[:, :, k, :].permute(1, 0, 2)
                    V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                               V_x[-1, :, :].copy())
                    pred_tr.append(V_pred_rel_to_abs)

                pred_tr = torch.Tensor(np.array(pred_tr))
                pred_tr = pred_tr.permute(2, 1, 0, 3)
                pred_tr = pred_tr.data.cpu().numpy().copy()
                viz_results(vis, obs, pred_gt, pred_tr, dist_goal=None, dist_traj=None, name='pred_test', logger=logger, \
                            normalized=False, img_path=None)
            writer.add_scalar('Loss/val', loss_batch / batch_count, epoch)
    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):
    train_data_path = './data/processed/{}_train.pkl'.format(args.dataset)
    val_data_path = './data/processed/{}_val.pkl'.format(args.dataset)
    with open(train_data_path, 'rb') as f:
        dataset_train = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        dataset_val = pickle.load(f)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)

    loader_val = DataLoader(
        dataset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=1)

    print('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    model = TrajectoryModel(args, number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0.1,
                            n_tcn=5, latent_dim=64, K=200, DEC_WITH_Z=True).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)
    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)
        vald(epoch, model, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
