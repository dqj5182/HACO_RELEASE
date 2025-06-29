import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib.core.config import cfg, update_config
from lib.models.model import HACO
from lib.core.base import compute_loss
from data.dataset import MultipleDatasets
from lib.utils.train_utils import get_optim_groups, get_transform, worker_init_fn
from lib.utils.log_utils import get_datetime


parser = argparse.ArgumentParser(description='Train HACO')
parser.add_argument('--backbone', type=str, default='hamer', choices=['hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
args = parser.parse_args()


# Import dataset
for i in range(len(cfg.DATASET.train_name)):
    exec(f'from data.{cfg.DATASET.train_name[i]}.dataset import {cfg.DATASET.train_name[i]}')


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(cfg.DATASET.workers) # Limit Torch
os.environ["OMP_NUM_THREADS"] = "4" # Limit OpenMP (NumPy, MKL)
os.environ["MKL_NUM_THREADS"] = "4" # Limit MKL operations


# Initialize directories
dataset_name = "_".join([name.lower() for name in (cfg.DATASET.train_name)])
experiment_dir = os.path.join(f'experiments_train_{dataset_name}', 'full', f'exp_{get_datetime()}')
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)


# Load config
update_config(backbone_type=args.backbone, exp_dir=experiment_dir)


from lib.core.config import logger
logger.info(f"!!!!!!!!!!!!!!! Max epochs {cfg.TRAIN.epoch}")


############## Dataset ###############
transform = get_transform(args.backbone)

train_datasets = []
for i in range(len(cfg.DATASET.train_name)):
    train_datasets.append(eval(f'{cfg.DATASET.train_name[i]}')(transform, 'train'))
train_datasets = MultipleDatasets(train_datasets, make_same_len=False)
############## Dataset ###############



############# Dataloader #############
train_dataloader = DataLoader(train_datasets, batch_size=cfg.TRAIN.batch, shuffle=True, num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
total_sample_num = len(train_dataloader) * cfg.TRAIN.batch
############# Dataloader #############


logger.info(f"# of train batch: {len(train_dataloader)}")


############# Model #############
model = HACO().to(device)
############# Model #############


############# Optmizer #############
# Optimization group
optim_groups = get_optim_groups(model)

# Optimizer
optimizer = torch.optim.AdamW(optim_groups, lr=cfg.TRAIN.lr, betas=cfg.TRAIN.betas, weight_decay=cfg.TRAIN.weight_decay)

# Scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.milestones, gamma=cfg.TRAIN.gamma)
############# Optmizer #############


############################### Train Loop ###############################
best_checkpoint_path = ''
start_epoch = 0
global_step = 0


for epoch in range(start_epoch, cfg.TRAIN.epoch): # loop over the dataset multiple times
    # Make dataloader as iterator
    train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)

    # Make model trainable
    torch.set_grad_enabled(True)
    model.train()

    # Iterate over samples to train the model
    for idx, data in train_iterator:
        ############# Run model #############
        outputs = model({'input': data['input_data'], 'target': data['targets_data'], 'meta_info': data['meta_info']}, mode="train")
        ############# Run model #############

        ############# Loss Function #############
        train_loss, loss_dict = compute_loss(outputs, data['targets_data'], epoch)
        ############# Loss Function #############

        train_iterator.set_description(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Mesh: {loss_dict['contact_mesh_loss']:.3f} | Mesh_336: {loss_dict['contact_336_loss']:.3f} | Mesh_84: {loss_dict['contact_84_loss']:.3f} | Joint: {loss_dict['contact_joint_loss']:.3f} | Reg: {loss_dict['regularization_loss']:.3f} | S-Reg: {loss_dict['smooth_regularization_loss']:.3f}")

        ############# Training process #############
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        ############# Training process #############


    # scheduler_hand.step()
    scheduler.step()

    global_step += 1

    logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Mesh: {loss_dict['contact_mesh_loss']:.3f} | Mesh_336: {loss_dict['contact_336_loss']:.3f} | Mesh_84: {loss_dict['contact_84_loss']:.3f} | Joint: {loss_dict['contact_joint_loss']:.3f} | Reg: {loss_dict['regularization_loss']:.3f} | S-Reg: {loss_dict['smooth_regularization_loss']:.3f}")
    ############################### Training Loop ###############################

    ############# Save model checkpoint ############# (TODO: SAVE CHECKPOINT FOR EPOCH ONLY WITH IMPROVEMENT)
    if epoch % cfg.TRAIN.print_freq == 0 or epoch == (cfg.TRAIN.epoch - 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"haco_full_epoch{epoch}.ckpt")

        checkpoint_out = {
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        torch.save(checkpoint_out, checkpoint_path)

        best_checkpoint_path = checkpoint_path
        logger.info(f"Model trained, best model path: {best_checkpoint_path}")
    ############# Save model checkpoint #############
############################### Train Loop ###############################


# Let us know that training is finally finished
logger.info('Model Training Finished!!!!!')