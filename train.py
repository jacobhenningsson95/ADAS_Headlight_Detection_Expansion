import config as c
from collections import OrderedDict
from torch.cuda.amp import autocast
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.loss import calc_loss
from model import PoseNet
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.log_helper import init_logger
from dataset import Dataset
import argparse
import warnings
import os
import matplotlib
from utils.vis_utils import visualize_iteration, initialize_visualizations
from torch.utils.data import DataLoader, WeightedRandomSampler


# for command line
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=False, type=int, metavar='N', help='Specify if cuda')
    parser.add_argument('--reload', default=None, type=str, metavar='N', help='continue training model')
    parser.add_argument('--visualizations', default=False, type=bool, metavar='N',
                        help='Enables visualisations in the training loop')
    args = parser.parse_args()
    return args


def save_checkpoint(model, optimizer, epoch, config):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config
    }

    path = os.path.join('Trained model', 'model_epoch%s.pth' % epoch)
    torch.save(state, path)


def load_checkpoint(filename, config, device):
    path = os.path.join('Trained model', filename)
    checkpoint = torch.load(path, map_location=device)

    saved_config = checkpoint["config"]
    model = PoseNet(saved_config['nstack'], saved_config['inp_dim'], saved_config['oup_dim'], bn=saved_config['bn'],
                    increase=saved_config['increase'])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    config['nstack'] = saved_config['nstack']

    if not config["override_saved_config"]:
        config = saved_config

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, config['learning_rate'])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]

    return config, model, optimizer, start_epoch


def initialize_model(config, device):
    model = PoseNet(config['nstack'], config['inp_dim'], config['oup_dim'], bn=config['bn'],
                    increase=config['increase'])
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, config['learning_rate'])

    start_epoch = 0

    return model, optimizer, start_epoch


def data_gen(config, Night_only=True):

    train_dataset = Dataset(config["dataset_path"], "train/images", "train/labels/keypoints", size=config['input_res'], max_num_car=config['max_num_car'],
                            max_num_light=config['max_num_light'], output_res=config['output_res'],
                            negative_samples=config["negative_samples"], day_samples=config["day_samples"])
    val_dataset = Dataset(config["dataset_path"], "val/images", "val/labels/keypoints", size=config['input_res'], max_num_car=config['max_num_car'],
                          max_num_light=config['max_num_light'], output_res=config['output_res'],
                          negative_samples=config["negative_samples"])

    if config["weighted_dataset"]:
        sample_weights  = train_dataset.generate_weights()
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True,
            pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    return train_loader, val_loader


def train(config, train_loader, model, criterion, optimizer, device, plot_dict=None):
    scaler = torch.cuda.amp.GradScaler()
    train_loss = []
    for img, img_name, seg_maps, instance_maps, instance_count, semantic_seg in tqdm(train_loader):

        img = img.to(device)
        seg_maps = seg_maps.to(device)
        instance_maps = instance_maps.to(device)  # 16 20 128 128
        instance_count = instance_count.to(device)
        semantic_seg = semantic_seg.to(device)

        with autocast(enabled=config['autocast']):
            output = model(img)  # torch.Size([batch_size, n_stack, output_dim, 128, 128])
            result = criterion(output, instance_maps=instance_maps, instance_count=instance_count, semantic_seg=semantic_seg)

        loss = 0
        for i in result:
            loss = loss + torch.mean(i)

        # compute gradient and do optimizing step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        if plot_dict is not None:
            visualize_iteration(output, img, instance_maps, loss, plot_dict, semantic_seg=semantic_seg, seg_maps=seg_maps, instance_count=instance_count)

    train_loss_mean = np.mean(train_loss)

    return OrderedDict([('loss', train_loss_mean)])


def validate(config, val_loader, model, criterion, device):
    # switch to evaluate mode
    model.eval()
    val_loss = []
    with torch.no_grad():

        for img, img_name, seg_maps, instance_maps, instance_count, semantic_seg in tqdm(val_loader):
            img = img.to(device)
            instance_maps = instance_maps.to(device)
            instance_count = instance_count.to(device)
            semantic_seg = semantic_seg.to(device)
            with autocast(enabled=config['autocast']):
                output = model(img)  # torch.Size([batch_size, n_stack, output_dim, 128, 128])
            result = criterion(output, instance_maps=instance_maps, instance_count=instance_count, semantic_seg=semantic_seg)

            loss = 0
            for i in result:
                loss = loss + torch.mean(i)

            val_loss.append(loss.item())

    val_loss_mean = np.mean(val_loss)

    return OrderedDict([('loss', val_loss_mean)])


def seed_and_settings(seed_value=46, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = c.__config__
    warnings.filterwarnings("ignore")
    matplotlib.use("TkAgg")
    if args.visualizations:
        matplotlib.use("TkAgg")
        plot_dict = initialize_visualizations()
    else:
        plot_dict = None
    # seed everything
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    Logger = init_logger(log_file=config['log_path'])
    return Logger, plot_dict, device, config


def main():
    args = parse_args()
    matplotlib.use("TkAgg")

    Logger, plot_dict, device, config = seed_and_settings(args=args)
    warnings.filterwarnings("ignore")

    if args.reload is not None:
        config, model, optimizer, start_epoch = load_checkpoint(args.reload, config, device)
    else:
        model, optimizer, start_epoch = initialize_model(config, device)

    criterion = calc_loss(config)
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    else:
        raise NotImplementedError

    # setting data loader
    train_loader, val_loader = data_gen(config)
    # logger
    best_val_loss = np.inf

    for epoch in range(start_epoch+1, config['epochs']):
        Logger.info('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train
        train_log = train(config, train_loader, model, criterion, optimizer, device, plot_dict)
        # log['train_loss'].append(train_log['loss'])
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        # Validate the model every "config['val_epoch']" epoch
        if epoch % config['val_epoch'] == 0:
            val_log = validate(config, val_loader, model, criterion, device)
            print('training loss %.8f - val_loss %.8f' % (train_log['loss'], val_log['loss']))
            # log['val_loss'].append(val_log['loss'])
        else:
            print('training loss %.8f' % (train_log['loss']))
            val_log = OrderedDict()
            val_log['loss'] = float('inf')
            # log['val_loss'].append(val_log['loss'])

        Logger.info(
            f'Epoch {epoch} - avg_train_loss: {train_log["loss"]:.8f}  avg_val_loss: {val_log["loss"]:.8f}')

        if val_log['loss'] < best_val_loss and not config['save_all_models']:
            if not os.path.exists('Trained model'):
                os.makedirs('Trained model')
            save_checkpoint(model, optimizer, epoch, config)
            best_val_loss = val_log['loss']
            Logger.info("\n==========\n==> saved best model\n==========\n")

        if config['save_all_models']:
            if not os.path.exists('Trained model'):
                os.makedirs('Trained model')
            save_checkpoint(model, optimizer, epoch, config)

        # early stopping
        # TBA
        torch.cuda.empty_cache()


if __name__ == '__main__':

    main()